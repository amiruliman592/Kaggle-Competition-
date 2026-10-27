"""
download_gong.py — Phase 6 M3: unlabelled GONG H-alpha images for self-training.

Source: NSO GONG archive, product "hag" = the competition's JPEGs (same frames, same
names YYYYMMDDhhmmss<station>h, 2048x2048; checked 2026-10-27 on 3 train files:
corr >= 0.99, mean |diff| ~2 grey levels; the archive copy is RGB at JPEG quality ~75,
the competition copy grey at ~90).
  https://gong2.nso.edu/HA/hag/YYYYMM/YYYYMMDD/<stem>.jpg

Exclusion (stricter than the host's +-1 h rule, HANDOVER §3):
  - every day of a 27-day val time block (same blocks as splits/val_ids.json), so
    val stays an honest check: filaments live for days, a near-copy would leak;
  - every test day +-1 day.
Days are drawn from the train years only (2011-2022), one frame per day, stations
B C L M T U balanced (the archive's "S" frames are not in the competition data),
a random frame from the middle 80% of that station's day (avoids low-sun frames).

Steps (resumable: rerun the same command after a disconnect):
  python src/external/download_gong.py manifest --n 3000 --out external/manifest.csv
  python src/external/download_gong.py download --manifest external/manifest.csv --out-dir <dir>
download keeps images within the train images' brightness / disk-size / sharpness
range in <dir>/images/ and moves the rest to <dir>/rejected/ (cloudy, cut, blurred);
qc.csv lists the numbers.
  python src/external/download_gong.py valcopies --out-dir <dir>   # archive copies of the val images
"""

import argparse
import csv
import datetime as dt
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CFG  # noqa: E402

BASE_URL = "https://gong2.nso.edu/HA/hag"
STATIONS = "BCLMTU"
BLOCK_DAYS = 27                       # = CFG.data.SPLIT_BLOCK_DAYS (split.time_block)
YEARS = (2011, 2022)                  # train years
FRAME_RE = re.compile(r'href="(\d{14}([A-Z])h\.jpg)"')
FIELDS = ["file", "timestamp", "station", "url"]
WORKERS = 4                           # parallel requests (polite to the archive)


def day_of(stem: str) -> dt.date:
    return dt.date(int(stem[:4]), int(stem[4:6]), int(stem[6:8]))


def block_of(day: dt.date) -> int:
    """Same block index as split.time_block."""
    return day.toordinal() // BLOCK_DAYS


def excluded_days(val_ids: Iterable[str], test_stems: Iterable[str],
                  start: dt.date, end: dt.date) -> Set[dt.date]:
    """Every day in [start, end] that external images must not come from."""
    val_blocks = {block_of(day_of(s)) for s in val_ids}
    near_test = {day_of(s) + dt.timedelta(k) for s in test_stems for k in (-1, 0, 1)}
    days = (start + dt.timedelta(k) for k in range((end - start).days + 1))
    return {d for d in days if block_of(d) in val_blocks or d in near_test}


def allowed_days(val_ids, test_stems, years=YEARS) -> List[dt.date]:
    start, end = dt.date(years[0], 1, 1), dt.date(years[1], 12, 31)
    bad = excluded_days(val_ids, test_stems, start, end)
    return [start + dt.timedelta(k) for k in range((end - start).days + 1)
            if start + dt.timedelta(k) not in bad]


def load_exclusion_ids():
    val_ids = json.loads((CFG.paths.ROOT / "splits" / "val_ids.json").read_text())["val_ids"]
    test_stems = [p.stem for p in CFG.paths.TEST_IMG_DIR.glob("*.jpeg")]
    assert len(test_stems) == 180, f"expected 180 test images, found {len(test_stems)}"
    return val_ids, test_stems


def day_url(day: dt.date) -> str:
    return f"{BASE_URL}/{day:%Y%m}/{day:%Y%m%d}/"


def parse_listing(html: str) -> Dict[str, List[str]]:
    """Directory page -> {station: [file names, time order]} for the competition stations."""
    frames: Dict[str, List[str]] = {}
    for name, station in FRAME_RE.findall(html):
        if station in STATIONS:
            frames.setdefault(station, []).append(name)
    return {k: sorted(v) for k, v in frames.items()}


def pick_frame(frames: Dict[str, List[str]], counts: Dict[str, int], rng: random.Random) -> Optional[str]:
    """Least-used station of the day (ties random), then a frame from the middle 80% of its day."""
    if not frames:
        return None
    least = min(counts[s] for s in frames)
    station = rng.choice(sorted(s for s in frames if counts[s] == least))
    names = frames[station]
    lo, hi = int(len(names) * 0.1), max(int(len(names) * 0.9), int(len(names) * 0.1) + 1)
    counts[station] += 1
    return rng.choice(names[lo:hi])


def fetch(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "solar-filament-research/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # 404 = no data that day: no retry
            if getattr(e, "code", None) == 404:
                return b""
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return b""


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_manifest(n: int, out: Path, seed: int = 42, chunk: int = 40):
    val_ids, test_stems = load_exclusion_ids()
    days = allowed_days(val_ids, test_stems)
    random.Random(seed).shuffle(days)
    rows = read_rows(out)
    checked_path = out.with_name(out.stem + "_checked_days.txt")   # days already looked at (resume)
    checked = set(checked_path.read_text().split()) if checked_path.exists() else set()
    counts = {s: 0 for s in STATIONS}
    for r in rows:
        counts[r["station"]] += 1
    todo = [d for d in days if f"{d:%Y%m%d}" not in checked]
    print(f"[Manifest] {len(days)} allowed days, {len(rows)} rows so far, {len(todo)} days unchecked")
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    with open(out, "a", newline="") as f, open(checked_path, "a") as fc, ThreadPoolExecutor(WORKERS) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i in range(0, len(todo), chunk):
            if len(rows) >= n:
                break
            batch = todo[i:i + chunk]
            pages = pool.map(lambda d: fetch(day_url(d)).decode("utf-8", "replace"), batch)
            for day, html in zip(batch, pages):        # in order + per-day seed -> same picks after a resume
                rng = random.Random(f"{seed}-{day:%Y%m%d}")
                name = pick_frame(parse_listing(html), counts, rng) if len(rows) < n else None
                if name:
                    row = {"file": name, "timestamp": name[:14], "station": name[14],
                           "url": day_url(day) + name}
                    w.writerow(row)
                    rows.append(row)
                fc.write(f"{day:%Y%m%d}\n")
            f.flush()
            fc.flush()
            print(f"[Manifest] {len(rows)}/{n} frames, stations {counts}")
    print(f"[Manifest] done: {len(rows)} rows -> {out}")


# ── download + quality check ────────────────────────────────────────────

def qc_stats(gray: np.ndarray) -> Dict[str, float]:
    """Disk size, disk brightness and sharpness (Laplacian variance, centre crop)."""
    import cv2

    disk = gray > 20
    c = gray[768:1280, 768:1280]
    return {"disk_frac": float(disk.mean()),
            "disk_mean": float(gray[disk].mean()) if disk.any() else 0.0,
            "sharpness": float(cv2.Laplacian(c, cv2.CV_64F).var())}


def qc_bounds(paths: List[Path], lo: float = 0.5, hi: float = 99.5, margin: float = 0.1) -> Dict[str, tuple]:
    """Range of each statistic over the train images (percentiles, widened by `margin`)."""
    import cv2

    stats = [qc_stats(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in paths]
    out = {}
    for k in stats[0]:
        v = np.array([s[k] for s in stats])
        a, b = np.percentile(v, [lo, hi])
        out[k] = (float(a - margin * (b - a)), float(b + margin * (b - a)))
    return out


def qc_ok(stats: Dict[str, float], bounds: Dict[str, tuple]) -> bool:
    return all(bounds[k][0] <= v <= bounds[k][1] for k, v in stats.items())


def download(manifest: Path, out_dir: Path):
    import cv2

    rows = read_rows(manifest)
    img_dir, rej_dir = out_dir / "images", out_dir / "rejected"
    img_dir.mkdir(parents=True, exist_ok=True)
    rej_dir.mkdir(parents=True, exist_ok=True)
    qc_path = out_dir / "qc.csv"
    done = {r["file"] for r in read_rows(qc_path)}
    todo = [r for r in rows if r["file"] not in done]
    print(f"[Download] {len(rows)} in manifest, {len(done)} done, {len(todo)} to go")
    train_paths = sorted(CFG.paths.TRAIN_IMG_DIR.glob("*.jpeg"))
    bounds = qc_bounds(train_paths)
    print("[Download] train QC range:", {k: tuple(round(x, 2) for x in v) for k, v in bounds.items()})
    fields = ["file", "ok", "disk_frac", "disk_mean", "sharpness"]
    new_file = not qc_path.exists()
    with open(qc_path, "a", newline="") as f, ThreadPoolExecutor(WORKERS) as pool:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        n_ok = 0
        for k, (row, data) in enumerate(zip(todo, pool.map(lambda r: fetch(r["url"]), todo))):
            gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE) if data else None
            if gray is None or gray.shape != (2048, 2048):
                w.writerow({"file": row["file"], "ok": 0, "disk_frac": 0, "disk_mean": 0, "sharpness": 0})
                continue
            stats = qc_stats(gray)
            ok = qc_ok(stats, bounds)
            n_ok += ok
            (img_dir if ok else rej_dir).joinpath(row["file"]).write_bytes(data)
            w.writerow({"file": row["file"], "ok": int(ok), **{k: round(v, 3) for k, v in stats.items()}})
            if (k + 1) % 100 == 0:
                f.flush()
                print(f"[Download] {k + 1}/{len(todo)} ({n_ok} kept)")
    kept = sum(int(r["ok"]) for r in read_rows(qc_path))
    print(f"[Download] done: {kept} images kept in {img_dir}, rest in {rej_dir}")


def val_copies(out_dir: Path):
    """Archive copies of the 141 val images (same frames): measures what the lower JPEG quality costs."""
    val_ids, _ = load_exclusion_ids()
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in val_ids if not (out_dir / f"{s}.jpg").exists()]
    with ThreadPoolExecutor(WORKERS) as pool:
        for stem, data in zip(todo, pool.map(lambda s: fetch(day_url(day_of(s)) + f"{s}.jpg"), todo)):
            if data:
                (out_dir / f"{stem}.jpg").write_bytes(data)
    n = len(list(out_dir.glob("*.jpg")))
    print(f"[Val copies] {n}/{len(val_ids)} archive copies in {out_dir}")


def main():
    p = argparse.ArgumentParser(description="GONG H-alpha external images (Phase 6 M3)")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--n", type=int, default=3000)
    m.add_argument("--out", type=Path, default=CFG.paths.ROOT / "external" / "manifest.csv")
    m.add_argument("--seed", type=int, default=42)
    d = sub.add_parser("download")
    d.add_argument("--manifest", type=Path, default=CFG.paths.ROOT / "external" / "manifest.csv")
    d.add_argument("--out-dir", type=Path, required=True)
    v = sub.add_parser("valcopies")
    v.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "manifest":
        build_manifest(args.n, args.out, args.seed)
    elif args.cmd == "download":
        download(args.manifest, args.out_dir)
    else:
        val_copies(args.out_dir)


if __name__ == "__main__":
    main()
