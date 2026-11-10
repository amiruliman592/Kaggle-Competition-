"""
weights_io.py — Store the final model weights inside the git repo (host rule: code, config and weights in one repo).

GitHub refuses files > 100 MB, so each checkpoint is
  1. stripped to the model weights only (Detectron2's model_final.pth also holds optimizer + scheduler state),
  2. split into 90 MB parts:  weights/<name>.pth.part00, .part01, ...
  3. listed in weights/manifest.json with its size and SHA-256.
`unpack` joins the parts back into weights/<name>.pth (git-ignored) and checks the SHA-256.

Usage:
  python src/weights_io.py pack --ckpt <drive>/x101_2048_m28/model_final.pth --name x101_2048_m28 --out weights
  python src/weights_io.py unpack --dir weights          # every model in the manifest
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Dict

PART_BYTES = 90 * 1024 * 1024


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pack(ckpt: Path, name: str, out_dir: Path, part_bytes: int = PART_BYTES) -> Dict:
    """Strip one checkpoint to {"model": state_dict}, split it into parts and add it to the manifest."""
    import torch

    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model = state["model"] if isinstance(state, dict) and "model" in state else state
    buf = io.BytesIO()
    torch.save({"model": model}, buf)
    data = buf.getvalue()

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{name}.pth.part*"):
        old.unlink()
    parts = []
    for i in range(0, len(data), part_bytes):
        p = out_dir / f"{name}.pth.part{i // part_bytes:02d}"
        p.write_bytes(data[i:i + part_bytes])
        parts.append(p.name)

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[name] = {"bytes": len(data), "sha256": _sha256(data), "parts": parts}
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"[Weights] {name}: {len(data) / 1e6:.1f} MB (from {ckpt.stat().st_size / 1e6:.1f} MB) "
          f"-> {len(parts)} parts in {out_dir}")
    return manifest[name]


def unpack(weights_dir: Path) -> Dict[str, Path]:
    """Join the parts of every model in the manifest into <name>.pth; checks size + SHA-256. Skips finished ones."""
    manifest = json.loads((weights_dir / "manifest.json").read_text())
    out = {}
    for name, info in manifest.items():
        target = weights_dir / f"{name}.pth"
        if target.exists() and target.stat().st_size == info["bytes"]:
            data = target.read_bytes()
        else:
            data = b"".join((weights_dir / p).read_bytes() for p in info["parts"])
        if len(data) != info["bytes"] or _sha256(data) != info["sha256"]:
            raise ValueError(f"{name}: joined weights do not match the manifest (size or SHA-256). "
                             "Are all parts present? Did git change line endings (see .gitattributes)?")
        if not target.exists():
            target.write_bytes(data)
        out[name] = target
        print(f"[Weights] {name}: OK ({len(data) / 1e6:.1f} MB, SHA-256 checked) -> {target}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Pack / unpack model weights stored in the repo")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", type=Path, default=Path("weights"))
    u = sub.add_parser("unpack")
    u.add_argument("--dir", type=Path, default=Path("weights"))
    args = parser.parse_args()
    if args.cmd == "pack":
        pack(args.ckpt, args.name, args.out)
    else:
        unpack(args.dir)


if __name__ == "__main__":
    main()
