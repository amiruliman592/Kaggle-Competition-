"""
m2f.py — Phase 5 step 2: Mask2Former (Hugging Face transformers, Swin backbone).

Instead of box -> 28x28 mask, each of 100 learned "queries" paints one whole
filament on a 1/4-resolution feature map. Starts from COCO-instance weights;
the class head is re-initialised for one class. Trained on random crops of the
native 2048 images (all annotators' labels, flips, brightness/contrast),
predicts on the full 2048 image. Raw predictions are saved exactly like
run_pipeline.py --step dump, so ablate_postprocess / ensemble / submit_from_raw
work unchanged.

Usage (Colab A100, notebooks/colab_phase5_a100.ipynb):
  python src/m2f.py train --name m2f_swin_s                  # resumes from output/m2f/<name>/last.pt
  python src/m2f.py train --name probe --max-iter 30 --no-resume
  python src/m2f.py dump  --name m2f_swin_s                  # raw_preds_<name>_{val,test}.json.gz
"""

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader, Dataset

from annotations import load_annotation_sets, select_annotations
from config import CFG
from preprocess import image_path
from split import load_split
from utils.rle import encode_mask, polygon_to_rle, save_predictions
from utils.seed import set_seed

M2F_MODEL = "facebook/mask2former-swin-small-coco-instance"
CROP = 1024              # training crop at native resolution (>= 2048 = whole image)
BATCH = 4
MAX_ITER = 12000
LR = 1e-4                # backbone gets LR x BACKBONE_LR_MULT
BACKBONE_LR_MULT = 0.1
WEIGHT_DECAY = 0.05
WARMUP = 200
CLIP = 0.01              # full-model grad-norm clip, as in the Mask2Former paper setup
MIN_CROP_AREA = 15       # instance slivers smaller than this at a crop edge are dropped
CHECKPOINT_PERIOD = 1000
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def run_dir(name: str) -> Path:
    return CFG.paths.OUTPUT_DIR / "m2f" / name


def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(((img_rgb.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1).copy())


def read_rgb(path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def pick_crop(bboxes: List[List[float]], size: int, hw=(2048, 2048), p_object: float = 0.8):
    """Top-left (x0, y0) of a size x size crop; usually near a random filament, sometimes anywhere."""
    if size >= min(hw):
        return 0, 0
    if bboxes and random.random() < p_object:
        x, y, w, h = random.choice(bboxes)
        cx = x + w / 2 + random.uniform(-size / 4, size / 4)
        cy = y + h / 2 + random.uniform(-size / 4, size / 4)
    else:
        cx, cy = random.uniform(0, hw[1]), random.uniform(0, hw[0])
    return (int(np.clip(round(cx - size / 2), 0, hw[1] - size)),
            int(np.clip(round(cy - size / 2), 0, hw[0] - size)))


class FilamentM2F(Dataset):
    """Training samples (one per annotator set with "all") -> crop tensor + instance masks."""

    def __init__(self, stems: List[str], label_policy: str = "all", crop: int = CROP, limit: int = 0):
        sets = load_annotation_sets()
        if label_policy == "all":
            samples = [(s, anns) for s in stems for anns in sets[s]]
        else:
            first = select_annotations(sets)
            samples = [(s, first[s]) for s in stems]
        self.samples = [(s, [a for a in anns if a["area"] >= 1]) for s, anns in samples]
        self.samples = [x for x in self.samples if x[1]][:limit or None]
        self.crop = crop

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        stem, anns = self.samples[i]
        img = read_rgb(image_path(stem, "train"))
        h, w = img.shape[:2]
        full = [coco_mask.decode(polygon_to_rle(a["segmentation"], (h, w))) for a in anns]
        size = min(self.crop, h, w)
        for _ in range(10):                      # a crop with at least one filament
            x0, y0 = pick_crop([a["bbox"] for a in anns], size, (h, w))
            masks = [m[y0:y0 + size, x0:x0 + size] for m in full]
            masks = [m for m in masks if m.sum() >= MIN_CROP_AREA]
            if masks:
                break
        else:                                    # centre on the largest filament
            x, y, bw, bh = max((a["bbox"] for a in anns), key=lambda b: b[2] * b[3])
            x0 = int(np.clip(round(x + bw / 2 - size / 2), 0, w - size))
            y0 = int(np.clip(round(y + bh / 2 - size / 2), 0, h - size))
            masks = [m[y0:y0 + size, x0:x0 + size] for m in full]
            masks = [m for m in masks if m.any()]
        img = img[y0:y0 + size, x0:x0 + size]
        masks = np.stack(masks)
        if random.random() < 0.5:
            img, masks = img[:, ::-1], masks[:, :, ::-1]
        if random.random() < 0.5:
            img, masks = img[::-1], masks[:, ::-1]
        alpha, beta = random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)   # contrast, brightness
        img = np.clip((img.astype(np.float32) - 128) * alpha + 128 * beta, 0, 255)
        return (to_tensor(img), torch.from_numpy(masks.astype(np.float32).copy()),
                torch.zeros(len(masks), dtype=torch.long))


def _seed_worker(_):
    """Crops/flips use `random`; torch gives every worker its own seed derived from CFG.SEED."""
    random.seed(torch.initial_seed() % 2 ** 32)


def collate(batch):
    return {"pixel_values": torch.stack([b[0] for b in batch]),
            "mask_labels": [b[1] for b in batch], "class_labels": [b[2] for b in batch]}


def build_model(pretrained: str = M2F_MODEL):
    from transformers import Mask2FormerForUniversalSegmentation
    return Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained, num_labels=1, id2label={0: "filament"}, label2id={"filament": 0},
        ignore_mismatched_sizes=True)


def autocast_dtype():
    """bf16 on GPUs that have it natively (A100), else fp32."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return None


def train(name: str, max_iter: int = MAX_ITER, crop: int = CROP, batch: int = BATCH, resume: bool = True,
          pretrained: str = M2F_MODEL, limit: int = 0):
    set_seed(CFG.SEED)
    random.seed(CFG.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = run_dir(name)
    out.mkdir(parents=True, exist_ok=True)

    train_ids, _ = load_split()
    ds = FilamentM2F(train_ids, CFG.data.TRAIN_LABEL_POLICY, crop, limit)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=len(ds) >= batch, collate_fn=collate,
                        num_workers=CFG.train.NUM_WORKERS, persistent_workers=CFG.train.NUM_WORKERS > 0,
                        worker_init_fn=_seed_worker)
    model = build_model(pretrained).to(device).train()
    backbone = [p for n, p in model.named_parameters() if "pixel_level_module.encoder" in n]
    rest = [p for n, p in model.named_parameters() if "pixel_level_module.encoder" not in n]
    opt = torch.optim.AdamW([{"params": backbone, "lr": LR * BACKBONE_LR_MULT}, {"params": rest, "lr": LR}],
                            weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / WARMUP) * (1 - min(it, max_iter - 1) / max_iter) ** 0.9)
    it = 0
    last = out / "last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        it = ck["iter"]
        print(f"[M2F] Resumed from {last} at iter {it}")

    dtype = autocast_dtype()
    print(f"[M2F] {pretrained} | {len(ds)} samples | crop {crop} | batch {batch} | {max_iter} iters | "
          f"{'bf16' if dtype else 'fp32'} | {device}")
    t0, losses = time.time(), []
    while it < max_iter:
        for b in loader:
            b = {"pixel_values": b["pixel_values"].to(device),
                 "mask_labels": [m.to(device) for m in b["mask_labels"]],
                 "class_labels": [c.to(device) for c in b["class_labels"]]}
            with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
                loss = model(**b).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            sched.step()
            it += 1
            losses.append(loss.item())
            if it % 20 == 0 or it == max_iter:
                sec = (time.time() - t0) / 20
                mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
                print(f"[M2F] iter {it}/{max_iter} loss {np.mean(losses):.3f} lr {sched.get_last_lr()[1]:.2e} "
                      f"time {sec:.2f} s/it (~{sec * (max_iter - it) / 3600:.1f} h left) mem {mem:.1f} GB",
                      flush=True)
                t0, losses = time.time(), []
            if it % CHECKPOINT_PERIOD == 0 or it == max_iter:
                tmp = out / "last.pt.tmp"
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "iter": it}, tmp)
                tmp.replace(last)
            if it >= max_iter:
                break
    torch.save(model.state_dict(), out / "model_final.pt")
    if device == "cuda":
        print(f"[M2F] Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print(f"[M2F] Done -> {out / 'model_final.pt'}")


@torch.no_grad()
def predict_image(model, img_rgb: np.ndarray, min_score: float, device: str) -> List[Dict]:
    """
    Full-image prediction -> RLE dicts with "score", highest first. Score =
    class probability x mean mask probability inside the mask (standard
    Mask2Former instance score); masks are upsampled logits > 0.
    """
    h, w = img_rgb.shape[:2]
    dtype = autocast_dtype()
    with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
        o = model(pixel_values=to_tensor(img_rgb)[None].to(device))
    cls = o.class_queries_logits[0].float().softmax(-1)[:, 0]          # last column = "no object"
    logits = o.masks_queries_logits[0].float()                          # (Q, H/4, W/4)
    prob = logits.sigmoid()
    fg = prob > 0.5
    score = cls * (prob * fg).sum((1, 2)) / fg.sum((1, 2)).clamp(min=1)
    out = []
    for q in torch.argsort(score, descending=True).tolist():
        if score[q] < min_score:
            break
        m = torch.nn.functional.interpolate(logits[q][None, None], size=(h, w), mode="bilinear",
                                            align_corners=False)[0, 0] > 0
        m = m.cpu().numpy()
        if m.any():
            rle = encode_mask(m)
            rle["score"] = float(score[q])
            out.append(rle)
    return out


def dump(name: str, limit: int = 0, pretrained: str = M2F_MODEL):
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(pretrained)
    model.load_state_dict(torch.load(run_dir(name) / "model_final.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    _, val_ids = load_split()
    test_stems = [p.stem for p in sorted(CFG.paths.TEST_IMG_DIR.glob("*.jpeg"))]
    for split, stems, src in (("val", val_ids, "train"), ("test", test_stems, "test")):
        stems = stems[:limit] if limit else stems
        t0 = time.time()
        preds = {s: predict_image(model, read_rgb(image_path(s, src)), CFG.infer.DUMP_SCORE_THRESHOLD, device)
                 for s in tqdm(stems, desc=f"M2F {split}")}
        sec = (time.time() - t0) / max(len(stems), 1)
        path = CFG.paths.OUTPUT_DIR / f"raw_preds_{name}_{split}.json.gz"
        save_predictions(preds, path)
        print(f"[Dump] {split}: {sum(map(len, preds.values()))} raw masks, {sec:.2f} s/img -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Mask2Former train / raw prediction dump")
    parser.add_argument("cmd", choices=["train", "dump"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--crop", type=int, default=CROP)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--pretrained", default=M2F_MODEL)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--label-policy", choices=["first", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="first N samples / images only (smoke test)")
    args = parser.parse_args()
    CFG.paths.ensure_output_dirs()
    CFG.data.TRAIN_LABEL_POLICY = args.label_policy
    if args.cmd == "train":
        train(args.name, args.max_iter, args.crop, args.batch, not args.no_resume, args.pretrained, args.limit)
    else:
        dump(args.name, args.limit, args.pretrained)


if __name__ == "__main__":
    main()
