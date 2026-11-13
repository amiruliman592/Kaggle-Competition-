"""
m81_rubric.py — Phase 8 M8.1 (Mirul): rubric numbers + Dice/IoU histograms for the final pipeline (exp101) on val.

CPU only. Reads the saved fused raw val predictions of exp101, applies the final post-processing
(src/final_pipeline.postprocess) and scores them against every annotator set (official PQ, as on Kaggle).

Outputs:
  results/phase8/Mirul/rubric_exp101.json   all numbers for the report
  docs/report/figures/fig_dice_iou.pdf      Dice + IoU histograms (also .png)

Usage (from the repo root):  python docs/report/m81_rubric.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from annotations import load_gt_entries  # noqa: E402
from evaluate import COVER_FRAC, MATCH_IOU, evaluate_official, match_image, official_dice  # noqa: E402
from final_pipeline import postprocess  # noqa: E402
from utils.rle import load_predictions  # noqa: E402

RAW = ROOT / "results/phase6/Mirul/raw_preds_tta2_m28_r50_val.json.gz"
OUT_JSON = ROOT / "results/phase8/Mirul/rubric_exp101.json"
OUT_FIG = ROOT / "docs/report/figures/fig_dice_iou"
N_BOOT, SEED = 2000, 42


def stats(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {"n": 0}
    q = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
            "q25": float(q[0]), "median": float(q[1]), "q75": float(q[2])}


def main():
    stems = json.loads((ROOT / "splits/val_ids.json").read_text())["val_ids"]
    entries = load_gt_entries(stems)
    preds = postprocess(load_predictions(RAW))
    preds = {s: preds.get(s, []) for s in stems}

    # 1) Headline numbers (must equal exp101: PQ 0.4480, Dice 0.6520)
    off = evaluate_official(entries, preds)
    dice = official_dice(entries, preds)
    assert abs(off["PQ"] - 0.4480) < 5e-4, f"PQ {off['PQ']:.4f} != exp101 0.4480"

    # 2) Per annotator entry: pair IoU/Dice, per-entry Dice, 1:N and N:1 relations
    pair_iou, tp_iou, entry_dice, entry_stem = [], [], [], []
    frag_k, merge_k = [], []          # k = pieces per fragmented GT / GTs per merged prediction
    n_frag = n_merge = 0
    per_entry = []                    # (stem, tp_iou list, fp, fn) for the bootstrap
    for eid, gt in entries.items():
        if not gt:
            continue
        stem = eid.split("-", 1)[1]
        pr = preds.get(stem, [])
        m = match_image(gt, pr)
        n_frag += m["n_fragmented_gt"]
        n_merge += m["n_merged_pred"]
        # per-entry samplewise Dice (same as official_dice's primary value)
        u_gt = coco_mask.merge(gt)
        a_gt = float(coco_mask.area(u_gt))
        if pr:
            u_pr = coco_mask.merge(pr)
            a_pr = float(coco_mask.area(u_pr))
            ui = float(coco_mask.iou([u_pr], [u_gt], [0])[0][0])
            inter = ui * (a_gt + a_pr) / (1 + ui)
            entry_dice.append(2 * inter / (a_gt + a_pr))
            iou = np.asarray(coco_mask.iou(pr, gt, [0] * len(gt)), dtype=float)  # (P, G)
            pair_iou.extend(iou[iou > 0].tolist())
            hit = iou > MATCH_IOU
            e_tp = iou[hit].tolist()
            tp_iou.extend(e_tp)
            fp, fn = int((hit.sum(1) == 0).sum()), int((hit.sum(0) == 0).sum())
            # k distribution, same rule as evaluate.match_image (piece >= 50% inside the other mask)
            pa = coco_mask.area(pr).astype(float)
            ga = coco_mask.area(gt).astype(float)
            it = iou * (pa[:, None] + ga[None, :]) / (1 + iou)
            in_gt = ((it / pa[:, None] >= COVER_FRAC) & (it > 0)).sum(0)
            cov = ((it / ga[None, :] >= COVER_FRAC) & (it > 0)).sum(1)
            frag_k.extend(in_gt[in_gt >= 2].tolist())
            merge_k.extend(cov[cov >= 2].tolist())
        else:
            entry_dice.append(0.0)
            e_tp, fp, fn = [], 0, len(gt)
        entry_stem.append(stem)
        per_entry.append((stem, e_tp, fp, fn))
    assert len(frag_k) == n_frag and len(merge_k) == n_merge
    assert abs(np.mean(entry_dice) - dice["Dice_samplewise"]) < 1e-9

    pair_iou = np.asarray(pair_iou)
    pair_dice = 2 * pair_iou / (1 + pair_iou)
    tp_iou = np.asarray(tp_iou)
    tp_dice = 2 * tp_iou / (1 + tp_iou)

    # 3) Uncertainty: bootstrap over the 141 val files (all annotator entries of a file move together)
    rng = np.random.default_rng(SEED)
    by_stem = {}
    for (s, e_tp, fp, fn), d in zip(per_entry, entry_dice):
        b = by_stem.setdefault(s, [0.0, 0, 0, 0, 0.0, 0])  # sum_iou, tp, fp, fn, sum_dice, n_entries
        b[0] += sum(e_tp); b[1] += len(e_tp); b[2] += fp; b[3] += fn; b[4] += d; b[5] += 1
    arr = np.array(list(by_stem.values()))
    boot = {"PQ": [], "SQ": [], "RQ": [], "Dice": []}
    for _ in range(N_BOOT):
        t = arr[rng.integers(0, len(arr), len(arr))].sum(0)
        sq = t[0] / t[1] if t[1] else 0.0
        rq = t[1] / (t[1] + 0.5 * t[2] + 0.5 * t[3])
        boot["PQ"].append(sq * rq); boot["SQ"].append(sq); boot["RQ"].append(rq); boot["Dice"].append(t[4] / t[5])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}

    res = {
        "source": str(RAW.relative_to(ROOT)).replace("\\", "/"),
        "split": "frozen val, 141 files, every annotator set (official Kaggle PQ)",
        "num_entries": off["num_entries"],
        "PQ": off["PQ"], "SQ": off["SQ"], "RQ": off["RQ"],
        "TP": off["TP"], "FP": off["FP"], "FN": off["FN"],
        "Dice_samplewise": dice["Dice_samplewise"], "Dice_global": dice["Dice_global"],
        "bootstrap_95ci_files": {"n_boot": N_BOOT, **ci},
        "n_pred_masks": int(sum(len(v) for v in preds.values())),
        "dist_pair_IoU_all_overlaps": stats(pair_iou),
        "dist_pair_Dice_all_overlaps": stats(pair_dice),
        "dist_IoU_matched_TP": stats(tp_iou),
        "dist_Dice_matched_TP": stats(tp_dice),
        "dist_Dice_per_entry": stats(entry_dice),
        "one_to_many_fragmented_gt": n_frag,
        "many_to_one_merged_pred": n_merge,
        "one_to_many_k_counts": {str(k): int(v) for k, v in zip(*np.unique(frag_k, return_counts=True))},
        "many_to_one_k_counts": {str(k): int(v) for k, v in zip(*np.unique(merge_k, return_counts=True))},
        "relation_rule": f"a piece counts if >= {COVER_FRAC:.0%} of its area lies inside the other mask",
        "note_pairs": "all_overlaps = every GT/pred pair with IoU > 0 (organisers' notebook plots); "
                      "matched_TP = pairs with IoU > 0.5 (the PQ true positives)",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1))

    # 4) Figure: Dice and IoU histograms over all overlapping pairs, TP part highlighted
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    bins = np.linspace(0, 1, 41)
    for ax, vals, tp, name in [(axes[0], pair_dice, tp_dice, "Dice"), (axes[1], pair_iou, tp_iou, "IoU")]:
        ax.hist(vals, bins=bins, color="#b8c4d6", label="all overlapping pairs")
        ax.hist(tp, bins=bins, color="#2f5d9a", label="matched (IoU > 0.5)")
        thr = 2 * MATCH_IOU / (1 + MATCH_IOU) if name == "Dice" else MATCH_IOU
        ax.axvline(thr, color="k", lw=0.8, ls="--")
        ax.set_xlabel(f"{name} per predicted / ground-truth pair")
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("pairs")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG.with_suffix(".pdf"))
    fig.savefig(OUT_FIG.with_suffix(".png"), dpi=200)

    print(json.dumps({k: res[k] for k in ["PQ", "SQ", "RQ", "TP", "FP", "FN", "Dice_samplewise",
                                          "bootstrap_95ci_files", "one_to_many_fragmented_gt",
                                          "many_to_one_merged_pred", "one_to_many_k_counts",
                                          "many_to_one_k_counts", "dist_Dice_matched_TP",
                                          "dist_IoU_matched_TP", "dist_pair_Dice_all_overlaps"]}, indent=1))
    print(f"saved {OUT_JSON.relative_to(ROOT)} and {OUT_FIG.relative_to(ROOT)}.pdf/.png")


if __name__ == "__main__":
    main()
