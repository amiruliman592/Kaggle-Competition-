"""
make_fig_pipeline.py — Figure 1 of the report: the final pipeline (exp101) as boxes and arrows.

Settings shown in the boxes come from src/final_pipeline.py (header and constants) and src/config.py.
Output: docs/report/figures/fig_pipeline.pdf (vector, full page width) and a .png preview.
Run:  python docs/report/make_fig_pipeline.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
W, H = 7.0, 1.75          # inches: full text width of the two-column ACM layout
FONT = 7                  # pt, readable at print size

# fill colours per stage (light, print-friendly)
C_IO, C_VIEW, C_MODEL, C_FUSE, C_POST = "#eeeeee", "#e3eefa", "#d6e6f5", "#fdebd3", "#e2f2e2"

ROW_TOP, ROW_BOT, ROW_MID = 1.25, 0.45, 0.85   # y centres (inches)
BOX_H = 0.62                                 # height of the one-row boxes
BIG_H = 1.45                                 # height of boxes that span both rows


def box(ax, x, y, w, h, text, fill, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=fill, ec="#555555", lw=0.6))
    lines = text.split("\n")
    if bold_first:
        ax.text(x + w / 2, y + h / 2 - 0.06, lines[0], ha="center", va="top", fontsize=FONT, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.20, "\n".join(lines[1:]), ha="center", va="top", fontsize=FONT - 0.5,
                linespacing=1.25)
    else:
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=FONT)
    return x, x + w


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.7, color="#333333",
                                 shrinkA=0, shrinkB=0))


def main():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    gap = 0.13
    x = 0.03
    # 1. input
    _, r_in = box(ax, x, ROW_MID, 0.72, BIG_H, "Test image\n2048 × 2048\nGONG H$\\alpha$\n\nno tiling,\nno pre-\nprocessing",
                  C_IO)
    # 2. four views
    x = r_in + gap
    _, r_view = box(ax, x, ROW_MID, 0.72, BIG_H, "4 views\noriginal\nh-flip\nv-flip\nh+v flip\n\nmasks flipped\nback", C_VIEW)
    # 3. two models
    x = r_view + gap
    _, r_mod = box(ax, x, ROW_TOP, 1.10, BOX_H, "Mask R-CNN X101\nFPN, 56 × 56 masks\nraw score ≥ 0.2", C_MODEL)
    box(ax, x, ROW_BOT, 1.10, BOX_H, "Mask R-CNN R50\nFPN, 28 × 28 masks\nraw score ≥ 0.2", C_MODEL)
    # 4. fusion 1, per model
    x = r_mod + gap
    _, r_f1 = box(ax, x, ROW_TOP, 0.98, BOX_H, "Fusion 1: views\nIoU ≥ 0.5 → union\nscore = mean of 4", C_FUSE)
    box(ax, x, ROW_BOT, 0.98, BOX_H, "Fusion 1: views\nIoU ≥ 0.5 → union\nscore = mean of 4", C_FUSE)
    # 5. fusion 2, models
    x = r_f1 + gap
    _, r_f2 = box(ax, x, ROW_MID, 0.98, BOX_H + 0.1, "Fusion 2: models\nIoU ≥ 0.3 →\nintersection\nscore = mean of 2",
                  C_FUSE)
    # 6. post-processing
    x = r_f2 + gap
    _, r_pp = box(ax, x, ROW_MID, 0.92, BIG_H, "Clean-up\nclosing (5 px)\nscore ≥ 0.6\narea ≥ 50 px²\noverlap removal\n"
                  "(higher score\nkeeps pixels;\n>20% covered\n→ dropped)", C_POST)
    # 7. output
    x = r_pp + gap
    box(ax, x, ROW_MID, W - 0.03 - x, BIG_H, "Submission\nCSV\n\nfilament_id,\nCOCO RLE", C_IO)

    # arrows
    arrow(ax, r_in, ROW_MID, r_in + gap, ROW_MID)
    for y in (ROW_TOP, ROW_BOT):
        arrow(ax, r_view, y, r_view + gap, y)                  # views -> each model
        arrow(ax, r_mod, y, r_mod + gap, y)                    # model -> its fusion 1
        arrow(ax, r_f1, y, r_f1 + gap, ROW_MID + (0.12 if y > ROW_MID else -0.12))  # fusion 1 -> fusion 2
    arrow(ax, r_f2, ROW_MID, r_f2 + gap, ROW_MID)
    arrow(ax, r_pp, ROW_MID, r_pp + gap, ROW_MID)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_pipeline.pdf")
    fig.savefig(OUT / "fig_pipeline.png", dpi=200)
    print("->", OUT / "fig_pipeline.pdf")


if __name__ == "__main__":
    main()
