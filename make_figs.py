"""Regenerate the encoder-comparison and ablation bar charts from the real results."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = ".."  # paper root, where the .tex expects the figures
d = json.load(open("results_real.json"))

CLASSES = ["BG", "Crop", "Weed"]


def grouped_bar(rows, labels, fname, title_legend):
    # rows: list of (name, [bg, crop, weed, mDS])
    cats = CLASSES + ["mDS"]
    x = np.arange(len(cats))
    n = len(rows)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (name, vals) in enumerate(rows):
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0.6, 1.0)
    ax.legend(ncol=min(n, 4), fontsize=8, loc="lower center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/{fname}", dpi=200)
    print("wrote", fname)


# Encoder comparison
enc_order = ["ResNet-18", "ResNet-34", "ResNet-50", "ResNet-101"]
enc_rows = []
for n in enc_order:
    v = d["encoder:" + n]
    enc_rows.append((n, v["dice_per_class"] + [v["mDS"]]))
grouped_bar(enc_rows, enc_order, "encoder_dice_comparison.png", "encoder")

# Ablation
abl_order = ["Full Model", "w/o ImageNet Pretrain", "w/o Skip Connections",
             "w/o Dropout", "Single Conv Decoder", "3 Decoder Blocks"]
abl_rows = []
for n in abl_order:
    v = d["ablation:" + n]
    abl_rows.append((n, v["dice_per_class"] + [v["mDS"]]))
grouped_bar(abl_rows, abl_order, "ablation_bar_chart.png", "ablation")
