"""Generate the real qualitative figure deblur_seg_vis.png from the trained model.

For two motion-blurred test patches, shows a 2x2 panel:
  top-left   : motion-blurred input
  top-right  : NAFNet-deblurred restoration
  bottom-left: ground-truth mask
  bottom-right: predicted mask (proposed pipeline)

Masks use the paper's three-class colour code: background=black, crop=green, weed=red.
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.models.nafnet import build_deblur_model
from src.models.unet_resnet import build_segmentation_model
from src.data.dataset import PlantSegDataset, read_split
from src.utils import load_config, get_device

OUTDIR = ".."
CMAP = np.array([[0, 0, 0], [0, 180, 0], [220, 30, 30]], dtype=np.uint8)  # bg, crop, weed


def colorize(mask):
    return CMAP[mask]


def main():
    cfg = load_config("configs/default.yaml")
    cfg["model"]["encoder"] = "resnet34"
    dev = get_device(cfg)
    naf = build_deblur_model(cfg).to(dev).eval()
    seg = build_segmentation_model(cfg).to(dev).eval()
    state = torch.load("runs/proposed_real.pt", map_location=dev)
    naf.load_state_dict(state["naf"])
    seg.load_state_dict(state["seg"])

    ids = read_split("data_splits/test.csv")
    ds = PlantSegDataset(ids, cfg["data"]["image_dir"], transform=None)

    # Pick the two test patches with the most weed pixels (most informative).
    weed_counts = [(int((ds[i][2] == 2).sum()), i) for i in range(len(ds))]
    weed_counts.sort(reverse=True)
    chosen = [weed_counts[0][1], weed_counts[3][1]]

    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    titles = ["Motion-blurred input", "Deblurred (NAFNet)", "GT mask", "Pred mask"]
    for r, idx in enumerate(chosen):
        blurry, sharp, mask = ds[idx]
        with torch.no_grad():
            xb = blurry.unsqueeze(0).to(dev)
            deb = naf(xb)
            pred = seg(deb).argmax(1)[0].cpu().numpy()
        blur_img = (blurry.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        deb_img = (deb[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        panels = [blur_img, deb_img, colorize(mask.numpy()), colorize(pred)]
        for c in range(4):
            axes[r, c].imshow(panels[c])
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=11)
        axes[r, 0].set_ylabel(f"Case {r + 1}", fontsize=11)

    legend = [Patch(facecolor=np.array(CMAP[i]) / 255, label=l)
              for i, l in enumerate(["Background", "Crop", "Weed"])]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(f"{OUTDIR}/deblur_seg_vis.png", dpi=200)
    print("wrote deblur_seg_vis.png using test patches", chosen)


if __name__ == "__main__":
    main()
