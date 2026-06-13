# Unified Semantic Segmentation — UNet-ResNet + NAFNet (weed/crop)

Reference implementation accompanying the manuscript by
**Qiyue Li, Shenshen Zou** (College of Plant Protection, Shandong Agricultural
University).

This repository contains the complete, self-contained training and evaluation
pipeline: a **three-class** semantic segmentation network (background, crop
[sorghum], weed) built from a **UNet decoder with a ResNet encoder**
(ResNet-18/34/50/101), an integrated **NAFNet deblurring front-end**, and a
**two-stage restoration-aware training protocol**. It also provides the
evaluation metrics, the complexity analysis (parameters / MACs), and the
stratified 70/15/15 data splits.

> **Scope note.** The three classes (background, crop, weed) are exactly the
> classes annotated in the public DeBlurWeedSeg dataset used here. The code,
> the configuration, the data, and the splits are all consistent with one
> another so the results are fully reproducible.

> **DOI:** archived on Zenodo at `https://doi.org/10.5281/zenodo.XXXXXXX`
> *(replace `XXXXXXX` with the DOI assigned after deposit; see
> "Archiving on Zenodo" below).*

---

## 1. Repository layout

```
code/
├── README.md
├── LICENSE                      # MIT
├── requirements.txt
├── CITATION.cff                 # software + paper citation metadata
├── .zenodo.json                 # Zenodo deposition metadata
├── CODE_AVAILABILITY.md         # text to paste into the manuscript
├── configs/
│   └── default.yaml             # all hyper-parameters (matches Section 4)
├── data_splits/                 # stratified 70/15/15 split csvs (seed 0)
│   ├── train.csv  (910)
│   ├── val.csv    (195)
│   └── test.csv   (195)
├── scripts/                     # one-command reproduction
│   ├── reproduce_encoder_comparison.sh
│   ├── reproduce_ablation.sh
│   └── reproduce_pipeline.sh
└── src/
    ├── models/
    │   ├── encoder.py           # ResNet-18/34/50/101 backbone (torchvision)
    │   ├── decoder.py           # configurable UNet decoder + seg head
    │   ├── unet_resnet.py       # full segmentation model
    │   └── nafnet.py            # NAFNet deblurring front-end
    ├── data/
    │   ├── dataset.py           # 3-class dataset over DeBlurWeedSeg patches
    │   └── transforms.py        # paired augmentation (Section 4)
    ├── losses.py                # CE / L1 / composite objectives
    ├── metrics.py               # Dice, IoU, Precision, Recall
    ├── make_splits.py           # regenerate stratified splits
    ├── train_seg.py             # segmentation training (encoder comparison, ablations)
    ├── train_deblur.py          # Stage 1: NAFNet pre-training
    ├── train_restoration_aware.py  # Stage 2: joint deblur+seg fine-tuning
    ├── evaluate.py              # evaluate a checkpoint on the test set
    └── complexity.py            # parameters / MACs
```

## 2. Installation

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Select the torch/torchvision CUDA build that matches your GPU/driver:
# https://pytorch.org/get-started/previous-versions/
```

Developed with **Python 3.10** and **PyTorch 2.1.0** on a single NVIDIA
GeForce RTX 3090 (24 GB), as reported in Section 4 of the paper.

## 3. Data

The imagery is the publicly available **DeBlurWeedSeg** dataset
(Genze et al., Mendeley Data, https://doi.org/10.17632/k4gvsjv4t3.1):
1,300 paired sharp/blurred 128×128 patches with three-class masks
(background, crop/sorghum, weed).

1. Download `data.zip` from the Mendeley record and extract it.
2. Point `data.image_dir` in `configs/default.yaml` at the resulting `gt/`
   folder (containing `0001.png` … `1300.png`).

The masks are RGB-coded: `[199,199,199]`→background, `[31,119,180]`→crop,
`[255,127,14]`→weed. The stratified 70/15/15 splits used in the paper are in
`data_splits/`. To regenerate them:

```bash
python -m src.make_splits \
  --image-dir "/path/to/DeBlurWeedSeg/data/gt" --out data_splits --seed 0
```

## 4. Quick start

Train the proposed segmentation model (ResNet-50) and test it:

```bash
python -m src.train_seg --config configs/default.yaml \
  --encoder resnet50 --seed 0 --out runs/resnet50_seed0
```

Full restoration-aware pipeline (NAFNet pre-train → joint fine-tune):

```bash
# Stage 1: deblurring front-end
python -m src.train_deblur --config configs/default.yaml --out runs/nafnet
# Stage 2: deblur + segmentation, jointly fine-tuned (lambda = 0.1)
python -m src.train_restoration_aware --config configs/default.yaml \
  --nafnet runs/nafnet/nafnet_best.pt --encoder resnet50 \
  --seed 0 --out runs/pipeline_resnet50_seed0
```

Evaluate any checkpoint:

```bash
python -m src.evaluate --config configs/default.yaml \
  --checkpoint runs/resnet50_seed0/best.pt --kind seg
```

Complexity analysis:

```bash
python -m src.complexity --config configs/default.yaml --encoder resnet50
```

## 5. Reproducing the paper experiments

| Experiment                              | Script                                      |
|-----------------------------------------|---------------------------------------------|
| Encoder comparison (ResNet-18/34/50/101)| `scripts/reproduce_encoder_comparison.sh`   |
| Ablation study                          | `scripts/reproduce_ablation.sh`             |
| Deblur+seg restoration-aware pipeline   | `scripts/reproduce_pipeline.sh`             |
| Parameters / MACs                       | `python -m src.complexity ...`              |

Every configuration is trained with five random seeds (0–4); report the mean
and standard deviation across the five `runs/*_seed{0..4}` outputs. All
hyper-parameters are fixed in `configs/default.yaml`:

* Optimiser: Adam (β₁=0.9, β₂=0.999), lr 1e-3, weight decay 1e-4
* Schedule: cosine annealing to 1e-6, 200 epochs, batch size 16
* Segmentation loss: pixel-wise multi-class cross-entropy
* NAFNet: 300-epoch L1 pre-training, then joint fine-tuning with the composite
  loss `L_seg + 0.1·L_deblur` (restoration lr reduced to 1e-5)

## 6. Ablation flags (`src/train_seg.py`)

| Flag                  | Configuration          |
|-----------------------|------------------------|
| `--no-pretrain`       | w/o ImageNet pretrain  |
| `--no-skip`           | w/o skip connections   |
| `--dropout 0.0`       | w/o dropout            |
| `--decoder-convs 1`   | single-conv decoder    |
| `--decoder-blocks 3`  | 3 decoder blocks       |

## 7. Archiving on Zenodo (for the DOI requested by the editor)

1. Sign in at https://zenodo.org with your ORCID/GitHub.
2. Either link your GitHub repo and publish a release (Zenodo mints a DOI
   automatically using `.zenodo.json` and `CITATION.cff`), **or** upload this
   `code/` folder as a `.zip` via *New upload*. The metadata fields are
   pre-filled from `.zenodo.json`.
3. Publish. Copy the assigned DOI and paste it into:
   * this README (top, "DOI:"),
   * `CITATION.cff` (`doi:` field),
   * the manuscript's **Code Availability** section (see `CODE_AVAILABILITY.md`).

## 8. License & citation

Code released under the MIT License (see `LICENSE`). The DeBlurWeedSeg imagery
remains subject to its original license. If you use this software, please cite
both the paper and the Zenodo software record (see `CITATION.cff`).

## 9. Acknowledgements

The deblurring front-end is based on NAFNet (Chen et al., ECCV 2022,
https://github.com/megvii-research/NAFNet). The dataset and the three-class
weed-segmentation baseline are from Genze et al.
(https://github.com/grimmlab/DeBlurWeedSeg).
