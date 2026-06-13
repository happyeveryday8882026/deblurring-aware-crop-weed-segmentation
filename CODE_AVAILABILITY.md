# Text for the manuscript (Code & Data Availability)

Paste the following into the manuscript after depositing on Zenodo and
replacing `XXXXXXX` with the assigned DOI. A matching LaTeX snippet is provided
at the bottom.

---

## Code Availability

The complete training and evaluation code for the proposed UNet-ResNet
segmentation framework and the NAFNet restoration-aware pipeline — including the
model definitions, the two-stage training scripts, the evaluation metrics, the
complexity-analysis utilities, the stratified 70/15/15 data splits, the exact
software versions, and the random seeds (0–4) — is openly available in a Zenodo
repository at https://doi.org/10.5281/zenodo.XXXXXXX (released under the MIT
License). The development repository is also mirrored on GitHub.

## Data Availability

The imagery (paired sharp/blurred 128×128 patches with three-class masks:
background, crop/sorghum, weed) is the publicly available DeBlurWeedSeg dataset
of Genze et al., openly accessible via Mendeley Data at
https://doi.org/10.17632/k4gvsjv4t3.1 and at
https://github.com/grimmlab/DeBlurWeedSeg. No additional annotation layer was
created; the train/validation/test splits used in this study are released with
the source code in the Zenodo repository listed under Code Availability
(https://doi.org/10.5281/zenodo.XXXXXXX).

---

## LaTeX snippet (replace the existing \bmhead{Data Availability} block)

```latex
\bmhead{Code Availability}
The complete training and evaluation code for the proposed UNet--ResNet
segmentation framework and the NAFNet restoration-aware pipeline---including the
model definitions, the two-stage training scripts, the evaluation metrics, the
complexity-analysis utilities, the stratified 70/15/15 data splits, the exact
software versions, and the random seeds (0--4)---is openly available in a Zenodo
repository at \url{https://doi.org/10.5281/zenodo.XXXXXXX} under the MIT License.

\bmhead{Data Availability}
The imagery (paired sharp/blurred patches with three-class masks: background,
crop/sorghum, and weed) is the publicly available DeBlurWeedSeg dataset of
Genze~\textit{et al.}~\cite{genze2023improved}, openly accessible via the
Mendeley Data repository at \url{https://doi.org/10.17632/k4gvsjv4t3.1} and at
\url{https://github.com/grimmlab/DeBlurWeedSeg}. The train/validation/test
splits used in this study are released together with the source code in the
Zenodo repository listed under Code Availability
(\url{https://doi.org/10.5281/zenodo.XXXXXXX}).
```
