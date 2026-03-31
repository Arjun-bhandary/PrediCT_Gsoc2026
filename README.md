<p align="center">
  <img src="https://developers.google.com/open-source/gsoc/resources/downloads/GSoC-Horizontal.svg" alt="GSoC Logo" width="400"/>
</p>

<h1 align="center">PrediCT — GSoC 2026 Evaluation Tasks</h1>

<p align="center">
  <b>Building and Comparing Segmentation Strategies for Coronary Artery Calcium</b><br/>
  <i>Machine Learning for Science (ML4SCI) / PREDICT</i>
</p>


---

## Overview

This repository contains the completed evaluation tasks for the **GSoC 2026 — ML4SCI / PREDICT** project: *"Building and Comparing Segmentation Strategies for Coronary Artery Calcium (CAC)."*

Model weights can be found **[over here](https://drive.google.com/drive/folders/1X7FDS6VY8C3INQ8kbQ3gppBSfH7bSQUO?usp=sharing)**.

The work is organized into two tasks:

| Task | Description | Key Deliverable |
|------|-------------|-----------------|
| **Task 1** (Common) | COCA dataset preprocessing & data loading pipeline | End-to-end DICOM → NIfTI pipeline with augmentation, stratified splits, and class-imbalance handling |
| **Task 2** (Specific) | Heart segmentation from cardiac CT | Four trained models compared; best achieves **Dice = 0.9288** on held-out validation |

---

## Repository Structure

```
PrediCT_Gsoc2026/
├── Task_1/                              # Common Task — COCA Preprocessing
│   ├── COCA_task1.ipynb                 # Full preprocessing pipeline notebook
│   ├── img.png                          # 9-panel dataset characterization figure
│   └── postproc.png                     # Post-processing visualization
│
└── Task2/                               # Specific Task — Heart Segmentation
    ├── SegFormer_ mit_b1/               # ★ Best model (Dice = 0.9288)
    │   ├── train.ipynb                  #   Training notebook
    │   └── output/
    │       ├── eval_stats.json          #   Per-scan Dice scores
    │       ├── training_curves.png      #   Loss & metric curves
    │       └── *_pred.png              #   Prediction visualizations
    │
    ├── AttentionUNet_DenseNet121/        # Attention U-Net + DenseNet121 encoder
    │   ├── train.ipynb
    │   └── output/
    │
    ├── AttentionUnet_ResNet18/           # Attention U-Net + ResNet18 encoder
    │   ├── train.ipynb
    │   └── Output/
    │
    └── UNet_ResNet18/                    # Vanilla U-Net + ResNet18 encoder
        ├── Train.ipynb
        └── output/
```

---

## Task 1 — COCA Dataset Preprocessing

### Dataset

The [Stanford COCA dataset](https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-b7e7-861c141a9ac7) contains **787 gated coronary CT scans** with XML annotations providing per-lesion masks labeled by artery (LAD, LCX, RCA, LM).

### Pipeline

The notebook `Task_1/COCA_task1.ipynb` implements a complete preprocessing pipeline:

1. **DICOM → NIfTI conversion** — Parses raw DICOM volumes via SimpleITK; extracts calcium annotations from plist-format XML using `cv2.fillPoly`; outputs paired image–mask NIfTI files.

2. **Isotropic resampling** — Standardizes all volumes to **0.7 mm** isotropic spacing using `COCA_resampler.py`.

3. **Dataset characterization** — Comprehensive analysis across all 787 volumes:
   - Sparsity analysis (positive voxel ratios)
   - HU distributions (global and within annotated regions)
   - Per-patient calcium burden quantification
   - Voxel spacing variability
   - Positive-slice ratios
   - Results exported as CSV/JSON and visualized in a 9-panel figure

4. **HU windowing** — Clips intensities to `[-175, 1500]` HU, retaining soft-tissue context while capturing the full dynamic range of calcified deposits.

5. **Patient-level stratified split** — 70/15/15 train/val/test split, stratified by calcium burden category (zero, low, high) to ensure representative distribution.

6. **Data augmentation** (HU-semantics-preserving):
   - Left-right flipping, random rotations (±15°), elastic deformation
   - Gaussian noise (σ = 0.02), small intensity shifts (±0.1)
   - Colour jitter and vertical flipping excluded as non-physical for CT

7. **Class-imbalance handling** — 2:1 positive-to-negative patch ratio via MONAI's `RandCropByPosNegLabeld`, combined with Dice + weighted BCE loss.

8. **Dual data loaders**:
   - **MONAI `CacheDataset`** — Caches preprocessed volumes in RAM for fast training
   - **Lightweight PyTorch `Dataset`** — Fallback with positive-biased patch extraction

### Preprocessing Summary

| Parameter | Value |
|-----------|-------|
| Total scans processed | 787 |
| Resampled spacing | 0.7 mm isotropic |
| HU window | [-175, 1500] |
| Train / Val / Test | 550 / 118 / 119 |
| Split strategy | Patient-level, stratified by calcium burden |
| Patch size | 160 × 160 × 160 |
| Pos:Neg sampling ratio | 2:1 |
| Loss function | Dice + Weighted BCE |

### Known Data Challenge

The COCA XML `ImageIndex` field does not correspond to the physical slice ordering used by SimpleITK (previously noted by the Stanford CS230 group), causing mask-to-slice misalignment. This is documented in the notebook with a proposed resolution via `SOPInstanceUID` or `ImagePositionPatient` z-coordinate matching.

---

## Task 2 — Heart Segmentation

### Objective

Train a model to segment the whole heart from cardiac CT scans, achieving a minimum **Dice score of 0.85** on a held-out validation set.

### Ground Truth Generation

Heart masks were generated by running [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) on **47 COCA scans** and binarizing the whole-heart label.

### Models Compared

All four models were trained using the [medic-ai](https://github.com/innat/medic-ai) library with **Keras 3 (PyTorch backend)** on a Tesla T4 GPU.

**Common training setup:**
- 47 scans (39 train / 8 validation)
- HU window: `[-175, 400]`
- Loss: BinaryDiceCE
- Optimizer: AdamW (weight decay = 5×10⁻⁴)
- LR schedule: Warmup cosine (warmup epochs → peak LR, cosine decay)
- Inference: Sliding window (overlap 0.5, Gaussian weighting)

| # | Model | Encoder | Params | Crop Size | Epochs | Mean Dice | Std |
|---|-------|---------|--------|-----------|--------|-----------|-----|
| 1 | **SegFormer** | MiT-B1 | **16.6M** | 160³ | 400 | **0.9288** | 0.0182 |
| 2 | Attention U-Net | DenseNet121 | 31.9M | 128³ | 400 | 0.8947 | 0.0576 |
| 3 | Attention U-Net | ResNet18 | 42.9M | 128³ | 200 | 0.8782 | 0.0417 |
| 4 | U-Net | ResNet18 | 42.6M | 128³ | 200 | 0.8227 | 0.0785 |

### Best Model — SegFormer (MiT-B1)

The SegFormer with a Mix Transformer B1 encoder achieves the highest Dice score (**0.9288**) with the fewest parameters (**16.6M**), demonstrating strong accuracy-to-parameter efficiency.

**Per-scan validation results:**

| Scan ID | Dice Score |
|---------|------------|
| be9fa3df918a | 0.9487 |
| 2eb903f4f204 | 0.9447 |
| 43ca4b1f6495 | 0.9417 |
| ed66d74f6692 | 0.9352 |
| 8781c0349102 | 0.9294 |
| d86566fdb41c | 0.9256 |
| 092c97ecec4b | 0.9164 |
| 4d9723eef946 | 0.8884 |

All 8 validation scans exceed **0.88** Dice. The train–validation gap of 0.02 indicates minimal overfitting despite only 39 training scans.

### Key Observations

- **SegFormer outperforms all U-Net variants** while using 2–2.5× fewer parameters, suggesting that the Mix Transformer encoder captures cardiac anatomy more efficiently than CNN-based encoders for this task.
- **Attention gates help** — both Attention U-Net variants outperform the vanilla U-Net, confirming that learned spatial attention benefits heart segmentation.
- **DenseNet121 > ResNet18** as an encoder backbone when paired with attention gates, likely due to denser feature reuse.
- **Longer training matters** — models trained for 400 epochs consistently outperform those trained for 200 epochs.

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (tested on Tesla T4)
- ~10 GB disk space for the COCA dataset subset

### Installation

```bash
git clone https://github.com/Arjun-bhandary/PrediCT_Gsoc2026.git
cd PrediCT_Gsoc2026

pip install nibabel matplotlib seaborn scikit-learn monai tqdm pandas
pip install git+https://github.com/innat/medic-ai.git
```

### Running the Notebooks

**Task 1 — Preprocessing:**
```bash
jupyter notebook Task_1/COCA_task1.ipynb
```
> Update `DATA_DIR` in the configuration cell to point to your local COCA dataset (NIfTI format from `COCA_processor.py` and `COCA_resampler.py`).

**Task 2 — Heart Segmentation:**
```bash
jupyter notebook "Task2/SegFormer_ mit_b1/train.ipynb"
```
> Update `CT_DIR` and `GT_DIR` to point to your cardiac CT inputs and TotalSegmentator ground-truth masks respectively.

### Data Preparation

1. Download the [COCA dataset](https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-b7e7-861c141a9ac7) from Stanford AIMI.
2. Run `COCA_processor.py` to convert DICOM volumes + XML annotations → NIfTI pairs.
3. Run `COCA_resampler.py` to resample to 0.7 mm isotropic spacing.
4. For Task 2, run [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) on a subset of scans to generate whole-heart ground-truth masks.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| [Keras 3](https://keras.io/) + PyTorch backend | Model training & inference |
| [medic-ai](https://github.com/innat/medic-ai) | SegFormer, U-Net, Attention U-Net architectures |
| [MONAI](https://monai.io/) | Medical image transforms, data loading, augmentation |
| [nibabel](https://nipy.org/nibabel/) | NIfTI file I/O |
| [SimpleITK](https://simpleitk.org/) | DICOM processing & resampling |
| [scikit-learn](https://scikit-learn.org/) | Stratified splitting |
| matplotlib / seaborn | Visualization |



