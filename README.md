# MoCo ResNet-50 Domain Adaptation (Office-31: amazon/images → dslr/images)

Contrastive pretraining with **MoCo** on Office-31 images (ResNet‑50 backbone), followed by a supervised fine‑tune with light **consistency / pseudo‑label** regularization.

<p align="left">
  <img src="https://img.shields.io/badge/PyTorch-2.x-red" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/Dataset-Office--31-lightgrey" />
</p>

## Results (from this notebook)
- **MoCo pretraining:** 300 epochs (loss 5.2226 → 2.3861)
- **Fine‑tune:** Best Test Acc = **85.14%** (epoch 100); Final Test Acc = **83.94%**



## Data
Place Office‑31 under `./data/` as:
```
data/
  amazon/images/...
  dslr/images/...
```
The notebook uses **source** = `amazon/images`, **target** = `dslr/images`.

## Training (Notebook)
Open `MoCO-resnet50-Domain-adaptation-Github-Ready.ipynb` and run all cells in order.

### Phase 1 — MoCo Pretraining
- Backbone: **ResNet‑50**
- Projection dim: **128**
- Queue size **K**: **1024**
- Optimizer: **AdamW**, lr = **0.0005**
- Scheduler: **CosineAnnealingLR** (`T_max` = 300)
- Epochs: **300**
- Batch size: **64**, workers: **4**

### Phase 2 — Supervised Fine‑tune (target domain)
- Optimizer: **AdamW**, lr = **0.0001**
- Scheduler: **CosineAnnealingLR** (`T_max` = 100)
- Loss: **CrossEntropy** (label_smoothing = 0.1)
- **Consistency** (weight = 1.0) with **pseudo‑labels** (confidence ≥ 0.95)
- Epochs: **100**
- Batch size: **64**, workers: **4**

## Checkpoints
The notebook saves intermediate/best weights, e.g.:
- `moco_phase2_resnet50.pth` (pretrained encoder)
- `best_finetune_phase2_resnet50.pth` (best fine‑tuned classifier)

## Acknowledgments
- He et al., **MoCo**: Momentum Contrast for Unsupervised Visual Representation Learning.
- Office‑31 dataset.

---
*Notebook: `MoCO-resnet50-Domain-adaptation-Github-Ready.ipynb`*
