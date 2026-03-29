# Weakly-supervised segmentation using inherently-explainable classification models

Official code for the paper:

**"Weakly-supervised segmentation using inherently-explainable classification models and their application to brain tumour      classification"**
Published in *Neurocomputing* (2026) · DOI: [10.1016/j.neucom.2026.133460](https://doi.org/10.1016/j.neucom.2026.133460) · Preprint: [arXiv:2206.05148](https://arxiv.org/abs/2206.05148)

The initial idea as a proof-of-concept was first presented as an abstract at **ISMRM-ESMRMB 2022**, London. This journal paper is a full extension of that work - including comprehensive experiments on the full BraTS 2020 dataset, and thorough evaluation.
Abstract on ResearchGate: [Learning to segment brain tumours using an explainable classifier](https://www.researchgate.net/publication/358357555_Learning_to_segment_brain_tumours_using_an_explainable_classifier)

---

## Overview

GPModels are inherently-explainable convolutional networks for simultaneous brain-tumour **classification** and **weakly-supervised segmentation** from multi-contrast MRI.

- **Task:** Multi-class classification (Healthy / LGG / HGG) with weakly-supervised segmentation via per-pixel heatmaps
- **Dataset:** [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/)
- **Input:** 2-D axial slices - 4 channels in order **T1 · T2 · T1CE · FLAIR** (indices 0-3), 240 × 240 px, max-normalised per slice
- **Output classes:** 0 = Healthy, 1 = LGG, 2 = HGG

Three model families are provided:

| Model | Description |
|---|---|
| **GPUNet** | U-Net with global max-pooling classifier head |
| **GPReconResNet** | ReconResNet-based encoder-decoder with reconstruction path |
| **GPShuffleUNet** | U-Net with pixel-shuffle (sub-pixel convolution) upsampling |

---

## Pre-trained Models on Hugging Face

A collection of pre-trained weights is available on the Hugging Face Hub:

**Collection:** [huggingface.co/collections/soumickmj/gp-models](https://huggingface.co/collections/soumickmj/gp-models)

### 4-channel models (T1 · T2 · T1CE · FLAIR) - *from the paper*

| Model | Hub repository |
|---|---|
| GPUNet | [soumickmj/GPUNet_BraTS2020_T1T2T1ceFlair_Axial](https://huggingface.co/soumickmj/GPUNet_BraTS2020_T1T2T1ceFlair_Axial) |
| GPReconResNet | [soumickmj/GPReconResNet_BraTS2020_T1T2T1ceFlair_Axial](https://huggingface.co/soumickmj/GPReconResNet_BraTS2020_T1T2T1ceFlair_Axial) |
| GPShuffleUNet | [soumickmj/GPShuffleUNet_BraTS2020_T1T2T1ceFlair_Axial](https://huggingface.co/soumickmj/GPShuffleUNet_BraTS2020_T1T2T1ceFlair_Axial) |

### Single-contrast models (T1CE only) - *not part of the paper*

| Model | Hub repository |
|---|---|
| GPUNet | [soumickmj/GPUNet_BraTS2020T1ce_Axial](https://huggingface.co/soumickmj/GPUNet_BraTS2020T1ce_Axial) |
| GPReconResNet | [soumickmj/GPReconResNet_BraTS2020T1ce_Axial](https://huggingface.co/soumickmj/GPReconResNet_BraTS2020T1ce_Axial) |
| GPShuffleUNet | [soumickmj/GPShuffleUNet_BraTS2020T1ce_Axial](https://huggingface.co/soumickmj/GPShuffleUNet_BraTS2020T1ce_Axial) |

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `torchcomplex` and `tricorder` are installed from GitHub (see `requirements.txt`). A CUDA-capable GPU is strongly recommended.

---

## Dataset Preparation

Pre-processed BraTS 2020 data must be supplied as pickle files. Use the scripts in the `dataset/` folder to prepare the data:

```
dataset/
  Brats2020/
    Brats_PreprocessingV2.py   # converts raw NIfTI → per-slice pickles
```

The loader (`utilities/load.py`) stacks contrasts as `(T1, T2, T1CE, FLAIR)` → tensor shape `(4, H, W)`. Slices are max-normalised per channel before being passed to the model.

---

## Training

Main training/test entry point: `classifier.py` (PyTorch Lightning).

```bash
python classifier.py \
  --network GP_ShuffleUNet \
  --Dataset Brats20 \
  --contrast allCont \
  --orient Axi \
  --normmode 3 \
  --depth 3 \
  --wf 6 \
  --trainPicklePath /path/to/pickles/train \
  --valPicklePath   /path/to/pickles/val \
  --outPath         /path/to/output
```

Available `--network` values: `GP_UNet`, `GP_ReconResNet`, `GP_ShuffleUNet`.

Set `--model_segclassify` to enable joint segmentation + classification training (requires ground-truth masks).

Training is logged with [Weights & Biases](https://wandb.ai). Pass `--wandb_project <project>` to enable.

---

## Loading Hugging Face Weights into the Native Model Classes

The Hugging Face checkpoints wrap the native models inside a `PreTrainedModel` (key prefix `net.`). To use the weights directly with the model classes in this repository, strip that prefix:

```python
import torch
from huggingface_hub import hf_hub_download
from model.ShuffleUnet.GP_ShuffleUNet import GP_ShuffleUNet

def load_hf_weights(model, repo_id: str, filename: str = "pytorch_model.bin"):
    """Load HF-Hub weights into a native GPModels model instance."""
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    state_dict = torch.load(ckpt_path, map_location="cpu")

    # HF wrapper adds "net." prefix; Lightning checkpoint stores under "state_dict"
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("loss"):          # skip loss module weights
            continue
        new_key = k[len("net."):] if k.startswith("net.") else k
        cleaned[new_key] = v

    model.load_state_dict(cleaned)
    return model

# Example - GPShuffleUNet (4-channel)
net = GP_ShuffleUNet(d=2, in_ch=4, num_features=64, n_levels=3, out_ch=3)
net = load_hf_weights(net, "soumickmj/GPShuffleUNet_BraTS2020_T1T2T1ceFlair_Axial")
net.eval()

# Classification + heatmap inference
x = torch.randn(1, 4, 240, 240)   # (B, 4, H, W) - max-normalised axial slice
with torch.no_grad():
    logits, heatmap = net(x)       # eval mode returns both
    pred = logits.argmax(dim=1)    # 0=Healthy, 1=LGG, 2=HGG
    # heatmap: (B, 3, H, W) - one channel per class
```

> When the model is in **train mode** it returns only logits `(B, 3)`. Switch to `model.eval()` to obtain the `(logits, heatmap)` tuple.

---

## Segmentation Pipeline: Heatmap → Binary Mask

The raw heatmap from the model is a continuous activation map. To obtain a discrete segmentation mask, apply a threshold to convert it to a binary mask. The full post-processing pipeline is implemented in [`Local_Post-Processing.py`](Local_Post-Processing.py).

### Step 1 - Obtain the heatmap

```python
model.eval()
with torch.no_grad():
    logits, heatmap = model(x)          # x: (B, 4, 240, 240)
    pred_class = logits.argmax(dim=1)   # 0=Healthy, 1=LGG, 2=HGG
    # heatmap: (B, 3, H, W) - one activation map per class
    # Select the channel corresponding to the predicted class:
    seg_map = heatmap[torch.arange(len(pred_class)), pred_class]  # (B, H, W)
```

### Step 2 - Post-process to binary mask

```python
from skimage.filters import threshold_multiotsu
import numpy as np

def heatmap_to_mask(seg_map_np: np.ndarray, offset: float = 0.5) -> np.ndarray:
    """
    Convert a continuous heatmap (H, W) to a binary segmentation mask.
    Uses multi-Otsu thresholding with an adjustable offset.
    See Local_Post-Processing.py for full evaluation and offset selection.
    """
    thresholds = threshold_multiotsu(seg_map_np)
    binary = seg_map_np > (thresholds[-1] - offset)
    return binary.astype(np.uint8)

seg_np = seg_map[0].cpu().numpy()     # (H, W)
binary_mask = heatmap_to_mask(seg_np)
```

### Threshold selection and full evaluation

`Local_Post-Processing.py` performs a systematic sweep over threshold offset values (default range `[0.4, 1.0]`), computes Dice and IoU scores against ground-truth masks, and saves visualisations as NIfTI files for inspection. Run it (after adjusting the `main_path`, `network`, `Dataset`, and `orient` variables at the top of the file) to find the best offset for your data.

---

## Credits

If you like this repository, please click on Star!

If you use this approach in your research or use code from this repository, please cite the journal paper. You may also cite the conference abstract where the work was first presented.

### Journal paper (Neurocomputing 2026)

> Soumick Chatterjee, Hadya Yassin, Florian Dubost, Andreas Nürnberger, Oliver Speck: *Weakly-supervised segmentation using inherently-explainable classification models and their application to brain tumour classification.* Neurocomputing, 2026. DOI: [10.1016/j.neucom.2026.133460](https://doi.org/10.1016/j.neucom.2026.133460)

```bibtex
@article{chatterjee2026weakly,
  title={Weakly-supervised segmentation using inherently-explainable classification models and their application to brain tumour classification},
  author={Chatterjee, Soumick and Yassin, Hadya and Dubost, Florian and N{\"u}rnberger, Andreas and Speck, Oliver},
  journal={Neurocomputing},
  pages={133460},
  year={2026},
  publisher={Elsevier}
}
```

### Conference abstract (ISMRM-ESMRMB 2022)

The ISMRM abstract presented the **original proof-of-concept**. The journal paper above is the full extension of this work.

> Soumick Chatterjee, Hadya Yassin, Florian Dubost, Andreas Nürnberger, Oliver Speck: *Learning to segment brain tumours using an explainable classifier.* ISMRM-ESMRMB 2022, London, May 2022. [ResearchGate](https://www.researchgate.net/publication/358357555_Learning_to_segment_brain_tumours_using_an_explainable_classifier)

```bibtex
@inproceedings{mickISMRM22gp,
  author    = {Chatterjee, Soumick and Yassin, Hadya and Dubost, Florian and N{\"u}rnberger, Andreas and Speck, Oliver},
  year      = {2022},
  month     = {05},
  pages     = {0171},
  title     = {Learning to segment brain tumours using an explainable classifier},
  booktitle = {ISMRM-ESMRMB 2022}
}
```

Thank you so much for your support.
