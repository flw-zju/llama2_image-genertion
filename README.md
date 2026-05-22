---

This is a simple PyTorch implementation of **Llama2**, designed for integration with **VQGAN** and **DiT** models.

---

## 1. VQGAN (`vqgan_sampling.py`)
- Llama2 is used for sequence generation.
- Added **head-specific elementwise gated attention**.
- Trained on the **AFHQ dataset** (128×128 resolution).

### Generated Samples
![VQGAN Samples](assets/vqgan_samples.png)

---

## 2. DiT (`rf_sampling.py`)
- Llama2 serves as the transformer backbone.
- Removed **KV-cache**; added **rectified flow timestep**.
- Replaced 1D RoPE with **2D RoPE**.
- Trained on the **Celeba-HQ dataset** (256×256 resolution) for unconditional generation.

### Pretrained Model Weights (OneDrive)
You can download the pretrained model files from OneDrive:
[Download](https://1drv.ms/f/c/2c0f1036b31b3ed6/IgCuByrJxeCsRJeZe_6O7uQ2ATF2ca7i7iK9bjrNYddL6Os?e=bK3p2b)

