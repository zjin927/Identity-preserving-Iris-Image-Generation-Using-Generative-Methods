# Iris Image Generation UI User Guide

## Overview

This is a simple Gradio-based Web UI for generating synthetic iris images from reference images. The UI allows you to:
- Select an input reference image
- Select a mask image
- Adjust the **strength** parameter (controls the degree of change outside the iris region)
- Other parameters are fixed (from `img2img_generation_example.sh`)

## File Description

1. **`latent-diffusion/img2img_single_image.py`**: Single image generation script that does not require a CSV file
2. **`iris_generation_ui.py`**: Gradio Web UI main program

## Installation

Make sure all dependencies are installed (including gradio):

```bash
pip install -r requirements.txt
```

Or install gradio separately:

```bash
pip install gradio
```

## Usage

### 1. Launch the UI

```bash
python iris_generation_ui.py
```

The UI will automatically open in your browser at: `http://127.0.0.1:7860` (or another available port)

### 2. Configure Paths

On first use, configure the following in the "Advanced Settings" section of the UI:
- **Config Path**: e.g., `local_configs/latent-diffusion/iris_stageA_sota_64.yaml`
- **Checkpoint Path**: Path to your model checkpoint file (.ckpt file)
- **Class ID**: Class ID for conditional generation (default: 0)

### 3. Generate Images

1. Upload a **reference image** (original iris image)
2. Upload a **mask image** (binary image, white regions indicate iris area)
3. Adjust the **Strength** slider (0.0-1.0)
   - Higher values result in more changes outside the iris region
   - Lower values keep the generated image closer to the original
4. Click the "Generate Image" button
5. The generated image will be displayed on the right

## Fixed Parameters

The following parameters are fixed (from `img2img_generation_example.sh` lines 48-55):
- `ddim_steps`: 100
- `denoise_steps`: 30 (optimized from 50 for faster generation)
- `eta`: 0.0
- `mask_dilate_px`: 2
- `mask_feather_px`: 3
- `seed`: 0

## Output

Generated images are saved to: `outputs/ui_generated/generated_<timestamp>.png`

## Command Line Usage (Optional)

If you prefer not to use the UI, you can also use the command line script directly:

```bash
cd latent-diffusion
python img2img_single_image.py \
  --config ../local_configs/latent-diffusion/iris_stageA_sota_64.yaml \
  --ckpt /path/to/your/iris_stageA_sota_64.ckpt \
  --ref_img /path/to/reference/image.png \
  --mask /path/to/mask/image.png \
  --class_id 0 \
  --out /path/to/output.png \
  --strength 0.5
```

## Notes

- Make sure CUDA is properly configured (if using GPU)
- Reference and mask images should be 256x256 pixels (will be automatically resized if not)
- Mask images should be binary (black and white), with white regions indicating the iris area
- First run may take time to load the model, please be patient
