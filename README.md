# Identity-Preserving Iris Image Generation (Latent Diffusion)

This repository contains code to generate synthetic near-infrared (NIR) iris images using a latent diffusion model (LDM) in a masked *img2img* setting.  
The focus is on identity-preserving generation: the iris texture is kept stable, while the periocular region (outside the iris) can be modified with a controllable strength parameter.

The repository only includes **generation code**. Evaluation scripts and experimental pipelines are intentionally not included.

---

## 1. Repository structure

The most relevant folders and files are:

```text
.
├─ latent-diffusion/                  # Fork of CompVis/latent-diffusion with custom script
│  ├─ ldm/                            # LDM model and DDIM sampler implementations
│  ├─ configs/                        # Original LDM config files
│  ├─ img2img_by_all_per_class_masked.py   # Main masked img2img generation script
│  └─ ...
├─ local_configs/
│  ├─ autoencoder/
│  │  └─ iris_ae_kl_64.yaml          # Autoencoder (Stage A) config (used during training)
│  └─ latent-diffusion/
│     └─ iris_stageA_sota_64.yaml    # Iris-specific LDM config (Stage B)
├─ manifests/
│  └─ iris_manifest_example.csv      # Example manifest describing dataset layout
├─ models/
│  └─ # Put the downloaded models here
├─ img2img_generation_example.sh   # Example job script
├─ requirements.txt
└─ README.md
```

**Not provided in this repository:**

- The ND-LG4000-LR dataset should be required directly from the source.
- The Syntehtics image generated and trained model checkpoints (`.ckpt` / `.pth`) should be available in the following [Link].
- Evaluation and benchmarking scripts.

Users are expected to provide their own data, masks, and trained checkpoints, in accordance with the respective licenses. You can access our trained checkpoints by contacting xxxx@xxx.xx

---

## 2. Installation

### 2.1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2.2. Create and activate a virtual environment

Example with Python 3.10:

```bash
python3 -m venv .venv_ldm
source .venv_ldm/bin/activate   # On Windows: .venv_ldm\Scripts\activate
```

### 2.3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file installs PyTorch, PyTorch Lightning, and the dependencies of the latent diffusion implementation, as well as optional packages for FID/LPIPS computation.

> Note: `requirements.txt` includes an editable install of the original latent-diffusion repository:
>
> ```text
> -e git+https://github.com/CompVis/latent-diffusion@...#egg=latent_diffusion
> ```
>
> This will pull the upstream code and apply it as a local package.  
> Iris-specific configs and scripts in this repository extend that implementation.

---

## 3. Data and manifest format

The repository does **not** contain any iris images or masks.  
Instead, data is described through a **CSV manifest**.

An example manifest is provided in:

```text
manifests/iris_manifest_example.csv
```

The required columns are:

- `img_path` – path to the original iris image (absolute or relative).
- `mask_path` – path to the corresponding iris segmentation mask (same resolution as `img_path`).
- `class_name` – subject identifier (e.g., `L02463`).
- `class_id` – integer subject ID (0-based).
- `split` – dataset split label (e.g., `train`, `val`, `test`).

The generation script assumes that:

- The same `class_id` identifies all images belonging to one subject.
- `class_name` can be used to distinguish left vs. right eye (e.g., left-eye classes starting with `L`); if no left-eye images are found, all images for that subject are used.
- The `split` column allows selecting a subset of the data for generation.

A minimal example row (for illustration only):

```csv
img_path,mask_path,class_name,class_id,split
/path/to/images/L02463d1890.png,/path/to/masks/L02463d1890.png,L02463,0,train
```

You should adapt the manifest paths and naming convention to your own dataset.

---

## 4. Pretrained models and configs

The LDM generation pipeline assumes a two-stage setup:

1. **Stage A – AutoencoderKL (AE-KL)**
   - Compresses a 256×256 iris image into a low-dimensional latent representation.
   - Configuration: `local_configs/autoencoder/iris_ae_kl_64.yaml`.

2. **Stage B – Latent Diffusion Model (LDM)**
   - Operates in latent space.
   - For this project, a class-conditional LDM is trained on iris identities.
   - Configuration: `local_configs/latent-diffusion/iris_stageA_sota_64.yaml`.

**Model weights (checkpoints) are not distributed in this repository.**  
You must train your own models or obtain compatible checkpoints and set their paths when running the generation script.

---

## 5. Masked *img2img* generation

The main entry point for generation is:

```text
latent-diffusion/img2img_by_all_per_class_masked.py
```

This script:

- Loads a pretrained LDM and its underlying autoencoder.
- Uses a CSV manifest to iterate over subjects (`class_id`).
- For each subject, selects one or more reference images (per class).
- Applies **masked img2img**:
  - The iris region (defined by the binary mask) is kept close to the autoencoder reconstruction.
  - Noise is injected only outside the iris region.
  - A `strength` parameter controls how deep in the diffusion process the starting point is, thus controlling the magnitude of changes in the periocular region.
- Saves generated images to a specified output directory with a stable naming scheme.

### 5.1. Basic usage (local run)

From the repository root:

```bash
cd latent-diffusion

python img2img_by_all_per_class_masked.py   --config  ../local_configs/latent-diffusion/iris_stageA_sota_64.yaml   --ckpt    /path/to/your/iris_stageA_sota_64.ckpt   --csv     /path/to/your/iris_manifest.csv   --split   train   --classes 0-811   --num_per_class 1   --outdir  ../outputs/ldm_s0_5_all_masked   --ddim_steps 100   --denoise_steps 50   --eta 0.0   --strength 0.5   --bs 4   --seed_base 0   --mask_dilate_px 2   --mask_feather_px 3
```

Key arguments:

- `--config` – LDM configuration file (YAML).
- `--ckpt` – path to the trained LDM checkpoint.
- `--csv` – path to the dataset manifest.
- `--split` – which subset of the manifest to use (e.g. `train`, `val`).
- `--classes` – which `class_id`s to generate for, e.g.:
  - `0-811` for all classes in range.
  - `0-9,20-30` for a subset.
- `--num_per_class` – number of synthetic images to generate per class.
- `--outdir` – output root directory.
- `--ddim_steps` – number of DDIM sampling steps (time grid resolution).
- `--denoise_steps` – number of reverse diffusion steps actually performed.
- `--eta` – stochasticity parameter in DDIM (0.0 for deterministic sampling).
- `--strength` – starting point on the diffusion time grid (0–1); higher values correspond to stronger changes outside the iris.
- `--mask_dilate_px` – number of pixels to dilate the iris mask (to safely cover the iris region).
- `--mask_feather_px` – feathering width in pixels for a smooth mask boundary.
- `--bs` – batch size.
- `--seed_base` – base random seed for reproducibility.

Generated images are written to `--outdir`, typically organised by class.

---

## 6. Example: running on HPC

An example LSF job script is provided in:

```text
img2img_generation_example.sh
```

It assumes a HPC module system and a virtual environment located at `<repo>/.venv_ldm`.  
You must adapt at least:

- `ENV_PATH` – path to your virtual environment.
- `CKPT_PATH` – path to your LDM checkpoint.
- `CSV_PATH` – path to your iris manifest CSV.

To submit the job:

```bash
cd /path/to/your/repo
bsub < img2img_gemeratopm_example.sh
```

The script will write logs to `logs/<JOBID>_ldm_img2img.out` and `logs/<JOBID>_ldm_img2img.err` (relative to the repository root).

---

## 7. License and acknowledgements

This repository builds upon the official **latent-diffusion** implementation by [CompVis](https://github.com/CompVis/latent-diffusion).  
Please refer to their repository for the original code license and model weight usage terms.

- Code in `latent-diffusion/` is derived from the CompVis implementation with minor adaptations for the iris generation setting.
- Any datasets and trained models used with this code must respect their respective licenses and usage agreements.

If you use this code in scientific work, please consider citing:

...

## 8. Disclaimer
These resources are available only for research purposes. In case any question contact: xxxxx@
