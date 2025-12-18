set -euo pipefail

#########################
# 1) Paths and basic setup
#########################

# Repository root (assuming this script lives in: <repo>/models/)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Virtual environment path (adapt to your own environment)
ENV_PATH="${REPO_ROOT}/.venv_ldm"

# Iris LDM config / checkpoint / manifest / output directory
CONFIG_PATH="${REPO_ROOT}/local_configs/latent-diffusion/iris_stageA_sota_64.yaml"
CKPT_PATH="/path/to/your/iris_stageA_sota_64.ckpt"    # TODO: change to your actual checkpoint path
CSV_PATH="/path/to/your/iris_manifest.csv"            # TODO: change to your actual manifest path
OUTDIR="${REPO_ROOT}/outputs/ldm_baseline_s0_5"       # TODO: change if you want a different output folder

mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/outputs"

#########################
# 2) Modules (example)
#########################

module purge
module load python3/3.10.16
module load cuda/11.7

# Activate virtual environment
source "${ENV_PATH}/bin/activate"

# Make sure the repo is on PYTHONPATH (for local configs and ldm code)
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

#########################
# 4) Run the generation script
#########################

cd "${REPO_ROOT}/latent-diffusion"

python img2img_by_all_per_class_masked.py \
  --config  "${CONFIG_PATH}" \
  --ckpt    "${CKPT_PATH}" \
  --csv     "${CSV_PATH}" \
  --classes 0-811 \
  --num_per_class 1 \
  --outdir "${OUTDIR}" \
  --ddim_steps 100 \
  --denoise_steps 50 \
  --eta 0.0 \
  --strength 0.5 \
  --bs 4 \
  --seed_base 0 \
  --mask_dilate_px 2 \
  --mask_feather_px 3