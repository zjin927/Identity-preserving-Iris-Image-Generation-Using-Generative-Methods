"""
Single image img2img generation script (wrapper around img2img_by_all_per_class_masked.py logic).
This script allows generating synthetic images from a single reference image without requiring a CSV manifest.
"""
import os
import sys
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path

# Add taming-transformers to path so taming module can be imported
script_dir = Path(__file__).parent.absolute()
taming_transformers_path = script_dir / "taming-transformers"
if str(taming_transformers_path) not in sys.path:
    sys.path.insert(0, str(taming_transformers_path))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def imread_to_tensor_3ch_01(path, H=256, W=256):
    """Load image and convert to tensor in [-1, 1] range."""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert g is not None, f"Image not found: {path}"
    if (g.shape[0], g.shape[1]) != (H, W):
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
    g = g.astype(np.float32) / 255.0
    x = np.stack([g, g, g], axis=-1)                # HWC
    x = torch.from_numpy(x).permute(2, 0, 1)[None]  # 1,3,H,W
    x = x * 2.0 - 1.0
    return x


def save_img01(x01, out_path):
    """Save tensor in [0, 1] range to image file."""
    x = x01.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    img = (x * 255.0).round().astype(np.uint8)
    Image.fromarray(img).save(out_path)


def build_latent_keep_mask(mask_path, H_img, W_img, S_lat,
                           dilate_px=2, feather_px=3, invert=False):
    """
    Build a latent-scale "keep/lock" mask (1 = locked) from a pixel-space binary iris mask.
    This mask is used to preserve the iris region during latent-space sampling.
    """
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    assert m is not None, f"Mask not found: {mask_path}"
    if (m.shape[0], m.shape[1]) != (H_img, W_img):
        m = cv2.resize(m, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

    m_bin = (m > 127).astype(np.uint8)  # 1 = iris
    if invert:
        m_bin = 1 - m_bin

    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
        )
        m_bin = cv2.dilate(m_bin, k, iterations=1)

    if feather_px > 0:
        m_soft = cv2.GaussianBlur(
            (m_bin * 255).astype(np.uint8), (0, 0),
            sigmaX=feather_px, sigmaY=feather_px
        )
        m_soft = m_soft.astype(np.float32) / 255.0
    else:
        m_soft = m_bin.astype(np.float32)

    # Downsample to latent resolution
    m_lat = cv2.resize(m_soft, (S_lat, S_lat), interpolation=cv2.INTER_AREA)
    m_lat = np.clip(m_lat, 0.0, 1.0)

    t = torch.from_numpy(m_lat)[None, None, :, :]  # (1,1,S,S)
    return t


def load_model(config_path, ckpt_path, device=None):
    """
    Load model and return model, sampler, and config.
    
    Args:
        config_path: Path to LDM config YAML
        ckpt_path: Path to model checkpoint
        device: torch device (if None, auto-detect)
    
    Returns:
        model, sampler, cfg
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f">> Loading model from {ckpt_path}...")
    cfg = OmegaConf.load(config_path)
    model = instantiate_from_config(cfg.model)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f">> Model loaded. missing={len(missing)}, unexpected={len(unexpected)}")

    model.to(device).eval()
    sampler = DDIMSampler(model)
    
    return model, sampler, cfg


def generate_single_image(
    config_path,
    ckpt_path,
    ref_img_path,
    mask_path,
    class_id,
    out_path,
    strength=0.5,
    ddim_steps=100,
    denoise_steps=50,
    eta=0.0,
    mask_dilate_px=2,
    mask_feather_px=3,
    seed=0,
    H=256,
    W=256,
    model=None,
    sampler=None,
    cfg=None
):
    """
    Generate a single synthetic image from a reference image.
    
    Args:
        config_path: Path to LDM config YAML (only used if model is None)
        ckpt_path: Path to model checkpoint (only used if model is None)
        ref_img_path: Path to reference image
        mask_path: Path to iris mask (binary image)
        class_id: Class ID for conditioning
        out_path: Output image path
        strength: Noise injection depth (0-1), controls changes outside iris
        ddim_steps: DDIM time-grid resolution
        denoise_steps: Number of denoising steps
        eta: DDIM stochasticity parameter
        mask_dilate_px: Mask dilation in pixels
        mask_feather_px: Mask feathering in pixels
        seed: Random seed
        H, W: Image dimensions
        model: Pre-loaded model (optional, for caching)
        sampler: Pre-loaded sampler (optional, for caching)
        cfg: Pre-loaded config (optional, for caching)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model if not provided
    if model is None or sampler is None or cfg is None:
        model, sampler, cfg = load_model(config_path, ckpt_path, device)

    # Latent-space info
    C = cfg.model.params.channels if "channels" in cfg.model.params else 4
    S_lat = cfg.model.params.image_size if "image_size" in cfg.model.params else 64
    latent_shape = (C, S_lat, S_lat)
    print(f">> latent_shape={latent_shape}")

    # Setup DDIM schedule
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False)

    # strength -> t_enc_idx
    assert 0.0 <= strength <= 1.0
    t_enc_idx = int(round(strength * (ddim_steps - 1)))
    t_enc_idx = max(0, min(ddim_steps - 1, t_enc_idx))

    decode_steps = max(1, int(denoise_steps))

    print(f">> schedule_steps={ddim_steps}, strength={strength} -> "
          f"t_enc_idx={t_enc_idx}, ddpm_t={sampler.ddim_timesteps[t_enc_idx].item()}, "
          f"denoise_steps={decode_steps}")

    # Build latent mask
    keep_mask_lat = build_latent_keep_mask(
        mask_path, H_img=H, W_img=W, S_lat=S_lat,
        dilate_px=mask_dilate_px, feather_px=mask_feather_px,
        invert=False
    ).to(device)  # (1,1,S,S)

    # Generate
    with torch.no_grad():
        try:
            ema_ctx = model.ema_scope()
        except Exception:
            from contextlib import nullcontext
            ema_ctx = nullcontext()

        with ema_ctx:
            # Class-conditional embedding
            labels = torch.tensor([int(class_id)], device=device, dtype=torch.long)
            cond_embed = model.get_learned_conditioning({"class_label": labels})  # (1,1,256)

            # Reference image -> latent
            x_ref = imread_to_tensor_3ch_01(ref_img_path, H=H, W=W).to(device)
            z0 = model.get_first_stage_encoding(model.encode_first_stage(x_ref))  # (1,4,S,S)

            # Generate noise
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            noise = torch.randn(
                z0.shape, dtype=z0.dtype, generator=g, device=device
            )

            # Encode: forward noising
            z_t = sampler.stochastic_encode(z0, t_enc_idx, noise=noise)

            # Decode: fixed number of steps + lock iris region via mask
            samples = sampler.decode_with_fixed_steps(
                z_t,
                cond=cond_embed,
                t_enc_idx=t_enc_idx,
                decode_steps=decode_steps,
                unconditional_guidance_scale=1.0,
                unconditional_conditioning=None,
                quantize_x0=False,
                img_callback=None,
                log_every_t=100,
                mask=keep_mask_lat,   # lock iris region
                x0=z0                 # use q_sample(x0, t) to overwrite the mask=1 region
            )

            # Decode to pixel space
            x_rec = model.decode_first_stage(samples)
            x01 = (x_rec + 1.0) / 2.0

            # Save
            os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
            save_img01(x01, out_path)
            print(f">> Generated image saved to: {out_path}")

    return out_path


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic iris image from single reference image")
    ap.add_argument("--config", required=True, help="Path to LDM config YAML")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    ap.add_argument("--ref_img", required=True, help="Path to reference image")
    ap.add_argument("--mask", required=True, help="Path to iris mask (binary image)")
    ap.add_argument("--class_id", type=int, required=True, help="Class ID for conditioning")
    ap.add_argument("--out", required=True, help="Output image path")
    ap.add_argument("--strength", type=float, default=0.5, help="Noise injection depth (0-1)")
    ap.add_argument("--ddim_steps", type=int, default=100, help="DDIM time-grid resolution")
    ap.add_argument("--denoise_steps", type=int, default=50, help="Number of denoising steps")
    ap.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity parameter")
    ap.add_argument("--mask_dilate_px", type=int, default=2, help="Mask dilation in pixels")
    ap.add_argument("--mask_feather_px", type=int, default=3, help="Mask feathering in pixels")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--H", type=int, default=256, help="Image height")
    ap.add_argument("--W", type=int, default=256, help="Image width")
    
    args = ap.parse_args()
    
    generate_single_image(
        config_path=args.config,
        ckpt_path=args.ckpt,
        ref_img_path=args.ref_img,
        mask_path=args.mask,
        class_id=args.class_id,
        out_path=args.out,
        strength=args.strength,
        ddim_steps=args.ddim_steps,
        denoise_steps=args.denoise_steps,
        eta=args.eta,
        mask_dilate_px=args.mask_dilate_px,
        mask_feather_px=args.mask_feather_px,
        seed=args.seed,
        H=args.H,
        W=args.W
    )


if __name__ == "__main__":
    main()

