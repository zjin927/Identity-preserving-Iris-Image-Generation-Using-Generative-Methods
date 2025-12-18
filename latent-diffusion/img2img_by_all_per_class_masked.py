import os, argparse, numpy as np, torch, pandas as pd, cv2
from PIL import Image
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


# ---------- utilities ----------
def parse_classes(spec, all_ids):
    if spec.lower() == "all":
        return sorted(all_ids)
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    out = [i for i in out if i in set(all_ids)]
    return sorted(list(dict.fromkeys(out)))


def imread_to_tensor_3ch_01(path, H=256, W=256):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert g is not None, path
    if (g.shape[0], g.shape[1]) != (H, W):
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
    g = g.astype(np.float32) / 255.0
    x = np.stack([g, g, g], axis=-1)                # HWC
    x = torch.from_numpy(x).permute(2, 0, 1)[None]  # 1,3,H,W
    x = x * 2.0 - 1.0
    return x


def save_img01(x01, out_path):
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


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--classes", required=True)
    # Current semantics: how many images to generate per input image
    # (previously this was interpreted as "how many per class").
    ap.add_argument("--num_per_class", type=int, default=1)
    ap.add_argument("--outdir", required=True)

    # img2img hyperparameters
    ap.add_argument("--ddim_steps", type=int, default=100, help="DDIM time-grid resolution")
    ap.add_argument("--denoise_steps", type=int, default=50, help="Fixed denoising steps (number of U-Net calls)")
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--strength", type=float, default=0.02, help="0~1, noise injection depth")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--W", type=int, default=256)
    # ref_pick is no longer used to select a unique per-class reference image,
    # but kept for backward compatibility.
    ap.add_argument("--ref_pick", choices=["first", "random"], default="first")

    # Mask construction parameters
    ap.add_argument("--mask_dilate_px", type=int, default=2,
                    help="Dilation (in pixels) for locked region, to better protect iris boundaries")
    ap.add_argument("--mask_feather_px", type=int, default=3,
                    help="Feathering to soften boundaries and avoid seams")
    ap.add_argument("--mask_invert", action="store_true",
                    help="Enable if the dataset mask semantics are inverted")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Read CSV
    df = pd.read_csv(args.csv)
    if args.split is not None:
        df = df[df["split"] == args.split].reset_index(drop=True)

    assert "img_path" in df.columns and "mask_path" in df.columns and "class_id" in df.columns, \
        "CSV must contain img_path, mask_path, class_id"

    id2name = (df[["class_id", "class_name"]]
               .drop_duplicates()
               .sort_values("class_id")
               .set_index("class_id")["class_name"].to_dict())
    all_ids = sorted(id2name.keys())
    sel_ids = parse_classes(args.classes, all_ids)
    assert len(sel_ids) > 0, "No matching class IDs found."

    # Build: class -> list of reference images; and img_path -> mask_path mapping
    pools = {}
    for cid in sel_ids:
        g = df[df["class_id"] == cid]
        if "class_name" in g.columns:
            gl = g[g["class_name"].astype(str).str.startswith("L")]
            pools[cid] = (gl if len(gl) > 0 else g)["img_path"].tolist()
        else:
            pools[cid] = g["img_path"].tolist()
        assert len(pools[cid]) > 0, f"class {cid} has no reference images."

    path2mask = dict(zip(df["img_path"], df["mask_path"]))

    # 2) Load model
    cfg = OmegaConf.load(args.config)
    model = instantiate_from_config(cfg.model)
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f">> ckpt loaded. missing={len(missing)}, unexpected={len(unexpected)}")

    model.cuda().eval()
    sampler = DDIMSampler(model)

    # Latent-space info
    C = cfg.model.params.channels if "channels" in cfg.model.params else 4
    S_lat = cfg.model.params.image_size if "image_size" in cfg.model.params else 64
    latent_shape = (C, S_lat, S_lat)
    print(f">> latent_shape={latent_shape}")

    # === Fixed schedule (time grid) ===
    steps = args.ddim_steps
    sampler.make_schedule(ddim_num_steps=steps, ddim_eta=args.eta, verbose=False)

    # strength -> t_enc_idx
    assert 0.0 <= args.strength <= 1.0
    t_enc_idx = int(round(args.strength * (steps - 1)))
    t_enc_idx = max(0, min(steps - 1, t_enc_idx))

    decode_steps = max(1, int(args.denoise_steps))

    print(f">> schedule_steps={steps}, strength={args.strength} -> "
          f"t_enc_idx={t_enc_idx}, ddpm_t={sampler.ddim_timesteps[t_enc_idx].item()}, "
          f"denoise_steps={decode_steps}")

    # 3) Sampling loop (encode + decode_with_fixed_steps + mask)
    try:
        ema_ctx = model.ema_scope()
    except Exception:
        from contextlib import nullcontext
        ema_ctx = nullcontext()

    global_idx = 0
    with torch.no_grad(), ema_ctx:
        device = torch.device("cuda")

        for cid in sel_ids:
            cls_dir = os.path.join(args.outdir, f"class_{int(cid):02d}")
            os.makedirs(cls_dir, exist_ok=True)

            # Class-conditional embedding (shared within this class)
            labels = torch.tensor([int(cid)], device=device, dtype=torch.long)
            cond_embed_1 = model.get_learned_conditioning({"class_label": labels})  # (1,1,256)

            pool = pools[cid]
            print(f"[class {cid:03d}] num_real_imgs={len(pool)}")

            # Key change: generate for each real image in this class (not a single per-class reference)
            for img_idx, ref_path in enumerate(pool):
                mask_path = path2mask[ref_path]
                print(f"  - [{cid:03d}] img_idx={img_idx:03d} ref={ref_path}  mask={mask_path}")

                # Build latent mask: 1 = lock iris; 0 = editable region
                keep_mask_lat = build_latent_keep_mask(
                    mask_path, H_img=args.H, W_img=args.W, S_lat=S_lat,
                    dilate_px=args.mask_dilate_px, feather_px=args.mask_feather_px,
                    invert=args.mask_invert
                ).to(device)  # (1,1,S,S)

                # For this reference image, generate num_per_class samples
                left = args.num_per_class
                rep_idx = 0  # generation index for this reference image

                while left > 0:
                    cur_bs = min(args.bs, left)

                    # Reference image -> latent
                    x_ref = imread_to_tensor_3ch_01(ref_path, H=args.H, W=args.W).to(device)
                    z0_single = model.get_first_stage_encoding(model.encode_first_stage(x_ref))  # (1,4,S,S)
                    z0 = z0_single.repeat(cur_bs, 1, 1, 1)  # (B,4,S,S)

                    # Batch mask
                    mask_b = keep_mask_lat.repeat(cur_bs, 1, 1, 1)  # (B,1,S,S)

                    # Independent noise per sample
                    noise = torch.empty_like(z0)
                    for b in range(cur_bs):
                        g = torch.Generator(device=device)
                        g.manual_seed(args.seed_base + global_idx + b)
                        noise[b] = torch.randn(
                            z0.shape[1:], dtype=z0.dtype, generator=g, device=device
                        )

                    # Encode: forward noising
                    z_t = sampler.stochastic_encode(z0, t_enc_idx, noise=noise)

                    # Decode: fixed number of steps + lock iris region via mask
                    cond_embed = cond_embed_1.expand(cur_bs, -1, -1)
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
                        mask=mask_b,   # lock iris region
                        x0=z0          # use q_sample(x0, t) to overwrite the mask=1 region
                    )

                    # Decode to pixel space
                    x_rec = model.decode_first_stage(samples)
                    x01 = (x_rec + 1.0) / 2.0

                    # Save
                    base_name = os.path.splitext(os.path.basename(ref_path))[0]
                    for b in range(cur_bs):
                        out_path = os.path.join(
                            cls_dir,
                            f"{int(cid):03d}_{img_idx:03d}_{rep_idx + b:02d}.png"
                        )
                        save_img01(x01[b:b+1], out_path)

                    global_idx += cur_bs
                    rep_idx += cur_bs
                    left -= cur_bs

    print(">> Done.")


if __name__ == "__main__":
    main()
