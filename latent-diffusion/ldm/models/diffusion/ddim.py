"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def decode(self,
               x_t,
               cond,
               t_enc_idx,
               img_callback=None,
               log_every_t=100,
               unconditional_guidance_scale=1.0,
               unconditional_conditioning=None,
               quantize_x0=False,
               ):
        """
        从时间 index=t_enc_idx 对应的 x_t 一路去噪回 t=0。
        使用当前的 self.ddim_timesteps 的前缀 [0..t_enc_idx] 作为 schedule。

        注意：这里的“去噪步数” = t_enc_idx+1（跟加噪深度一起变），
        但 schedule 结构本身是固定的（ddim_timesteps 不重建）。
        """
        device = self.model.betas.device
        shape = x_t.shape
        b = shape[0]

        assert hasattr(self, "ddim_timesteps"), "call make_schedule(...) before decode"

        total_ddim_steps = self.ddim_timesteps.shape[0]

        # 合法化 index
        t_enc_idx = int(t_enc_idx)
        t_enc_idx = max(0, min(total_ddim_steps - 1, t_enc_idx))

        # 利用 ddim_sampling 里 timesteps 参数的逻辑：
        #   timesteps (标量) -> 取 self.ddim_timesteps 的前缀
        #   subset_end = timesteps-1
        #   timesteps = self.ddim_timesteps[:subset_end]
        #
        # 我们想要最后一个 index = t_enc_idx，对应前缀长度 = t_enc_idx+1
        # 所以设传入 timesteps_arg = t_enc_idx + 2
        timesteps_arg = min(t_enc_idx + 2, total_ddim_steps)

        samples, intermediates = self.ddim_sampling(
            cond,
            shape,
            x_T=x_t,
            ddim_use_original_steps=False,
            timesteps=timesteps_arg,
            quantize_denoised=quantize_x0,
            img_callback=img_callback,
            log_every_t=log_every_t,
            temperature=1.0,
            noise_dropout=0.0,
            score_corrector=None,
            corrector_kwargs=None,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
        )
        return samples

    @torch.no_grad()
    def stochastic_encode(self, x0, t_enc_idx, noise=None):
        """
        按当前已有的 self.ddim_timesteps，把 x0 前向加噪到 index=t_enc_idx 对应的时间步。
        - x0: (B,C,H,W) latent
        - t_enc_idx: int, 0 <= t_enc_idx < len(self.ddim_timesteps)
        - noise: (B,C,H,W)，不传就自己采样
        返回: x_t
        """
        assert hasattr(self, "ddim_timesteps"), "call make_schedule(...) before stochastic_encode"

        device = self.model.betas.device
        b = x0.shape[0]

        # 合法化 index
        t_enc_idx = int(t_enc_idx)
        t_enc_idx = max(0, min(self.ddim_timesteps.shape[0] - 1, t_enc_idx))

        # 取出这个 index 对应的 ddpm 时间步（整数）
        ddpm_t = self.ddim_timesteps[t_enc_idx]
        ts = torch.full((b,), ddpm_t, device=device, dtype=torch.long)

        # 噪声
        if noise is None:
            noise = noise_like(x0.shape, device, repeat_noise=False)

        # 用模型自带的 q_sample 做前向加噪
        x_t = self.model.q_sample(x0, ts, noise=noise)
        return x_t

    # --- ADD: 从 t_enc_idx 开始，用“固定去噪步数”往 0 去噪；支持 mask 锁定区域 ---
    @torch.no_grad()
    def decode_with_fixed_steps(self,
                                x_t,
                                cond,
                                t_enc_idx,
                                decode_steps,
                                unconditional_guidance_scale=1.0,
                                unconditional_conditioning=None,
                                quantize_x0=False,
                                img_callback=None,
                                log_every_t=100,
                                mask=None,
                                x0=None):
        """
        固定去噪步数的解码：从 index=t_enc_idx 对应的 x_t 走 decode_steps 步回到 ~0。
        - 采用 self.ddim_timesteps 的 index 作为“时间网格”；在 [t_enc_idx -> 0] 区间线性取 decode_steps 个 index。
        - 若传入 mask & x0：在每一步都用 img = q_sample(x0, t) 的 masked 混合来“锁住”被遮罩区域。
          注意：mask数值应在[0,1]，mask=1 表示“锁住/保持原图”（即用 q_sample(x0,t) 覆盖）。
        """
        assert hasattr(self, "ddim_timesteps"), "call make_schedule(...) before decode_with_fixed_steps"

        device = self.model.betas.device
        img = x_t
        b = img.shape[0]

        total_ddim_steps = self.ddim_timesteps.shape[0]
        t_enc_idx = int(t_enc_idx)
        t_enc_idx = max(0, min(total_ddim_steps - 1, t_enc_idx))
        decode_steps = max(1, int(decode_steps))

        # 生成固定长度的 index 序列：t_enc_idx → 0
        index_seq = np.linspace(t_enc_idx, 0, decode_steps, dtype=np.int64)
        index_seq = np.clip(index_seq, 0, total_ddim_steps - 1)

        intermediates = {'x_inter': [img], 'pred_x0': [img]}

        for i, index_int in enumerate(index_seq):
            step = self.ddim_timesteps[int(index_int)]
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            # 如果有掩膜：用 q_sample(x0, ts) 的对应时间步替换被锁定区域
            if mask is not None:
                assert x0 is not None, "mask 混合需要提供 x0=z0"
                img_orig = self.model.q_sample(x0, ts)  # 与当前 ts 对齐
                img = img_orig * mask + (1. - mask) * img

            img, pred_x0 = self.p_sample_ddim(
                img, cond, ts,
                index=int(index_int),
                use_original_steps=False,
                quantize_denoised=quantize_x0,
                temperature=1.0,
                noise_dropout=0.0,
                score_corrector=None,
                corrector_kwargs=None,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning
            )

            if img_callback is not None:
                img_callback(pred_x0, i)
            if (i % log_every_t == 0) or (i == decode_steps - 1):
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img
