#!/usr/bin/env python3
"""
Qwen-Image 2.1 reference pipeline, for parity checks of the C code.

Runs the official diffusers components (Qwen3-VL text encoding through
QwenImage21Pipeline.encode_prompt, QwenImage21Transformer2DModel with its
KV cache, FlowMatchEulerDiscreteScheduler, AutoencoderKLQwenImage21) the
way QwenImage21Pipeline.__call__ does for text-to-image, in float32, and
dumps the intermediate tensors as raw little-endian float32.

The text encoder is freed before the transformer loads, and the
transformer before the VAE, so a float32 run fits in 64 GB.

Usage:
    python debug/qimage_reference.py -d ./qwen-image-2.1-turbo \\
        -p "a fish" -W 256 -H 256 --seed 42 --blocks -o /tmp/qimage_ref

    # Start from the noise iris generates for a seed (see debug/dump_noise.c):
    python debug/qimage_reference.py ... --noise /tmp/noise.bin

    # Only decode a latent [64, H/16, W/16]:
    python debug/qimage_reference.py -d MODEL -W 256 -H 256 --decode latent.bin -o OUT

    # Reference images (repeat --image), at --ref-resolution square-equivalent:
    python debug/qimage_reference.py -d MODEL -p "..." --image a.png --ref-resolution 512 ...

Outputs in the -o directory:
    tokens.txt, drop_idx.txt  encoder token ids (whole template) and the
                              number of leading system tokens dropped
    emb.bin                   text embeddings after the drop [L, 4096]
    noise.bin                 initial noise [64, H/16, W/16]
    sigmas.txt                scheduler sigmas (num_steps + 1 values)
    rope_cos.bin, rope_sin.bin  first-step RoPE tables [L + N, 64]
    step<i>_v.bin             velocity at step i [N, 64]
    step<i>_x.bin             latent tokens after step i [N, 64]
    block<l>_s<i>.bin         (--blocks) hidden states after block l at steps
                              0 and 1: [L + N, dim] at step 0, [N, dim] after
    latent.bin                final latent [64, H/16, W/16] (normalized)
    image.bin, image.png      decoded image [4, H, W] in [-1, 1]

With --image, also:
    ref<i>_rgba.bin           VAE input [4, H_i, W_i] in [-1, 1]
    ref<i>_vision_rgb.bin     vision input, over white, [3, H_i, W_i] in 0..255
    ref<i>_latent.bin         normalized VAE mean [64, H_i/16, W_i/16]
    enc_<stage>.bin           first reference: conv_in, down0..4, mid, norm_out,
                              conv_out, quant [C, h, w]
    pixel_values.bin, grid_thw.txt  processor output [patches, 1536]
    vis_patch.bin, vis_in.bin  patch embedding, and with position embedding
    vis_rope_cos.bin, vis_rope_sin.bin, vis_block<l>.bin (l = 0, 8, 16, 24, 26)
    deep<k>.bin, image_embeds.bin   deepstack and main merger outputs
    position_ids.bin          int32, shape printed
    txt_embeds.bin, txt_layer<l>.bin (l = 0, 1, 2, 35), txt_layer3_in.bin
                              layer outputs are before the deepstack addition,
                              txt_layer3_in.bin is after it
    image_pad_mask.txt        1 for slot rows of emb.bin
    segments.txt              prefill segments: start end is_text
    In this mode emb.bin keeps the slot rows and block dumps cover the whole
    prefix at step 0, as upstream does.

L is the text length after the drop, N = (H/16) * (W/16). The C side reads
emb.bin and noise.bin with: ./iris -d MODEL -e emb.bin --noise noise.bin
-W W -H H -o out.png. Pass --emb FILE to skip the text encoder.
"""

import argparse
import gc
import json
import os

import numpy as np
import torch


def dump(path, t):
    t.detach().to(torch.float32).contiguous().cpu().numpy().tofile(path)


def prepare_refs(paths, resolution, out):
    """The condition image preparation of QwenImage21Pipeline.__call__.
    Returns (images for the vision encoder, VAE input tensors)."""
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
    from PIL import Image

    proc = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
    vision, vae_in = [], []
    for i, path in enumerate(paths):
        img = Image.open(path)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        w, h, _ = calculate_dimensions(resolution * resolution, img.size[0] / img.size[1])
        resized = proc.resize(img, width=w, height=h)
        vision.append(resized)
        vae_in.append(proc.preprocess(img, width=w, height=h).unsqueeze(2))
        dump(out(f"ref{i}_rgba.bin"), vae_in[-1][0, :, 0])
        white = Image.new("RGB", resized.size, (255, 255, 255))
        white.paste(resized, mask=resized.getchannel("A"))
        np.asarray(white, dtype=np.float32).transpose(2, 0, 1).tofile(out(f"ref{i}_vision_rgb.bin"))
        print(f"reference {i}: {w}x{h}")
    return vision, vae_in


def hook_text_encoder(te, out):
    """Dump the vision tower and the text model around an image-conditioned
    encoding. Returns the hook handles."""
    visual, text = te.model.visual, te.model.language_model
    hooks = []

    def output_hook(name):
        def hook(_module, _inputs, output):
            dump(out(name), output[0] if isinstance(output, tuple) else output)
        return hook

    def visual_pre(_module, args, kwargs):
        dump(out("pixel_values.bin"), args[0])
        grid = kwargs.get("grid_thw", args[1] if len(args) > 1 else None)
        np.savetxt(out("grid_thw.txt"), grid.cpu().numpy(), fmt="%d")

    def block0_pre(_module, args, kwargs):
        dump(out("vis_in.bin"), args[0] if args else kwargs["hidden_states"])
        cos, sin = kwargs["position_embeddings"]
        dump(out("vis_rope_cos.bin"), cos)
        dump(out("vis_rope_sin.bin"), sin)

    def text_pre(_module, _args, kwargs):
        ids = kwargs["position_ids"].detach().cpu().numpy().astype(np.int32)
        ids.tofile(out("position_ids.bin"))
        print(f"position_ids shape: {ids.shape}")
        dump(out("txt_embeds.bin"), kwargs["inputs_embeds"])

    def layer3_pre(_module, args, kwargs):
        dump(out("txt_layer3_in.bin"), args[0] if args else kwargs["hidden_states"])

    hooks.append(visual.register_forward_pre_hook(visual_pre, with_kwargs=True))
    hooks.append(visual.patch_embed.register_forward_hook(output_hook("vis_patch.bin")))
    hooks.append(visual.blocks[0].register_forward_pre_hook(block0_pre, with_kwargs=True))
    for i in (0, 8, 16, 24, 26):
        hooks.append(visual.blocks[i].register_forward_hook(output_hook(f"vis_block{i}.bin")))
    for k, merger in enumerate(visual.deepstack_merger_list):
        hooks.append(merger.register_forward_hook(output_hook(f"deep{k}.bin")))
    hooks.append(visual.merger.register_forward_hook(output_hook("image_embeds.bin")))
    hooks.append(text.register_forward_pre_hook(text_pre, with_kwargs=True))
    for i in (0, 1, 2, 35):
        hooks.append(text.layers[i].register_forward_hook(output_hook(f"txt_layer{i}.bin")))
    hooks.append(text.layers[3].register_forward_pre_hook(layer3_pre, with_kwargs=True))
    return hooks


def encode(model_dir, prompt, device, out, vision=None):
    """Text embeddings [L, dim] and the slot mask [L], via the official
    pipeline's encode_prompt."""
    from diffusers import QwenImage21Pipeline

    pipe = QwenImage21Pipeline.from_pretrained(
        model_dir, transformer=None, vae=None, torch_dtype=torch.float32).to(device)
    ids = []

    def capture_ids(_module, _args, kwargs):
        ids.extend(kwargs["input_ids"][0].tolist())

    hooks = [pipe.text_encoder.register_forward_pre_hook(capture_ids, with_kwargs=True)]
    if vision:
        hooks += hook_text_encoder(pipe.text_encoder, out)
    with torch.no_grad():
        emb, mask, image_pad_mask = pipe.encode_prompt(prompt=prompt, image=vision, device=device)
    for hk in hooks:
        hk.remove()
    if mask is not None and not bool(mask.all()):
        raise SystemExit("unexpected padding in a single-prompt encoding")
    if bool(image_pad_mask.any()) != bool(vision):
        raise SystemExit("image tokens do not match the --image options")

    with open(out("tokens.txt"), "w") as f:
        f.write("\n".join(str(i) for i in ids) + "\n")
    with open(out("drop_idx.txt"), "w") as f:
        f.write(f"{pipe._drop_idx}\n")
    print(f"tokens: {len(ids)}, dropped: {pipe._drop_idx}, embeddings: {tuple(emb.shape[1:])}")

    emb = emb[0].to(torch.float32).cpu()
    pad = image_pad_mask[0].bool().cpu()
    np.savetxt(out("image_pad_mask.txt"), pad.numpy().astype(np.int32), fmt="%d")
    del pipe
    gc.collect()
    return emb, pad


def encode_refs(model_dir, vae_in, device, out):
    """Normalized VAE means, as _encode_vae_image computes them, packed
    into one [tokens, 64] tensor. Also returns the (1, h, w) shapes."""
    from diffusers import AutoencoderKLQwenImage21

    vae = AutoencoderKLQwenImage21.from_pretrained(
        model_dir, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device)

    def stage(name):
        def hook(_module, _inputs, output):
            dump(out(f"enc_{name}.bin"), output[0] if isinstance(output, tuple) else output)
        return hook

    enc = vae.encoder
    hooks = [enc.conv_in.register_forward_hook(stage("conv_in")),
             enc.mid_block.register_forward_hook(stage("mid")),
             enc.norm_out.register_forward_hook(stage("norm_out")),
             enc.conv_out.register_forward_hook(stage("conv_out")),
             vae.quant_conv.register_forward_hook(stage("quant"))]
    hooks += [b.register_forward_hook(stage(f"down{i}")) for i, b in enumerate(enc.down_blocks)]

    tokens, shapes = [], []
    with torch.no_grad():
        for i, x in enumerate(vae_in):
            z = vae.encode(x.to(device)).latent_dist.mode()
            for hk in hooks:
                hk.remove()
            hooks = []
            z = (z - mean) / std
            dump(out(f"ref{i}_latent.bin"), z[0, :, 0])
            _, c, _, h, w = z.shape
            tokens.append(z.view(1, c, h * w).transpose(1, 2).cpu())
            shapes.append((1, h, w))
    del vae
    gc.collect()
    return torch.cat(tokens, dim=1), shapes


def denoise(model_dir, emb, h, w, args, device, out, pad=None, ref_tokens=None, ref_shapes=()):
    """The QwenImage21Pipeline.__call__ denoising loop (no CFG, KV cache on).
    Returns the final latent tokens [N, 64]."""
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

    transformer = QwenImage21Transformer2DModel.from_pretrained(
        model_dir, subfolder="transformer", torch_dtype=torch.float32).to(device).eval()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")
    print("scheduler config:", dict(scheduler.config))
    channels = transformer.config.in_channels
    n = h * w

    if args.noise:
        noise = torch.from_numpy(np.fromfile(args.noise, dtype=np.float32)).view(1, channels, h, w)
    else:
        noise = torch.randn((1, 1, channels, h, w), generator=torch.Generator().manual_seed(args.seed),
                            dtype=torch.float32)[:, 0]
    dump(out("noise.bin"), noise[0])
    latents = noise.view(1, channels, n).transpose(1, 2).to(device)   # _pack_latents

    with open(os.path.join(model_dir, "model_index.json")) as f:
        sigmas = json.load(f).get("sample_sigmas")
    if sigmas is None:
        steps = args.steps or 40
        sigmas = np.linspace(1.0, 1 / steps, steps)
    elif args.steps:
        print(f"note: Turbo's sample_sigmas override --steps {args.steps}")
    mu = calculate_shift(n, scheduler.config.get("base_image_seq_len", 256),
                         scheduler.config.get("max_image_seq_len", 4096),
                         scheduler.config.get("base_shift", 0.5),
                         scheduler.config.get("max_shift", 1.15))
    scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    np.savetxt(out("sigmas.txt"), scheduler.sigmas.cpu().numpy(), fmt="%.8f")
    timesteps = scheduler.timesteps

    emb = emb.unsqueeze(0).to(device)
    txt_len = emb.shape[1]
    if pad is None:
        pad = torch.zeros(txt_len, dtype=torch.bool)
    img_mask = torch.cat([pad[None], torch.ones(1, n // 4, dtype=torch.bool)], dim=1).to(device)
    img_shapes = [[*ref_shapes, (1, h, w)]]
    if ref_shapes:
        from diffusers.models.transformers.transformer_qwenimage21 import (
            _qwenimage21_prefix_segments)
        expanded = torch.repeat_interleave(img_mask[0], torch.where(img_mask, 4, 1)[0])
        ids, target = transformer.build_token_metadata(expanded, img_shapes[0])
        with open(out("segments.txt"), "w") as f:
            for start, end, is_text in _qwenimage21_prefix_segments(ids, int((~target).sum())):
                f.write(f"{start} {end} {int(is_text)}\n")
    kv_cache = QwenImage21KVCache(len(transformer.transformer_blocks))

    step = {"i": 0}
    hooks = []

    def rope_hook(_module, _inputs, output):
        if step["i"] == 0:
            dump(out("rope_cos.bin"), output.real)
            dump(out("rope_sin.bin"), output.imag)
    hooks.append(transformer.pos_embed.register_forward_hook(rope_hook))
    if args.blocks:
        for idx, block in enumerate(transformer.transformer_blocks):
            def block_hook(_module, _inputs, output, idx=idx):
                if step["i"] < 2:
                    dump(out(f"block{idx}_s{step['i']}.bin"), output[0])
            hooks.append(block.register_forward_hook(block_hook))

    scheduler.set_begin_index(0)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            step["i"] = i
            model_input = latents if ref_tokens is None else torch.cat(
                [ref_tokens.to(device), latents], dim=1)
            v = transformer(hidden_states=model_input, timestep=t.expand(1).to(latents.dtype) / 1000,
                            encoder_hidden_states=emb, encoder_hidden_states_mask=None,
                            img_shapes=img_shapes, img_mask=img_mask, kv_cache=kv_cache,
                            kv_cache_mode="extract" if i == 0 else "cached",
                            return_dict=False)[0][:, -n:]
            latents = scheduler.step(v, t, latents, return_dict=False)[0]
            dump(out(f"step{i}_v.bin"), v[0])
            dump(out(f"step{i}_x.bin"), latents[0])
            print(f"step {i + 1}/{len(timesteps)}: sigma {float(scheduler.sigmas[i]):.6f}")

    for hk in hooks:
        hk.remove()
    del transformer, kv_cache
    gc.collect()
    return latents[0].cpu()


def decode(model_dir, latent, h, w, device, out):
    """latent [64, h, w] (normalized) -> image [4, 16h, 16w] in [-1, 1]."""
    from diffusers import AutoencoderKLQwenImage21
    from diffusers.image_processor import VaeImageProcessor

    vae = AutoencoderKLQwenImage21.from_pretrained(
        model_dir, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    z = latent.view(1, -1, 1, h, w).to(device)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device)
    with torch.no_grad():
        image = vae.decode(z * std + mean, return_dict=False)[0][:, :, 0]
    dump(out("image.bin"), image[0])
    pil = VaeImageProcessor(vae_scale_factor=16).postprocess(image, output_type="pil")[0]
    print(f"image mode: {pil.mode}")
    pil.save(out("image.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--model-dir", required=True)
    ap.add_argument("-p", "--prompt", default="")
    ap.add_argument("-W", "--width", type=int, default=256)
    ap.add_argument("-H", "--height", type=int, default=256)
    ap.add_argument("--steps", type=int, default=None,
                    help="base model steps (default 40; Turbo has a fixed grid)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", help="raw float32 noise [64, H/16, W/16]")
    ap.add_argument("--emb", help="raw float32 text embeddings [L, context_in_dim] (skips the encoder)")
    ap.add_argument("--decode", help="only decode this raw float32 latent [64, H/16, W/16]")
    ap.add_argument("--blocks", action="store_true", help="dump per-block hidden states")
    ap.add_argument("--image", action="append", default=[], help="reference image (repeatable)")
    ap.add_argument("--ref-resolution", type=int, default=1024,
                    help="the pipeline's output_resolution, sizes the references")
    ap.add_argument("--text-only", action="store_true", help="stop after the text encoder")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("-o", "--out-dir", default="/tmp/qimage_ref")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    def out(name):
        return os.path.join(args.out_dir, name)

    # The pipeline floors sizes to multiples of 32 (one token is 16x16 px,
    # and the target occupies 2x2 token slots of the encoder sequence).
    width, height = args.width // 32 * 32, args.height // 32 * 32
    h, w = height // 16, width // 16

    if args.decode:
        latent = torch.from_numpy(np.fromfile(args.decode, dtype=np.float32))
        decode(args.model_dir, latent, h, w, args.device, out)
        return

    vision, vae_in, pad = None, [], None
    if args.image:
        vision, vae_in = prepare_refs(args.image, args.ref_resolution, out)
    if args.emb:
        with open(os.path.join(args.model_dir, "transformer", "config.json")) as f:
            dim = json.load(f).get("context_in_dim", 4096)
        emb = torch.from_numpy(np.fromfile(args.emb, dtype=np.float32)).view(-1, dim)
        if args.image:
            raise SystemExit("--emb cannot be combined with --image")
    else:
        emb, pad = encode(args.model_dir, args.prompt, args.device, out, vision)
    dump(out("emb.bin"), emb)
    if args.text_only:
        return

    ref_tokens, ref_shapes = encode_refs(args.model_dir, vae_in, args.device, out) if vae_in else (None, [])
    tokens = denoise(args.model_dir, emb, h, w, args, args.device, out, pad, ref_tokens, ref_shapes)
    latent = tokens.transpose(0, 1).reshape(-1, h, w)   # _unpack_latents
    dump(out("latent.bin"), latent)
    decode(args.model_dir, latent, h, w, args.device, out)


if __name__ == "__main__":
    main()
