This is a C implementation of four image synthesis model families:
- Flux.2 Klein (4B and 9B variants)
- Z-Image-Turbo (6B)
- FLUX.3 Image Klein (5.6B)
- Qwen-Image 2.1 (about 7B transformer, base and Turbo)

The project is called "Iris" (from the Greek goddess of the rainbow).
The Flux models are created by Black Forest Labs.
Z-Image-Turbo is published as `Tongyi-MAI/Z-Image-Turbo`.
Qwen-Image 2.1 is published as `Qwen/Qwen-Image-2.1` and `Qwen/Qwen-Image-2.1-Turbo`.

Model type and architecture are autodetected from model metadata/config files.
Do not rely on hardcoded dimensions when a config value is available.

# Naming Convention

- **`iris_`** prefix for all shared/generic identifiers
- **`_flux`** postfix on internal functions/types specific to the Flux model family
- **`_zimage`** postfix on internal functions/types specific to Z-Image
- **`_flux3`** postfix on internal functions/types specific to FLUX.3
- **`_qimage`** postfix on internal functions/types specific to Qwen-Image 2.1
- **No postfix** on public API functions (they route internally by model type)
- **Unchanged**: `qwen3_*`, `safetensors_*`, `zi_*` internal helpers (component-level namespaces)

# Supported Model Variants

Flux variants:
- **4B Distilled** (`flux-klein-4b`): 4 steps, no CFG, fast.
- **4B Base** (`flux-klein-4b-base`): 50 steps default, CFG, higher quality but much slower.
- **9B Distilled** (`flux-klein-9b`): 4 steps, larger model, higher quality. Non-commercial license.
- **9B Base** (`flux-klein-9b-base`): 50 steps default, CFG. Non-commercial license.

Z-Image variant:
- **Z-Image-Turbo** (`zimage-turbo`): distilled S3-DiT model, default 8 NFE (9 scheduler values), guidance 0.0.

FLUX.3 variants (FLUX Kommunity License):
- **Image Klein** (`flux-3-klein`): step-distilled flow map, 16 steps default (trained for 4..16), no CFG.
- **Image Klein Base** (`flux-3-klein-base`): 50 steps, CFG 3.5.

Qwen-Image 2.1 variants (Qwen Research License, non-commercial):
- **Turbo** (`qwen-image-2.1-turbo`): fixed 8-sigma grid from `model_index.json`
  (`sample_sigmas`), no CFG. `--steps` cannot change it.
- **Base** (`qwen-image-2.1`): 40 steps default, no CFG (upstream default too).

# High-Level Capabilities

- Flux supports both txt2img and img2img with text conditioning.
- Z-Image currently supports txt2img path in this codebase.
- FLUX.3 supports txt2img and reference images (editing, multi-reference). References
  are prepared like the official image processor: one reference is kept whole
  (downscaled so its 16-px canvas fits 4 MP, reflect-padded, output cropped back),
  several are downscaled under 1 MP and center-cropped. Side cap: IRIS_VAE_MAX_DIM.
- Qwen-Image supports txt2img and reference images (`-i`, up to 16). A reference is
  sized to about 1024x1024 and takes about 1024 text encoder tokens and 4096
  prefix tokens. QWEN3_QIMAGE_MAX_SEQ_LEN (17280) limits the encoder sequence.
  A run over it, or over the memory estimate, fails before any expensive stage
  (see Qwen-Image Long Sequences). A text-only prompt over it is cut with a warning.
- Text embeddings are generated via Qwen3.
- VAE is used for latent/image conversion in both families.

# File Structure

```
iris.c                    - Main library (model load, generation routing)
iris_transformer_flux.c   - Flux diffusion transformer (MMDiT)
iris_transformer_zimage.c - Z-Image transformer (S3-DiT)
iris_transformer_flux3.c  - FLUX.3 transformer (per-mode + joint blocks)
iris_transformer_qimage.c - Qwen-Image 2.1 transformer (single-stream, text prefix KV)
iris_vae_qimage.c         - Qwen-Image 2.1 VAE (Wan-derived, 2D)
iris_qimage.h             - Qwen-Image internal API (scheduler, transformer, VAE)
iris_sample.c             - Sampling/denoising loops (Euler ODE)
iris_qwen3.c              - Qwen3 text encoder
iris_qwen3_vision.c       - Qwen3-VL vision tower (Qwen-Image references)
iris_qwen3_tokenizer.c    - BPE tokenizer (added tokens, NFC, Qwen2 pre-tokenizer)
qwen3_rope_test.c         - Text encoder positions against a hand-written layout
iris_nfc_table.h          - Unicode tables for NFC, from debug/gen_nfc_table.py
iris_vae.c                - VAE encoder/decoder
iris_kernels.c            - CPU kernels (softmax, RMSNorm, etc.)
iris_metal.m              - Metal GPU acceleration runtime
iris_metal.h              - Metal/GPU API surface
iris_dev.h                - Device (CUDA / ROCm / Metal) compute API
iris_cuda.cu              - CUDA / ROCm (HIP) implementation of iris_dev.h
iris_dev_metal.m          - Metal implementation of iris_dev.h
iris_shaders.metal        - Metal compute kernels
iris_safetensors.c        - Weight loading
iris_image.c              - Image I/O (PNG/PPM/JPEG)
png.c                     - PNG encoder/decoder
jpeg.c                    - JPEG decoder
iris_cli.c                - Interactive CLI mode (REPL)
embcache.c                - Embedding cache (4-bit quantized)
linenoise.c               - Line editing library
terminals.c               - Terminal handling
main.c                    - CLI entry point
```

# Build Targets

This project implements five targets:
- MPS: Apple Silicon GPU path. The transformers use iris_dev.h
  (iris_dev_metal.m), like Qwen-Image's text encoder and VAE; the other text
  encoders and VAEs use iris_metal.m.
- CUDA / ROCm: NVIDIA / AMD GPU path (iris_dev.h, implemented by iris_cuda.cu,
  one source for nvcc and hipcc). All the transformers, the Qwen3 text
  encoder and the VAE.
- BLAS: optimized CPU inference via BLAS/OpenBLAS.
- generic: pure C fallback, very slow.

# Development Rules

- No additional project dependencies. Acceptable external deps are BLAS/OpenBLAS, Metal/MPS from macOS,
  and for the GPU builds the CUDA toolkit (runtime, cuBLAS) or ROCm (HIP runtime, hipBLAS).
- Reject tiny speed gains that add complexity; prefer substantial wins.
- Always test code modifications with `make test`.
- Once changes are validated, commit them.
- Never add or commit unrelated unstaged files.
- Keep code simple and understandable; leave no dead code.
- If you optimize one backend, verify others were not regressed.
- Stick to standard C; avoid compiler-specific tricks/pragmas unless strictly required.

# How To Run

Flux examples:

    ./iris -d flux-klein-4b -p "a cat and a dog playing" -o /tmp/test.png
    ./iris -d flux-klein-4b-base -p "a cat and a dog playing" -o /tmp/test.png
    ./iris -d flux-klein-9b -p "a cat and a dog playing" -o /tmp/test.png
    ./iris -d flux-klein-9b-base -p "a cat and a dog playing" -o /tmp/test.png

Z-Image example:

    ./iris -d zimage-turbo -p "a fish" -o /tmp/zimage.png

FLUX.3 example:

    ./iris -d flux-3-klein -p "a fish" -o /tmp/flux3.png

Qwen-Image example:

    ./iris -d qwen-image-2.1-turbo -p "a fish" -W 1024 -H 1024 -o /tmp/qimage.png
    ./iris -d qwen-image-2.1-turbo -p "make it a watercolor" -i photo.png -o /tmp/qedit.png

If model weights are missing, use the download script only after user approval.

# Python Reference Implementations

For parity checks/debugging:

Flux references:
- Python venv in `./flux_env/`
- Official Flux Python code in `./flux2/`

Z-Image references:
- `flux_env/lib/python3.12/site-packages/diffusers/models/transformers/transformer_z_image.py`
- `flux_env/lib/python3.12/site-packages/diffusers/pipelines/z_image/pipeline_z_image.py`
- `flux_env/lib/python3.12/site-packages/diffusers/schedulers/scheduling_flow_match_euler_discrete.py`
- `flux_env/lib/python3.12/site-packages/diffusers/models/autoencoders/autoencoder_kl.py`

FLUX.3 reference:
- `debug/flux3_reference.py`: official pipeline (text encoding, Euler sampling,
  VAE decode) on the official Flux3Transformer2DModel source, fetched from a
  pinned diffusers commit (FLUX.3 is not in a diffusers release yet). Dumps
  every intermediate tensor as raw float32; `--noise` starts from a C noise file.

Qwen-Image reference:
- `debug/qimage_reference.py`: the QwenImage21Pipeline text-to-image loop on
  the official diffusers classes (encode_prompt, KV cache, scheduler, VAE) in
  float32. Dumps tokens, embeddings, sigmas, RoPE, per-step v/x, per-block
  states, latent and image. `--emb` skips the text encoder, `--decode` only
  decodes a latent. C side: `./iris -d MODEL -e emb.bin --noise noise.bin`.
- `debug/dump_noise.c`: writes the noise iris draws for a seed, for `--noise`.
- `debug/dump_emb.c`: writes the text embeddings iris computes for a prompt.
- With `--image` the reference also dumps the reference images, VAE stages,
  vision tower and text layers. `debug/dump_qimage_refs.c` compares the
  transformer on them, `debug/dump_qimage_text.c` the text encoder (tokens and
  positions exactly), `debug/dump_qimage_e2e.c` the whole pipeline from real
  images with the dumped noise, at any `--ref-resolution`.

Rules:
- Never add/commit `flux_env/` or `flux2/`.
- If missing, ask user before recreating/downloading.

# Debugging Notes

- Reusable debug scripts belong in `./debug`.
- One-off throwaway debugging tools should be created in `/tmp` and discarded.
- JPEG code has dedicated tests/tools in `./jpg_test`.

# Flux Pipeline Overview

```
1. Text Encoding:    prompt -> Qwen3 -> [512, text_dim] embeddings
2. Latent Init:      random noise [H/16, W/16, 128]
3. Denoising Loop:   double blocks -> single blocks -> final layer -> velocity
4. VAE Decode:       latents -> VAE decoder -> RGB image

img2img: VAE-encode reference images and pass as extra tokens (in-context conditioning).

Base CFG: run transformer twice per step (empty prompt + conditioned prompt):
v = v_uncond + guidance * (v_cond - v_uncond)
```

# Flux Key Architecture Constants

Read from runtime config; reference values:

Flux 4B transformer:
- hidden=3072
- heads=24
- head_dim=128
- mlp=9216
- double_blocks=5
- single_blocks=20
- text_dim=7680

Flux 9B transformer:
- hidden=4096
- heads=32
- head_dim=128
- mlp=12288
- double_blocks=8
- single_blocks=24
- text_dim=12288

Qwen3 (4B/9B encoders used for Flux):
- 4B: hidden=2560, q_heads=32, kv_heads=8, layers=36, output_dim=7680
- 9B: hidden=4096, q_heads=32, kv_heads=8, layers=36, output_dim=12288

Shared:
- latent_ch=128 (Flux latent after patchification)
- head_dim=128

# Flux Critical Implementation Details

- Concatenation order for attention is `[TEXT, IMAGE]`, not `[IMAGE, TEXT]`.
- AdaLN formula is `out = (1 + scale) * norm(x) + shift`.
- Final layer modulation split is `(scale, shift)`, not `(shift, scale)`.
- RoPE pair rotation is:
  - `out0 = cos * x0 - sin * x1`
  - `out1 = cos * x1 + sin * x0`

# Flux RoPE (Rotary Position Embedding)

4-axis RoPE with 32 dims per axis (128 total):
- Axis 0 (dims 0..31): T position
- Axis 1 (dims 32..63): H position
- Axis 2 (dims 64..95): W position
- Axis 3 (dims 96..127): L sequence index

Token usage:
- Image tokens: use H/W axes, T/L identity.
- Text tokens: use L axis only.
- Reference image tokens (img2img): same as image plus T offset (10, 20, 30...).

# Flux Timestep Embedding

1. Scale timestep by 1000.
2. Sinusoidal embedding (128 freqs -> 256 dims).
3. MLP: linear(256->hidden) + SiLU + linear(hidden->hidden).

# Flux Text Encoder (Qwen3)

Chat template:
```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Flux extraction: concatenate layers 8, 17, 26 (0-indexed):
- 4B: [seq, 7680]
- 9B: [seq, 12288]

# Flux VAE

- Latent space: 32 channels, 16x spatial compression.
- Patchify (encode): [B, 32, H/8, W/8] -> [B, 128, H/16, W/16]
- Unpatchify (decode): [B, 128, H/16, W/16] -> [B, 32, H/8, W/8]
- Channel multipliers: [1,2,4,4] -> [128,256,512,512]

# Flux Double Block Flow

Input: `img_hidden [img_seq, hidden]`, `txt_hidden [txt_seq, hidden]`

1. AdaLN normalize both streams (shift1, scale1)
2. Stream-specific Q/K/V projections
3. QK per-head RMSNorm
4. RoPE (image axes 1/2, text axis 3)
5. Joint attention over concatenated KV
6. Output projection + gate1
7. Residual add
8. AdaLN normalize (shift2, scale2)
9. SiLU-gated MLP
10. gate2 + residual add

Per-stream modulation params: shift1, scale1, gate1, shift2, scale2, gate2.

# Flux Single Block Flow

Input: concatenated `[txt_hidden, img_hidden]`

1. AdaLN normalize (shift/scale from t_emb)
2. Fused QKV+MLP projection -> [Q,K,V,gate,up]
3. QK normalization
4. RoPE (text uses axis 3, image uses axes 1/2)
5. Full self-attention
6. SwiGLU: `silu(gate) * up`
7. Concat attention output + MLP output
8. Output projection
9. Gated residual add

# Z-Image Pipeline Overview

```
1. Text Encoding:    prompt -> Qwen3 hidden_states[-2] -> [seq, 2560]
2. Latent Init:      random noise [16, H/8, W/8] then patch-space processing
3. Denoising Loop:   noise_refiner(2) -> context_refiner(2) -> main layers(30)
4. Final Layer:      norm + modulation + projection to patch channels
5. Unpatchify:       [num_patches, 64] -> [16, H/8, W/8]
6. VAE Decode:       latents -> image
```

Z-Image uses sequence order `[IMAGE_TOKENS | CAPTION_TOKENS]` in its transformer path.

# Z-Image Architecture (S3-DiT)

Overview:
- Model: Z-Image-Turbo (6B)
- Architecture: S3-DiT single-stream style with refiners
- Steps: default 8 NFE (scheduler array has 9 sigma values)
- guidance: 0.0 (no CFG)

Transformer config (reference):
- dim=3840
- n_heads=30
- n_kv_heads=30
- head_dim=128
- n_layers=30
- n_refiner_layers=2 (noise + context)
- in_channels=16
- patch_size=2
- cap_feat_dim=2560
- axes_dims=[32,48,48]
- rope_theta=256.0
- ffn_hidden=10240

# Z-Image Text Encoder

Uses the same Qwen3 4B base encoder family, but extraction differs from Flux:
- extract only `hidden_states[-2]` (layer 34/36)
- output shape [seq, 2560]

Chat template is the same as Flux Qwen3 template.

# Z-Image VAE Notes

Z-Image VAE is Flux-like architecture with different latent settings:
- latent_channels=16 (Flux uses 32)
- scaling_factor=0.3611
- shift_factor=0.1159
- patchification for transformer input uses 2x2 over 16ch -> 64-dim patch vector

Decode uses:
- `latents = (latents / scaling_factor) + shift_factor`, then decode.

# Z-Image Block Structure

With modulation (noise_refiner + main layers):
- modulation vector split into: scale_msa, gate_msa, scale_mlp, gate_mlp
- gates go through tanh
- normalization is scaled (no additive shift in block modulation path)
- attention and FFN residuals are gated

Without modulation (context_refiner):
- no adaLN modulation
- standard residual attention + FFN structure

# Z-Image RoPE

3-axis RoPE, head_dim=128 split as:
- Axis T: 32 dims
- Axis H: 48 dims
- Axis W: 48 dims

Theta is 256.0.
Caption and image position-id construction must match CPU/GPU path exactly.

# Z-Image Scheduler

Default scheduler follows official diffusers FlowMatch Euler behavior (static shift).
Important points:
- terminal sigma is appended as 0
- implementation should match official sigma construction semantics
- explicit `--linear` or `--power` options override default schedule for parity/testing

# Z-Image Critical Implementation Details

- Transformer sequence order is `[IMAGE | CAPTION]` in Z-Image path.
- Step previews must decode correctly patchified latent representation (`patch_size=2`),
  otherwise shown step resolution/content can be misleading.
- Caption padding and position-id alignment must match between CPU and MPS paths.
- Keep immutable/static and dynamic matrix paths separate in MPS GEMM caching.

# Z-Image Runtime Lessons (Keep)

- Keep transformer loaded across generations in interactive workflows; avoid load/free per image.
- Precompute per-step modulation once and reuse across modulated blocks.
- Keep GPU fallback policy inside the zImage GPU forward path; avoid immediate full CPU fallback.
- Run zImage final layer on GPU path where possible; read back only what is needed for decode.
- Run image/caption embedding projections on GPU path in MPS mode.
- Warm bf16 weight caches at load time to reduce first-step jitter in steady-state usage.
- Keep embedding-cache path wired for repeated prompts in distilled interactive mode.
- Maintain zImage-specific denoising timing counters for targeted profiling.

# Z-Image Known Pitfalls (Historical Bugs)

1. **MPS SGEMM B-cache misuse caused VAE decode corruption (hue/border artifacts).**
   - Root cause: generic SGEMM cached matrix B by pointer, but VAE attention K/V are dynamic temporaries.
   - Fix: split API paths:
     - `iris_metal_sgemm`: generic path, no B-pointer cache
     - `iris_metal_sgemm_cached`: explicit static-weight cached path
   - Static weight call sites (for example linear layers) use cached variant only.

2. **Scheduler parity bug with official Python implementation.**
   - Z-Image sigma schedule must follow official diffusers FlowMatch behavior.
   - Incorrect sigma handling can waste steps and make show-steps output confusing.

3. **Step-preview mismatch bug.**
   - Z-Image preview decode must account for patch-size transformation, otherwise preview steps do not match final decode behavior.

4. **CPU/GPU position-id mismatch under padded caption sequences.**
   - Using non-padded cap length in one path and padded length in another changes RoPE indexing and output quality.

5. **RMSNorm temporary weight caching pitfalls.**
   - Temporary fused norm weights must not be pointer-cached as immutable weights.

# FLUX.3 Architecture (Image Klein)

Transformer config (reference): dim=3072, heads=24, head_dim=128, mlp_ratio=3,
num_layers=5 (per mode), num_single_layers=28, joint_attention_dim=20480,
axes_dims_rope=[32,32,32,32], rope_theta=10000, flow_map=true (distilled).

- Token modes: text, image, image_cond (references). Each mode first runs its
  own 5 blocks (self-attention within the mode), then the 28 joint blocks run
  over `[TEXT, IMAGE, REFS]`. Only image tokens reach the output layer.
- All blocks are identical: LayerNorm (no affine) -> `(1 + scale) * n + shift`
  -> attention and SwiGLU FFN in parallel on the same input ->
  `x += gate * (attn + ffn)`. Separate to_q/to_k/to_v, QK RMSNorm with weights.
- FFN linear_in output is `[gate | up]` halves: `silu(gate) * up`.
- Modulation is per mode, not per block: one projection per mode for all
  per-mode blocks and one for all joint blocks, split as (shift, scale, gate).
  The output layer splits as (scale, shift). All projections have no bias.
- Image tokens use timestep t, text and references timestep 0. Flow-map models
  add the embedding of the target timestep (next sigma) to every mode.
- RoPE: consecutive-pair rotation. Image (0, row/2, col/2, 0), text
  (0, 0, 0, index), reference i (0, row/2, col/2, -(i+1)).
- Scheduler: sigmas = linspace(1, 1/n, n), static shift 6.93, terminal 0.
  Plain Euler: `x += (sigma_next - sigma) * v`.

# FLUX.3 Text Encoder

- Qwen3-VL-4B text model (tensors under `model.language_model.`), same shape
  as Qwen3-4B, rope_theta=5e6. With text-only input its MRoPE is plain RoPE.
- Chat template has NO `<think>` block.
- Concatenate hidden_states[4, 8, ..., 32] -> [seq, 20480].
- Pad (with `<|endoftext|>`) to the next multiple of 80 tokens. Padding is
  masked inside Qwen3-VL but NOT in the transformer, and pad tokens sit at
  RoPE position 1 (transformers 4.57 behavior, used for training).

# FLUX.3 Known Pitfalls (Historical Bugs)

1. **Tokenizer merges format**: the Qwen3-VL tokenizer.json stores merges as
   `"a b"` strings, not `["a", "b"]` pairs; unparsed merges silently give
   character-level tokens.
   The pre-tokenizer must follow the Qwen2 split regex exactly (all models):
   a simplified split gave different tokens for hyphenated words, quotes,
   punctuation before newlines, repeated spaces, and changed the images.
   Before it, added tokens split the text (even inside a prompt) and the pieces
   go through the tokenizer's NFC normalizer.
2. **GPU attention threadgroup size**: the causal attention kernels use tree
   reductions that need a power-of-two thread count. Flux/Z-Image always used
   512 tokens; Flux3 windows (80, 160, ...) exposed the bug.
3. **RoPE table buffer caching**: the text encoder refills its RoPE tables per
   prompt, so they must not go through the pointer-keyed rope buffer cache.
4. **Metal weight cache keys**: on MPS builds, Flux3 linears pass the mmapped
   bf16 weight pointers; a reused f32 conversion scratch would be served stale.

# Qwen-Image 2.1 Architecture

Transformer config (reference): dim=4096, heads=32, head_dim=128, 32 blocks,
mlp_ratio=3, in/out channels 64, patch_size 1, axes_dims_rope=[16,56,56],
theta 10000 (hardcoded upstream), eps 1e-6, causal_condition=true. No biases.

- Sequence `[TEXT | IMAGE]`; image tokens are latent pixels in raster order.
- Block (sequential): `x += tanh(g1) * attn(LN(x) * (1 + s1))`, then
  `x += tanh(g2) * mlp(LN(x) * (1 + s2))`; no shift. MLP is
  `out(silu(gate_layer(n)) * proj(n))`.
- One modulation projection for all blocks: `[s1, g1, s2, g2]`. Text uses
  t = 0, image tokens t. Output layer: `LN(x) * (1 + scale)` (scale only).
- Attention: text causal, image tokens see all text and image tokens.
- Text input: RMSNorm with weight + 1 (zero-centered), linear, GELU-tanh, linear.
- RoPE consecutive pairs; text token i at (i, i, i); image (T, r - (H - H/2),
  c - (W - W/2)), T the text length. Angles are float32 like the reference.
- Text is independent of the image and the step, so iris prefills its K/V
  once per prompt (the reference KV cache, filled before step 0) and each
  step runs only image tokens against them.
- Scheduler: FlowMatchEulerDiscreteScheduler settings read from
  `scheduler/scheduler_config.json`, applied to Turbo's sample_sigmas too.
- VAE: 2D convs, per-pixel RMS norms (F.normalize * sqrt(C) * gamma), residual
  up blocks with a DupUp shortcut, mid single-head attention, RGBA out. Latent
  denorm `z * std + mean` (64 values each from config). The encoder mirrors it
  (down conv pads right and bottom by 1, AvgDown shortcut), is loaded at the
  first encode and freed by iris_vae_release_qimage(); the latent is the
  normalized posterior mean.
- Reference preparation (iris_image.c): RGBA, 32-px grid with Python's round,
  premultiplied Lanczos, vision copy over white. The resize is PIL's 8-bit one
  (fixed-point weights, rounded and clipped after each pass), not
  iris_image_resize, whose float passes FLUX.3 relies on: the encoder turns the
  difference of up to 6/255 RMS into 1-7% of latent error.

# Qwen-Image Text Encoder

- Qwen3-VL-8B text model (`model.language_model.`), same shape as Qwen3-8B.
- Raw template with a system message, no `<think>`; the system block tokens
  are dropped. Embedding = last layer output BEFORE the final norm.
- Only real tokens reach the transformer; the encoder pads to a multiple of
  80 after them (causal, so harmless).
- Work buffers follow the real sequence (qwen3_ensure_work). attn_scores holds
  one head, seq^2 floats: the CPU path runs heads one at a time.
- Tokenizer may live in `processor/`; Turbo's encoder may be one
  `model.safetensors`.

With references (qwen3_encode_images):
- The user turn is `<image1><|vision_start|>pad * n<|vision_end|>`, then
  ` <image2>...`, then the prompt. `<imageN>` is literal text, every piece
  between special tokens is tokenized alone, n = (H/32) * (W/32).
- The main vision embeddings replace the pad rows after the embedding lookup,
  the deepstack features are added to them after layers 0, 1 and 2. Text
  before the first image is unaffected.
- MRoPE, all in qwen3_set_rope_positions: text (p, p, p); image
  (p, p + row, p + col) in merged blocks, then p += max(h, w). With
  mrope_section [24, 20, 20], frequencies i < 60 follow height when i%3 == 1,
  width when i%3 == 2, the others the frame.
- Slot rows are not returned: the transformer fills them with latents, and
  iris_qimage_ref_t.pos counts the text rows before each block.
- Reference pipeline (iris_generate_refs_qimage): prepare, VAE encode, vision
  tower, text, each freed before the next. A transformer left by the previous
  generation is freed first: beside all three the REPL peaks at 41 GB instead of
  29 GB, and reloading it costs under 2 s. Text-only generations keep it. Without
  -W -H the target follows the
  LAST reference at 1024x1024 area (FLUX.3 follows the first), and -W -H never
  change the reference size.

# Qwen-Image Long Sequences

- A reference takes about 1024 encoder tokens: two need 2100, ten 10300.
- Metal encoder: with mapped weights Qwen-Image runs qwen3_forward_dev, the path
  CUDA and ROCm use for everything (QWEN3_USE_DEV, keyed on the extraction mode).
  The other models stay on the native iris_metal.m path, which ends at
  CAUSAL_ATTENTION_MAX_SEQ (1280, threadgroup scores, Metal allows 32 KB) and
  which FLUX.3's byte-exact hashes rest on. Qwen-Image falls back to it up to 1280
  tokens when the device path cannot allocate or is not initialized; --no-mmap
  and library callers without iris_set_mmap() also stay on it, with different
  output. Time and memory are equal, and
  the device path is closer: text rows 1.1% against 7.1% off float32 at 17 tokens,
  1.3% against 6.7% at 1060, upstream bf16 7.0% and 6.8%. See pitfall 3 below for
  why the final latent does not follow.
- iris_dev_attention on Metal: GQA by broadcasting the query heads of a kv head
  against it, and queries split in chunks of ATTN_MAX_SCORES scores, raising
  causal (query i sees keys j < i + n). That is what runs a long causal segment
  of the prefill too. The causal bias is computed in the graph, a constant
  would stay with every cached graph. CUDA needs none of this: its GEMM path
  already works in query chunks of ATTN_SCRATCH.
- ATTN_MAX_SCORES (2^31) sits above every byte-exact case: at 2^29 the 1-MP
  reference runs split, and MPSGraph rounds a split run differently.
- iris_fit_qimage() refuses a run before the first stage: over the token limit
  (without the device path on Metal, mapped weights: 1280), or when the largest
  stage does not fit. Stages: iris_transformer_memory_qimage() (prefix,
  activations, weights on a GPU, what MPSGraph holds on Metal, the scores of
  causal text in the GEMM attention and its input on CUDA and ROCm),
  qwen3_memory() (layer weights: device bf16; clean mapped pages on the CPU;
  f32 with --no-mmap, twice that on Metal, whose cache copies them) and
  iris_vae_memory_qimage(). Available: iris_dev_memory(held = 1) (Metal: the
  working set budget, capped by free and file-backed pages and what the process
  holds, since a new run replaces it; CUDA: free VRAM plus the scratch buffers a
  run reuses) or 90% of the RAM on the CPU. A run that is short unloads what the
  last one left (iris_unload_qimage(): transformer, VAE activations and device
  weights), so a repeat is judged like a first run. Every reference run is
  checked; a text-only run only past 1280 tokens, so short prompts need no
  tokenizer and behave as before. The figures were measured on all three, ROCm
  excepted.
- Prefix: released after sampling when it has references or is past 1280 tokens
  (iris_transformer_release_prefix_qimage()), so the decode and the next prompt
  do not sit beside gigabytes nothing will read. Short text-only prefixes stay
  for a repeated prompt.
- Decode: dv_buffers() asks iris_dev_memory() first, since Metal allocates past
  its budget. A decode that does not fit unloads the transformer.
- Text-only: an encoder that does not fit beside the previous transformer
  unloads the same (in iris_encode_text(), so the REPL does too).
- Metal with mapped weights cannot run the CPU encoder path, so past 1280 tokens
  without the device path the encoder fails with an error.
- A free run with two or more references can end far from float32 although
  each step does not: r9 (two references, a vague prompt) has velocities within
  1.5-2.3% and the latent 28% off. A 1e-6 noise change moves it 2e-5, and the
  float32 pipeline fed upstream's bf16 embeddings ends 21% off. A prompt that
  pins the scene ends 3% off.

# Qwen3-VL Vision Tower

`qwen3_vision_encode()`, one image at a time (config in `text_encoder/config.json`):
- Patches are block-major (block row, block column, row and column in block),
  so merged rows are raster and each 2x2 block is contiguous.
- Patch embedding: the Conv3d's temporal slices summed into one linear.
  Position embedding: bilinear on the 48x48 table, align_corners.
- 27 pre-norm blocks: `qkv` with bias, split-half RoPE (pairs (i, i + 36)) on Q
  and K, no head norm, GELU-tanh MLP. Mergers use exact GELU; deepstack mergers
  (after blocks 8, 16, 24) norm after flattening, the main one before.
- The stream grows to rms 162, max 14000 in the last blocks, which amplifies
  rounding: at 512x512 / 1024x1024, upstream f32 vs f64 is 2.4e-5 / 7.5e-5 on the
  output, upstream bf16 vs f32 5% / 16%, iris on Metal 3% / 6% (input rounding,
  not attention: f32 attention on the host still gives 5.4%).

# Qwen-Image Known Pitfalls

1. **Turbo sigmas go through the scheduler too**: the pipeline passes
   sample_sigmas through the same set_timesteps as the base schedule. Turbo's
   config turns dynamic shift and terminal stretch off, so they come out
   unchanged; do not special-case Turbo, read the scheduler config.
2. **DupUp shortcut**: `out[o, 2y+i, 2x+j] = in[(((o*ft + ft-1)*2+i)*2+j) /
   repeats, y, x]` (first_chunk keeps temporal slot ft-1). It is not a plain
   nearest upsample for the channel-reducing blocks.
3. **Device path rounding**: bf16 activations give about 1% relative error vs
   the float32 reference, the same as running upstream in bf16. Some reference
   runs amplify it: a 1e-6 change of the noise moves the two-reference test
   (r2) by 1e-5 against 1.7e-6 with one reference, so that run ends 3.8% off
   on Metal and 4.4% off in upstream bf16, and 1e-4 off on the CPU in f32.
   Long sequences do the same: 2100 and 3100 tokens end 2.5e-4 off on the CPU.
   The ending point of such a run is one draw of the rounding: on r12, 1e-5 of
   noise gives 2.3-4.2% on Metal and 1.9-9.2% on CUDA, and float32 inputs 2.9-7.2%
   and 3.9-9.1%, with the same velocities (1-3%). So r2 ends 7.1% and r12 5.9% off
   on CUDA (3.9% and 2.8% on Metal), the two-reference test measures 3.42 (1.86)
   and its threshold is 6. One reference ends where Metal does (0.6-1.3%).
   The Metal transformer sets the floor of a text-only run, not the encoder.
   On the f32 CPU transformer the native encoder's text error moves the latent
   0.6-2.4%, the device encoder's 0.1-0.9%. On Metal, float32 embeddings end
   0.8-4.4% off, about where the device encoder ends. Native against device
   then scatters both ways: over 13 text-only runs a run moved by 0.47x to 2.4x,
   the geometric mean by 1.01x. Upstream bf16 is 4.5% on the same 256x256 run.
4. **Causal device attention**: the prefill and the Qwen-Image encoder are the
   causal callers of iris_dev_attention in MPS builds. Metal used to return without writing, so
   text keys lost their attention output and velocities were 11-32% off.
5. **VAE memory**: a 2048x2048 decode needs about 22 GB of GPU activations. When
   they do not fit next to the transformer, it is unloaded first (the next
   generation reloads it). The CPU decoder is the last resort: without BLAS it
   takes minutes at 256x256.

# Device Backend (CUDA / ROCm / Metal)

- iris_dev.h is a small API for GPU-resident inference: plain device pointers,
  bf16 stored as uint16_t, one stream (iris_dev_download()/sync() wait).
  iris_cuda.cu maps the few CUDA/HIP runtime and cuBLAS/hipBLAS names that
  differ, so the same source builds with nvcc or hipcc (-DIRIS_HIP).
- Models own their device weights (uploaded explicitly at load / first use).
  Never key device data on host pointers (see the Metal pitfalls above).
- GEMMs: bf16 inputs, f32 accumulation. Fused weights are assembled on the
  device with iris_dev_upload()/iris_dev_upload_2d().
- iris_dev_copy_2d(): stream-ordered device-to-device bf16 copy (Qwen-Image
  copies each block's cached text K|V into the shared QKV buffer with it).
- iris_dev_rope(): split-half rotation without norm, any head_dim (the vision
  tower: head_dim 72, attention takes the GEMM path on CUDA).
- Attention: flash kernels for head_dim 128, non-causal, optional key mask
  (CUDA: mma.sync + cp.async + ldmatrix; ROCm: gfx11 WMMA, computing transposed
  products so each lane owns a query). Other cases use batched GEMMs around a
  softmax.
- Uploads: CUDA goes through pinned staging buffers.
- Metal (iris_dev_metal.m): device pointers are the CPU addresses of shared
  MTLBuffers, mapped back to (buffer, offset) by a sorted allocation table.
  Work goes into one command buffer, committed by sync/download/upload/free.
  GEMMs and attention use MPSGraph (bf16 matmul, fused SDPA on contiguous
  copies of q/k/v, key mask and causal triangle as an additive bias). In MPS
  builds the transformers, the Qwen-Image VAE and the Qwen-Image encoder use
  it: the Flux VAE device path is compiled out (VAE_DEV).

Known pitfalls:
1. rocBLAS has only fallback (slow) bf16 -> f32 GEMM kernels on gfx1151:
   iris_dev_linear_f32() computes bf16 and widens it on HIP.
2. On a GB10, pageable copies from mmapped files run at 0.5-3 GB/s, and GPU
   reads of pageable memory (ATS) are slow for one-pass copies: use the pinned
   staging path. GEMMs can read host memory fine, but uploads must not rely on it.
3. RDNA3 WMMA: A/B operands hold a row/column per lane (replicated in both
   half-waves); the accumulator holds column lane%16, rows 2e + lane/16.
4. A Metal command buffer holds 32 residency sets and an MPSGraph takes about
   two. The VAE encodes dozens of shapes, which aborted at small sizes (large
   ones hid it, MPSGraph commits by itself under load): run_graph flushes after
   CMD_MAX_GRAPHS distinct graphs.
5. Plain Metal tanh() returns NaN for large arguments (fast-math): use
   precise::tanh().

# Test Commands

Basic verification:
```bash
make test
```

`make test` starts with `make ropetest`, which needs no weights: rounding hides
wrong text encoder positions in the images, so it checks the RoPE tables directly.

Manual Flux sanity:
```bash
./iris -d flux-klein-4b -p "A fluffy orange cat sitting on a windowsill" \
  --seed 42 --steps 2 -o /tmp/iris_test.png -W 64 -H 64
```

Manual Z-Image sanity:
```bash
./iris -d zimage-turbo -p "a fish" --seed 43 --steps 8 -o /tmp/zimage_test.png
```

Manual FLUX.3 sanity:
```bash
./iris -d flux-3-klein -p "A fluffy orange cat sitting on a windowsill" \
  --seed 42 --steps 4 -W 256 -H 256 -o /tmp/flux3_test.png
```

Manual Qwen-Image sanity:
```bash
./iris -d qwen-image-2.1-turbo -p "A fluffy orange cat sitting on a windowsill" \
  --seed 42 -W 256 -H 256 -o /tmp/qimage_test.png
```

# Flux Known Pitfalls (Historical Bugs)

1. **Unified RoPE kernel indexing**: GPU must use consecutive pairs `(d, d+1)`, not axis-half indexing.
2. **GPU caching of timestep params**: step-dependent shift/scale/gate must not be cached as static weights.
3. **CLI mode CFG routing**: base models in interactive mode must go through `iris_generate()` (CFG-aware), not distilled-only embedding path.
