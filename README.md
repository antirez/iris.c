# Iris - a C inference pipeline for image synthesis models

Iris is an inference pipeline that generates images from text prompts using open weights diffusion transformer models. It is implemented entirely in C, with zero external dependencies beyond the C standard library. MPS, CUDA, ROCm and BLAS acceleration are optional but recommended. Under macOS, a BLAS API is part of the system, so nothing is required.

The name comes from the Greek goddess Iris, messenger of the gods and personification of the rainbow.

Supported model families:

- **[FLUX.2 Klein](https://bfl.ai/models/flux-2-klein)** (by [Black Forest Labs](https://bfl.ai/)):
  - **4B distilled** (4 steps, auto guidance set to 1, very fast).
  - **4B base** (50 steps for max quality, or less. Classifier-Free Diffusion Guidance, much slower but more generation variety).
  - **9B distilled** (4 steps, larger model, higher quality. Non-commercial license).
  - **9B base** (50 steps, CFG, highest quality. Non-commercial license).
- **[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)** (by Tongyi-MAI):
  - **6B** (8 NFE / 9 scheduler steps, no CFG, fast).
- **[FLUX.3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein)** (by [Black Forest Labs](https://bfl.ai/)):
  - **5.6B distilled** (16 steps by default, trained for 4 to 16, no CFG. FLUX Kommunity License).
  - **5.6B base** (50 steps, CFG. FLUX Kommunity License). Implemented but not tested yet: the weights were not accessible at the time of writing.
  - Text-to-image, editing (`-i image`) and multi-reference generation (`-i` repeated).

All the models run fully on the GPU with the Metal (Apple), CUDA (NVIDIA) and ROCm (AMD) backends.

## Quick Start

```bash
# Build (choose your backend)
make mps       # Apple Silicon (fastest)
# or: make cuda    # NVIDIA GPU
# or: make rocm    # AMD GPU
# or: make blas    # Intel Mac / Linux with OpenBLAS
# or: make generic # Pure C, no dependencies

# Download a model (~16GB) - pick one:
./download_model.sh 4b                   # using curl
# or: pip install huggingface_hub && python download_model.py 4b

# Generate an image
./iris -d flux-klein-4b -p "A woman wearing sunglasses" -o output.png
```

If you want to try the base model, instead of the distilled one (much slower, higher quality), use the following instructions. Use 10 steps if your computer is quite slow, instead of the default of 50, it will still work well enough to test it (10 seconds to generate a 256x256 image on a MacBook M3 Max).
```
./download_model.sh 4b-base
# or: pip install huggingface_hub && python download_model.py 4b-base
./iris -d flux-klein-4b-base -p "A woman wearing sunglasses" -o output.png
```

If you want to try the 9B model (higher quality, non-commercial license, ~30GB download):
```bash
# 9B is a gated model - you need a HuggingFace token
# 1. Accept the license at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
# 2. Get your token from https://huggingface.co/settings/tokens
./download_model.sh 9b --token YOUR_TOKEN
# or: python download_model.py 9b --token YOUR_TOKEN
# or: set HF_TOKEN env var
./iris -d flux-klein-9b -p "A woman wearing sunglasses" -o output.png
```

For Z-Image-Turbo:
```bash
# Download Z-Image-Turbo (~12GB)
pip install huggingface_hub && python download_model.py zimage-turbo
./iris -d zimage-turbo -p "a fish" -o fish.png
```

For FLUX.3 Image Klein (gated, accept the license on HuggingFace first, ~20GB):
```bash
./download_model.sh flux3 --token YOUR_TOKEN
# or: python download_model.py flux3 --token YOUR_TOKEN
./iris -d flux-3-klein -p "a fish" -o fish.png
```

That's it. No Python runtime required at inference time.

## Example Output

![Woman with sunglasses](images/woman_with_sunglasses.png)

*Generated with: `./iris -d flux-klein-4b -p "A picture of a woman in 1960 America. Sunglasses. ASA 400 film. Black and White." -W 512 -H 512 -o woman.png`*

### Image-to-Image Example

![antirez to drawing](images/antirez_to_drawing.png)

*Generated with: `./iris -i antirez.png -o antirez_to_drawing.png -p "make it a drawing" -d flux-klein-4b`*

## Features

- **Zero dependencies**: Pure C implementation, works standalone. BLAS optional for ~30x speedup (Apple Accelerate on macOS, OpenBLAS on Linux)
- **GPU acceleration**: Metal on Apple Silicon Macs (automatic), CUDA on NVIDIA and ROCm on AMD GPUs. On Apple Silicon it is faster than PyTorch's optimized MPS pipeline
- **Runs where Python can't**: Memory-mapped weights (default) enable inference on 8GB RAM systems where the Python ML stack cannot run at all
- **Text-to-image**: Generate images from text prompts
- **Image-to-image**: Transform existing images guided by prompts (Flux models)
- **Multi-reference**: Combine multiple reference images (e.g., `-i car.png -i beach.png` for "car on beach")
- **Integrated text encoder**: Qwen3 encoder built-in (4B or 8B depending on model, Qwen3-VL 4B for FLUX.3), no external embedding computation needed
- **Memory efficient**: Automatic encoder release after encoding (up to ~16GB freed)
- **Memory-mapped weights**: Enabled by default. Reduces peak memory from ~16GB to ~4-5GB. Fastest mode on MPS; BLAS users with plenty of RAM may prefer `--no-mmap` for faster inference
- **Size-independent seeds**: Same seed produces similar compositions at different resolutions. Explore at 256x256, then render at 512x512 with the same seed
- **Terminal image display**: watch the resulting image without leaving your terminal (Ghostty, Kitty, iTerm2, WezTerm, or Konsole).

### Terminal Image Display

![Kitty protocol example](images/kitty-example.png)

Display generated images directly in your terminal with `--show`, or watch the denoising process step-by-step with `--show-steps`:

```bash
# Display final image in terminal (auto-detects Kitty/Ghostty/iTerm2/WezTerm/Konsole)
./iris -d flux-klein-4b -p "a cute robot" -o robot.png --show

# Display each denoising step (slower, but interesting to watch)
./iris -d flux-klein-4b -p "a cute robot" -o robot.png --show-steps
```

Requires a terminal supporting the [Kitty graphics protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/) (such as [Kitty](https://sw.kovidgoyal.net/kitty/) or [Ghostty](https://ghostty.org/)), the iTerm2 inline image protocol ([iTerm2](https://iterm2.com/), [WezTerm](https://wezfurlong.org/wezterm/)), or [Konsole](https://konsole.kde.org/). Terminal type is auto-detected from environment variables.

Use `--zoom N` to adjust the display size (default: 2 for Retina displays, use 1 for non-HiDPI screens).

## Usage

### Text-to-Image

```bash
./iris -d flux-klein-4b -p "A fluffy orange cat sitting on a windowsill" -o cat.png
```

### Image-to-Image

Transform an existing image based on a prompt:

```bash
./iris -d flux-klein-4b -p "oil painting style" -i photo.png -o painting.png
```

FLUX.2 uses **in-context conditioning** for image-to-image generation. Unlike traditional approaches that add noise to the input image, FLUX.2 passes the reference image as additional tokens that the model can attend to during generation. This means:

- The model "sees" your input image and uses it as a reference
- The prompt describes what you want the output to look like
- Results tend to preserve the composition while applying the described transformation

**Tips for good results:**
- Use descriptive prompts that describe the desired output, not instructions
- Good: `"oil painting of a woman with sunglasses, impressionist style"`
- Less good: `"make it an oil painting"` (instructional prompts may work less well)

**Super Resolution:** Since the reference image can be a different size than the output, you can use img2img for upscaling:

```bash
./iris -d flux-klein-4b -i small.png -W 1024 -H 1024 -o big.png -p "Create an exact copy of the input image."
```

The model will generate a higher-resolution version while preserving the composition and details of the input.

### Multi-Reference Generation

Combine elements from multiple reference images:

```bash
./iris -d flux-klein-4b -i car.png -i beach.png -p "a sports car on the beach" -o result.png
```

Each reference image is encoded separately and passed to the transformer with different positional embeddings (T=10, T=20, T=30, ...). The model attends to all references during generation, allowing it to combine elements from each.

**Example:**
- Reference 1: A red sports car
- Reference 2: A tropical beach with palm trees
- Prompt: "combine the two images"
- Result: A red sports car on a tropical beach

You can specify up to 16 reference images with multiple `-i` flags. The prompt guides how the references are combined.

### Interactive CLI Mode

Start without `-p` to enter interactive mode:

```bash
./iris -d flux-klein-4b
```

Generate images by typing prompts. Each image gets a `$N` reference ID:

```
iris> a red sports car
Done -> /tmp/iris-.../image-0001.png (ref $0)

iris> a tropical beach
Done -> /tmp/iris-.../image-0002.png (ref $1)

iris> $0 $1 combine them
Generating 256x256 (multi-ref, 2 images)...
Done -> /tmp/iris-.../image-0003.png (ref $2)
```

**Prompt syntax:**
- `prompt` - text-to-image
- `512x512 prompt` - set size inline
- `$ prompt` - img2img with last image
- `$N prompt` - img2img with reference $N
- `$0 $3 prompt` - multi-reference (combine images)

**Commands:** `!help`, `!save`, `!load`, `!seed`, `!size`, `!steps`, `!guidance`, `!linear`, `!power`, `!explore`, `!show`, `!quit`

### Command Line Options

**Required:**
```
-d, --dir PATH        Path to model directory
-p, --prompt TEXT     Text prompt for generation
-o, --output PATH     Output image path (.png or .ppm)
```

**Generation options:**
```
-W, --width N         Output width in pixels (default: 256)
-H, --height N        Output height in pixels (default: 256)
-s, --steps N         Sampling steps (default: auto, 4 distilled / 50 base / 9 zimage)
-S, --seed N          Random seed for reproducibility
-g, --guidance N      CFG guidance scale (default: auto, 1.0 distilled / 4.0 base / 0.0 zimage)
    --linear          Use linear timestep schedule (see below)
    --power           Use power curve timestep schedule (see below)
    --power-alpha N   Set power schedule exponent (default: 2.0)
    --base            Force base model mode (undistilled, CFG enabled)
```

**Image-to-image options:**
```
-i, --input PATH      Reference image (can be specified multiple times)
```

**Output options:**
```
-q, --quiet           Silent mode, no output
-v, --verbose         Show detailed config and timing info
    --show            Display image in terminal (auto-detects Kitty/Ghostty/iTerm2/WezTerm/Konsole)
    --show-steps      Display each denoising step (slower)
    --zoom N          Terminal image zoom factor (default: 2 for Retina)
```

**Other options:**
```
-m, --mmap            Memory-mapped weights (default, fastest on MPS)
    --no-mmap         Disable mmap, load all weights upfront
    --no-license-info Suppress non-commercial license warning (9B model)
-e, --embeddings PATH Load pre-computed text embeddings (advanced)
-h, --help            Show help
```

## Reproducibility

The seed is always printed to stderr, even when random:
```
$ ./iris -d flux-klein-4b -p "a landscape" -o out.png
Seed: 1705612345
...
Saving... out.png 256x256 (0.1s)
```

To reproduce the same image, use the printed seed:
```
$ ./iris -d flux-klein-4b -p "a landscape" -o out.png -S 1705612345
```

## PNG Metadata

Generated PNG images include metadata with the seed and model information, so you can always recover the seed even if you didn't save the terminal output:

```bash
# Using exiftool
exiftool image.png | grep iris

# Using Python/PIL
python3 -c "from PIL import Image; print(Image.open('image.png').info)"

# Using ImageMagick
identify -verbose image.png | grep -A1 "Properties:"
```

The following metadata fields are stored:
- `iris:seed` - The random seed used for generation
- `iris:model` - The model used
- `Software` - Program identifier

## Building

Choose a backend when building:

```bash
make            # Show available backends
make generic    # Pure C, no dependencies (slow)
make blas       # BLAS acceleration (~30x faster)
make mps        # Apple Silicon Metal GPU (fastest, macOS only)
make cuda       # NVIDIA GPU (needs the CUDA toolkit, with cuBLAS)
make rocm       # AMD GPU (needs ROCm, with hipBLAS)
```

**Recommended:**
- macOS Apple Silicon: `make mps`
- macOS Intel: `make blas`
- Linux with an NVIDIA / AMD GPU: `make cuda` / `make rocm`
- Linux with OpenBLAS: `make blas`
- Linux without OpenBLAS: `make generic`

The CUDA and ROCm backends run the whole pipeline of every model on the
GPU (text encoder, transformer, VAE), in bf16 with f32 accumulation. Add
`OPENBLAS=1` to use OpenBLAS for the CPU fallback (used when the GPU runs
out of memory). The build targets the GPU of the machine it runs on; set
`GPU_ARCH` (e.g. `GPU_ARCH=sm_121` or `GPU_ARCH=gfx1151`) to build for
another one, and `CUDA_HOME` / `ROCM_HOME` for non-standard install paths.

For `make blas` on Linux, install OpenBLAS first:
```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# Fedora
sudo dnf install openblas-devel
```

Other targets:
```bash
make clean      # Clean build artifacts
make info       # Show available backends for this platform
```

## Testing

Run the test suite to verify your build produces correct output:

```bash
make test        # Run all tests
make test-quick  # Run only the quick 64x64 test
```

The tests compare generated images against reference images in `test_vectors/`. A test passes if the maximum pixel difference is within tolerance (to allow for minor floating-point variations across platforms).

**Test cases:**
| Test | Size | Steps | Purpose |
|------|------|-------|---------|
| Quick | 64x64 | 2 | Fast txt2img sanity check |
| Full | 512x512 | 4 | Validates txt2img at larger resolution |
| img2img | 256x256 | 4 | Validates image-to-image transformation |
| Z-Image | 256x256 | 2 | Z-Image smoke test (auto-detected) |
| FLUX.3 | 256x256 | 4 | FLUX.3 Klein against the official pipeline (auto-detected) |

You can also run the test script directly for more options:
```bash
python3 run_test.py --help
python3 run_test.py --quick
python3 run_test.py --flux-binary ./iris --model-dir /path/to/model
```

## Model Download

Download model weights from HuggingFace using one of these methods:

**4B Distilled model** (~16GB, fast 4-step inference):
```bash
./download_model.sh 4b                   # using curl
# or: python download_model.py 4b        # using huggingface_hub
```

**4B Base model** (~16GB, 50-step inference with CFG, higher quality):
```bash
./download_model.sh 4b-base
# or: python download_model.py 4b-base
```

**9B models** (~30GB, higher quality, non-commercial license):
```bash
# 9B models are gated - require HuggingFace authentication
# 1. Accept the license at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
# 2. Get a token from https://huggingface.co/settings/tokens
./download_model.sh 9b --token YOUR_TOKEN       # distilled
./download_model.sh 9b-base --token YOUR_TOKEN   # base (CFG, highest quality)
# or: python download_model.py 9b --token YOUR_TOKEN
# You can also set the HF_TOKEN environment variable
```

**Z-Image-Turbo** (~12GB):
```bash
pip install huggingface_hub && python download_model.py zimage-turbo
```

**FLUX.3 Image Klein** (~20GB, FLUX Kommunity License):
```bash
# Gated: accept the license at https://huggingface.co/black-forest-labs/flux-3-image-klein
./download_model.sh flux3 --token YOUR_TOKEN        # distilled
./download_model.sh flux3-base --token YOUR_TOKEN   # base (CFG)
# or: python download_model.py flux3 --token YOUR_TOKEN
```

| Model | Directory | Size | Components |
|-------|-----------|------|------------|
| 4B distilled | `./flux-klein-4b` | ~16GB | VAE (~300MB), Transformer (~4GB), Qwen3-4B (~8GB) |
| 4B base | `./flux-klein-4b-base` | ~16GB | VAE (~300MB), Transformer (~4GB), Qwen3-4B (~8GB) |
| 9B distilled | `./flux-klein-9b` | ~30GB | VAE (~300MB), Transformer (~17GB), Qwen3-8B (~15GB) |
| 9B base | `./flux-klein-9b-base` | ~30GB | VAE (~300MB), Transformer (~17GB), Qwen3-8B (~15GB) |
| Z-Image-Turbo | `./zimage-turbo` | ~12GB | VAE, Transformer (~6B), Qwen3-4B |
| FLUX.3 distilled | `./flux-3-klein` | ~20GB | VAE (~300MB), Transformer (~11GB), Qwen3-VL-4B (~9GB) |
| FLUX.3 base | `./flux-3-klein-base` | ~20GB | VAE (~300MB), Transformer (~11GB), Qwen3-VL-4B (~9GB) |

## How Fast Is It?

Benchmarks on **Apple M3 Max** (128GB RAM), Flux distilled model (4 steps).

The MPS implementation is faster than the PyTorch optimized pipeline at all resolutions.

| Size | C (MPS) | PyTorch (MPS) |
|------|---------|---------------|
| 256x256 | 5.2s | 11s |
| 512x512 | 7.6s | 13s |
| 1024x1024 | 19s | 25s |

**Notes:**
- All times measured as wall clock, including model loading, no warmup. PyTorch times exclude library import overhead (~5-10s) to be fair.
- The base model is roughly 25x slower (50 steps x 2 passes per step vs 4 steps x 1 pass). It actually produces acceptable results even with 10 steps, so you can tune quality/time. The 25x figure is not exactly accurate because it only covers the denoising steps: text encoding and VAE use the same time for both the models, however such steps are a minor percentage of the generation time.
- The C BLAS backend (CPU) is not shown.
- The `make generic` backend (pure C, no BLAS) is approximately 30x slower than BLAS and not included in benchmarks.
- The fastest implementation for Metal remains [the Draw Things app](https://drawthings.ai/) that can produce a 1024x1024 image in just 14.23 seconds (in the same hardware), however it is worth noting that it uses 6-bit quantized weights, while this implementation uses the official BF16 weights. The 6-bit quantization used by Draw Things provides both a big memory win and a moderate speed advantage (not nearly as much as it could in an LLM, where causal attention is dominated by memory bandwidth); if we account for this, the performance is comparable.

### Timings on recent GPUs

FLUX.3 Image Klein distilled, default 16 steps, wall clock including model
loading:

| Size | Apple M5 Max (MPS) | NVIDIA GB10 (CUDA) | AMD Strix Halo, Radeon 8060S (ROCm) |
|------|--------------------|--------------------|--------------------------------------|
| 256x256 | 3.4s | 4.8s | 5.8s |
| 512x512 | 5.3s | 6.2s | 9.7s |
| 1024x1024 | 16.2s | 14.0s | 31.9s |

FLUX.2 Klein 4B distilled, 4 steps, 1024x1024, wall clock including model
loading: Apple M5 Max 6.2s, NVIDIA GB10 6.1s, AMD Strix Halo 9.1s.
Z-Image-Turbo, 8 NFE, 1024x1024: Apple M5 Max 16.1s, NVIDIA GB10 13.0s, AMD Strix Halo
23.3s.

Most of the time at small sizes is model loading (uploading the weights).

### Community Benchmarks

The following timings for 512x512 generation (Flux distilled model, 4 steps) were reported by users. They can serve as a rough indication of the performance you could expect, but results vary widely depending on the hardware, Metal availability (the code is heavily optimized for Apple Silicon via MPS), and whether BLAS acceleration is used on CPU.

| Hardware | Backend | 512x512 |
|----------|---------|---------|
| M3 Ultra | MPS | 4.5s |
| M3 Max | MPS | 7.6s |
| MacBook Pro M4 | MPS | 19s |
| MacBook Pro M1 Max | MPS | 39.9s |
| Apple M1 Pro | MPS | 42.4s |
| AMD Ryzen 7800X3D | BLAS | 47.8s |
| Intel i5-1135G7 | BLAS | 218s |

## Resolution Limits

**Default maximum resolution**: 1792x1792 pixels. To opt in to larger images
without rebuilding, use `--allow-large-images` together with the usual output
dimensions:

```sh
./iris -d flux-3-klein -i input.png -W 1920 -H 1280 \
  --allow-large-images -p "Create an exact copy of the input image." -o output.png
```

With this flag, VAE working buffers are allocated for the requested dimensions
instead of the default square maximum. Larger prepared references can grow the
buffers, and subsequent operations reuse their capacity. Normal output reports
the planned buffer allocation before it happens, for example:

```text
VAE working buffers for 1920x1280: 14.06 GiB (model and GPU memory additional)
```

This estimates the three CPU VAE working buffers, not total RAM use. More memory
is needed for model weights, attention, and GPU operations. Allocation can
succeed before the OS encounters memory pressure, so this flag does not
guarantee successful generation or prevent swapping. Model quality limits and
FLUX.3's reference-area limits still apply.

FLUX.2 keeps the reference's own resolution, independently of the output size,
when memory-efficient GPU attention is available. Manual attention retains its
4 GiB budget: references can be reduced to fit, and impossible output budgets
are rejected instead of continuing with a 16×16 reference. On Metal,
`IRIS_USE_MANUAL_SDPA=1` forces the conservative manual path for debugging.
The existing 4096-pixel CLI side limit and tensor indexing limits still apply.

Library callers can opt in with `iris_load_dir_ex(model_dir, 1)`; the existing
`iris_load_dir(model_dir)` keeps the default behavior.

**Minimum resolution**: 64x64 pixels.

Dimensions should be multiples of 16 (the VAE downsampling factor).

## Model Architecture

### FLUX.2 Klein

All Flux models share the same rectified flow transformer architecture, differing only in dimensions:

| Component | 4B | 9B |
|-----------|-----|-----|
| Transformer hidden | 3072 | 4096 |
| Attention heads | 24 | 32 |
| Head dim | 128 | 128 |
| Double blocks | 5 | 8 |
| Single blocks | 20 | 24 |
| Text Encoder | Qwen3-4B (2560 hidden, 36 layers) | Qwen3-8B (4096 hidden, 36 layers) |
| VAE | AutoencoderKL, 128 latent channels, 8x spatial compression | Same |

Architecture dimensions are read automatically from the model's config JSON files at load time.

The distilled and base variants differ in inference:

| | Distilled | Base |
|---|-----------|------|
| Steps | 4 | 50 (default) |
| CFG guidance | 1.0 (none) | 4.0 (default) |
| Passes per step | 1 | 2 (conditioned + unconditioned) |

The model type (distilled vs base, 4B vs 9B) is autodetected from the model directory. Use `--base` to force base model mode if autodetection fails.

**Classifier-Free Guidance (CFG)**: The base model runs the transformer twice per step -- once with an empty prompt (unconditioned) and once with the real prompt (conditioned). The final velocity is `v = v_uncond + guidance * (v_cond - v_uncond)`. This makes each step ~2x slower than the distilled model, and the base model needs ~12x more steps, making it roughly 25x slower overall.

### Z-Image-Turbo

Z-Image-Turbo uses an S3-DiT single-stream architecture with noise and context refiners:

| Component | Z-Image-Turbo |
|-----------|---------------|
| Transformer dim | 3840 |
| Attention heads | 30 |
| Head dim | 128 |
| Main layers | 30 |
| Refiner layers | 2 (noise) + 2 (context) |
| Text Encoder | Qwen3-4B (hidden_states[-2]) |
| VAE | 16 latent channels, patch_size=2 |

### FLUX.3 Image Klein

FLUX.3 processes three token modes (text, image, reference images): each first goes through its own blocks, attending only within the mode, then all of them are joined for the shared blocks:

| Component | FLUX.3 Image Klein |
|-----------|--------------------|
| Transformer hidden | 3072 |
| Attention heads | 24 (head dim 128) |
| Per-mode blocks | 5 per mode (separate weights for text, image, references) |
| Joint blocks | 28 over `[TEXT, IMAGE, REFS]` |
| Block | parallel attention + SwiGLU FFN, one gated residual |
| RoPE | 4 axes (t, h, w, l) x 32 dims, theta 10000, image positions scaled by 0.5 |
| Text Encoder | Qwen3-VL-4B text model, hidden_states[4, 8, ..., 32] concatenated (20480 dims) |
| VAE | Same as FLUX.2 |

Image tokens are modulated by the timestep, text and reference tokens by timestep 0. The distilled checkpoint is a flow map: every step is also told the timestep it jumps to. The prompt is padded to a multiple of 80 tokens and the transformer attends to the padding too, so the padding is part of the model recipe.

## Timestep Schedules

Each model family has its own default schedule. Alternative schedules (`--linear`, `--power`, `--sigmoid`, `--flowmatch`) are available for experimentation. Any schedule can be used with any model.

### Flux Distilled (4B / 9B)

The distilled models use a **shifted sigmoid** schedule (matching the official BFL distillation). This schedule concentrates most steps in the high-noise regime and is part of the distillation training -- changing it will produce poor results. Use 4 steps (the default).

### Flux Base (4B-base / 9B-base)

The base models default to the same shifted sigmoid schedule. At 50 steps it works very well, but 50 steps is slow. For quick tests, **10 steps** already produce decent results.

The shifted sigmoid can look extremely unbalanced at low step counts -- for example at 10 steps, the first 5 steps cover only 12% of the denoising trajectory while the last 5 cover 88%. The `--linear` flag switches to a uniform schedule where each step covers an equal portion of the trajectory, which sometimes produces more realistic results at reduced step counts. The `--power` flag provides a middle ground: a power curve (`t = 1 - (i/n)^a`) that is denser at the start and sparser at the end, but less extreme than the shifted sigmoid. The default exponent is 2.0 (quadratic); use `--power-alpha` to adjust it (1.0 = linear, higher = more front-loaded).

```bash
# Base model, 10 steps with default schedule
./iris -d flux-klein-4b-base -p "a cat" -o cat.png -s 10

# Base model with linear schedule
./iris -d flux-klein-4b-base -p "a cat" -o cat.png -s 10 --linear

# Base model with power schedule (quadratic by default)
./iris -d flux-klein-4b-base -p "a cat" -o cat.png -s 10 --power

# Power schedule with custom exponent
./iris -d flux-klein-4b-base -p "a cat" -o cat.png -s 10 --power-alpha 1.5
```

### Z-Image-Turbo

Z-Image-Turbo uses the official diffusers **FlowMatch Euler** schedule by default (static shift). The default is 8 NFE (9 scheduler values, with the terminal sigma at 0 making the last step a no-op).

For quick tests, **4 steps with `--linear`** works well and is twice as fast as the default:

```bash
# Default schedule (8 NFE)
./iris -d zimage-turbo -p "a fish" -o fish.png

# Quick test: 4 steps with linear schedule
./iris -d zimage-turbo -p "a fish" -o fish.png -s 4 --linear
```

### FLUX.3 Image Klein

FLUX.3 uses FlowMatch Euler with a large static shift (6.93), which keeps the noise level high for most of the run. The distilled model defaults to 16 steps but was trained for 4 to 16, so fewer steps are a valid speed/quality trade-off:

```bash
./iris -d flux-3-klein -p "a fish" -o fish.png          # 16 steps
./iris -d flux-3-klein -p "a fish" -o fish.png -s 4     # 4x faster
```

### Cross-model schedules

You can use any schedule with any model via `--sigmoid` and `--flowmatch`:

```bash
# Flux base with Z-Image's FlowMatch schedule
./iris -d flux-klein-4b-base -p "a cat" -o cat.png -s 10 --flowmatch

# Z-Image with Flux's shifted sigmoid schedule
./iris -d zimage-turbo -p "a fish" -o fish.png --sigmoid
```

This is not really useful with the Flux distilled models, but is interesting with both the base models of Flux and with Z-Image Turbo, even if it is a distilled model: the 9 steps training gives it enough flexibility, and other schedulers may be interesting especially at reduced steps count (quick preview).

### Interactive mode

In interactive CLI mode, toggle schedules with `!linear`, `!power [alpha]`, `!sigmoid`, or `!flowmatch`.

If you have a terminal supporting the iTerm2 or Kitty terminal graphics protocols, it is strongly suggested to test the different schedulers with `--show` and `--show-steps` options. It is quite an experience to see the denoising process happening in different ways.

## Memory Requirements

### 4B model

With mmap (default):

| Phase | Memory |
|-------|--------|
| Text encoding | ~2GB (layers loaded on-demand) |
| Diffusion | ~1-2GB (blocks loaded on-demand) |
| Peak | ~4-5GB |

With `--no-mmap` (all weights in RAM):

| Phase | Memory |
|-------|--------|
| Text encoding | ~8GB (encoder weights) |
| Diffusion | ~8GB (transformer ~4GB + VAE ~300MB + activations) |
| Peak | ~16GB (if encoder not released) |

### 9B model

With mmap (default):

| Phase | Memory |
|-------|--------|
| Text encoding | ~3-4GB (larger layers loaded on-demand) |
| Diffusion | ~2-3GB (more/larger blocks loaded on-demand) |
| Peak | ~8-10GB |

With `--no-mmap` (all weights in RAM):

| Phase | Memory |
|-------|--------|
| Text encoding | ~15GB (Qwen3-8B encoder weights) |
| Diffusion | ~17GB (transformer ~17GB + VAE ~300MB + activations) |
| Peak | ~32GB (if encoder not released) |

The text encoder is automatically released after encoding, reducing peak memory during diffusion. If you generate multiple images with different prompts, the encoder reloads automatically.

## Memory-Mapped Weights (Default)

Memory-mapped weight loading is enabled by default. Use `--no-mmap` to disable and load all weights upfront.

```bash
./iris -d flux-klein-4b -p "A cat" -o cat.png           # mmap (default)
./iris -d flux-klein-4b -p "A cat" -o cat.png --no-mmap # load all upfront
```

**How it works:** Instead of loading all model weights into RAM upfront, mmap keeps the safetensors files memory-mapped and loads weights on-demand:

- **Text encoder (Qwen3):** Each of the 36 transformer layers (~400MB each) is loaded, processed, and immediately freed. Only ~2GB stays resident instead of ~8GB.
- **Denoising transformer:** Each of the 5 double-blocks (~300MB) and 20 single-blocks (~150MB) is loaded on-demand and freed after use. Only ~200MB of shared weights stays resident instead of ~4GB.

This reduces peak memory from ~16GB to ~4-5GB, making inference possible on 16GB RAM systems where the Python ML stack cannot run at all.

**Performance varies by backend:**

- **MPS, CUDA, ROCm:** mmap is the **fastest** mode. The weights are stored in bf16 format and go from the memory-mapped files to GPU memory without conversion; the transformer weights are uploaded once, at load time.

- **BLAS (CPU):** mmap is **slightly slower** but uses much less RAM. BLAS requires f32 weights, so each block must be converted from bf16->f32 on every step (25 blocks x 4 steps = 100 conversions). With `--no-mmap`, this conversion happens once at startup. **Recommendation:** If you have 32GB+ RAM and use BLAS, try `--no-mmap` for faster inference. If RAM is limited, mmap lets you run at all.

- **Generic (pure C):** Same tradeoffs as BLAS, but slower overall.

## C Library API

The library can be integrated into your own C/C++ projects. Link against `libiris.a` and include `iris.h`.

### Text-to-Image Generation

Here's a complete program that generates an image from a text prompt:

```c
#include "iris.h"
#include <stdio.h>

int main(void) {
    /* Load the model. This loads VAE, transformer, and text encoder. */
    iris_ctx *ctx = iris_load_dir("flux-klein-4b");
    if (!ctx) {
        fprintf(stderr, "Failed to load model: %s\n", iris_get_error());
        return 1;
    }

    /* Configure generation parameters. Start with defaults and customize. */
    iris_params params = IRIS_PARAMS_DEFAULT;
    params.width = 512;
    params.height = 512;
    params.seed = 42;  /* Use -1 for random seed */

    /* Generate the image. This handles text encoding, diffusion, and VAE decode. */
    iris_image *img = iris_generate(ctx, "A fluffy orange cat in a sunbeam", &params);
    if (!img) {
        fprintf(stderr, "Generation failed: %s\n", iris_get_error());
        iris_free(ctx);
        return 1;
    }

    /* Save to file. Format is determined by extension (.png or .ppm). */
    iris_image_save(img, "cat.png");
    printf("Saved cat.png (%dx%d)\n", img->width, img->height);

    /* Clean up */
    iris_image_free(img);
    iris_free(ctx);
    return 0;
}
```

Compile with:
```bash
gcc -o myapp myapp.c -L. -liris -lm -framework Accelerate  # macOS
gcc -o myapp myapp.c -L. -liris -lm -lopenblas              # Linux
```

### Image-to-Image Transformation

Transform an existing image guided by a text prompt using in-context conditioning:

```c
#include "iris.h"
#include <stdio.h>

int main(void) {
    iris_ctx *ctx = iris_load_dir("flux-klein-4b");
    if (!ctx) return 1;

    /* Load the input image */
    iris_image *photo = iris_image_load("photo.png");
    if (!photo) {
        fprintf(stderr, "Failed to load image\n");
        iris_free(ctx);
        return 1;
    }

    /* Set up parameters. Output size defaults to input size. */
    iris_params params = IRIS_PARAMS_DEFAULT;
    params.seed = 123;

    /* Transform the image - describe the desired output */
    iris_image *painting = iris_img2img(ctx, "oil painting of the scene, impressionist style",
                                         photo, &params);
    iris_image_free(photo);  /* Done with input */

    if (!painting) {
        fprintf(stderr, "Transformation failed: %s\n", iris_get_error());
        iris_free(ctx);
        return 1;
    }

    iris_image_save(painting, "painting.png");
    printf("Saved painting.png\n");

    iris_image_free(painting);
    iris_free(ctx);
    return 0;
}
```

### Generating Multiple Images

When generating multiple images with different seeds but the same prompt, you can avoid reloading the text encoder:

```c
iris_ctx *ctx = iris_load_dir("flux-klein-4b");
iris_params params = IRIS_PARAMS_DEFAULT;
params.width = 256;
params.height = 256;

/* Generate 5 variations with different seeds */
for (int i = 0; i < 5; i++) {
    iris_set_seed(1000 + i);

    iris_image *img = iris_generate(ctx, "A mountain landscape at sunset", &params);

    char filename[64];
    snprintf(filename, sizeof(filename), "landscape_%d.png", i);
    iris_image_save(img, filename);
    iris_image_free(img);
}

iris_free(ctx);
```

Note: The text encoder (~8GB) is automatically released after the first generation to save memory. It reloads automatically if you use a different prompt.

### Error Handling

All functions that can fail return NULL on error. Use `iris_get_error()` to get a description:

```c
iris_ctx *ctx = iris_load_dir("nonexistent-model");
if (!ctx) {
    fprintf(stderr, "Error: %s\n", iris_get_error());
    /* Prints something like: "Failed to load VAE - cannot generate images" */
    return 1;
}
```

### API Reference

**Core functions:**
```c
iris_ctx *iris_load_dir(const char *model_dir);   /* Load model, returns NULL on error */
void iris_free(iris_ctx *ctx);                     /* Free all resources */

iris_image *iris_generate(iris_ctx *ctx, const char *prompt, const iris_params *params);
iris_image *iris_img2img(iris_ctx *ctx, const char *prompt, const iris_image *input,
                          const iris_params *params);
```

**Image handling:**
```c
iris_image *iris_image_load(const char *path);     /* Load PNG, JPEG, or PPM */
int iris_image_save(const iris_image *img, const char *path);  /* 0=success, -1=error */
int iris_image_save_with_seed(const iris_image *img, const char *path, int64_t seed);  /* Save with metadata */
iris_image *iris_image_resize(const iris_image *img, int new_w, int new_h);
void iris_image_free(iris_image *img);
```

**Utilities:**
```c
void iris_set_seed(int64_t seed);                  /* Set RNG seed for reproducibility */
const char *iris_get_error(void);                  /* Get last error message */
void iris_release_text_encoder(iris_ctx *ctx);     /* Manually free ~8GB (optional) */
int iris_is_distilled(iris_ctx *ctx);              /* 1 = distilled, 0 = base */
void iris_set_base_mode(iris_ctx *ctx);            /* Force base model mode */
```

### Parameters

```c
typedef struct {
    int width;              /* Output width in pixels (default: 256) */
    int height;             /* Output height in pixels (default: 256) */
    int num_steps;          /* Denoising steps, 0 = auto (4 distilled, 50 base, 9 zimage) */
    int64_t seed;           /* Random seed, -1 for random (default: -1) */
    float guidance;         /* CFG guidance scale, 0 = auto (1.0 distilled, 4.0 base, 0.0 zimage) */
    int linear_schedule;    /* Use linear timestep schedule (0 = shifted sigmoid) */
    int power_schedule;     /* Use power curve timestep schedule */
    float power_alpha;      /* Exponent for power schedule (default: 2.0) */
} iris_params;

/* Initialize with sensible defaults (auto steps and guidance from model type) */
#define IRIS_PARAMS_DEFAULT { 256, 256, 0, -1, 0.0f, 0, 0, 2.0f }
```

## Debugging

### Comparing with Python Reference

When debugging img2img issues, the `--debug-py` flag allows you to run the C implementation with exact inputs saved from a Python reference script. This isolates whether differences are due to input preparation (VAE encoding, text encoding, noise generation) or the transformer itself.

**Setup:**

1. Set up the Python environment:
```bash
python -m venv flux_env
source flux_env/bin/activate
pip install torch diffusers transformers safetensors einops huggingface_hub
```

2. Clone the flux2 reference (for the model class):
```bash
git clone https://github.com/black-forest-labs/flux flux2
```

3. Run the Python debug script to save inputs:
```bash
python debug/debug_img2img_compare.py
```

This saves to `/tmp/`:
- `py_noise.bin` - Initial noise tensor
- `py_ref_latent.bin` - VAE-encoded reference image
- `py_text_emb.bin` - Text embeddings from Qwen3

4. Run C with the same inputs:
```bash
./iris -d flux-klein-4b --debug-py -W 256 -H 256 --steps 4 -o /tmp/c_debug.png
```

5. Compare the outputs visually or numerically.

**What this helps diagnose:**
- If C and Python produce identical outputs with identical inputs, any differences in normal operation are due to input preparation (VAE, text encoder, RNG)
- If outputs differ even with identical inputs, the issue is in the transformer or sampling implementation

### Debug Scripts

The `debug/` directory contains Python scripts for comparing C and Python implementations:

- `debug_img2img_compare.py` - Full img2img comparison with step-by-step statistics
- `debug_rope_img2img.py` - Verify RoPE position encoding matches between C and Python

## License

MIT
