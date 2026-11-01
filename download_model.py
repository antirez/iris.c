#!/usr/bin/env python3
"""
Download FLUX.2-klein model files from HuggingFace.

Usage:
    python download_model.py MODEL [--token TOKEN] [--output-dir DIR]

Requirements:
    pip install huggingface_hub

This downloads the VAE, transformer, and Qwen3 text encoder needed for inference.
"""

import argparse
import sys
from pathlib import Path

MODELS = {
    "4b": ("black-forest-labs/FLUX.2-klein-4B", "./flux-klein-4b"),
    "4b-base": ("black-forest-labs/FLUX.2-klein-base-4B", "./flux-klein-4b-base"),
    "9b": ("black-forest-labs/FLUX.2-klein-9B", "./flux-klein-9b"),
    "9b-base": ("black-forest-labs/FLUX.2-klein-base-9B", "./flux-klein-9b-base"),
    "zimage-turbo": ("Tongyi-MAI/Z-Image-Turbo", "./zimage-turbo"),
    "flux3": ("black-forest-labs/flux-3-image-klein", "./flux-3-klein"),
    "flux3-base": ("black-forest-labs/flux-3-image-klein-base", "./flux-3-klein-base"),
    "qimage": ("Qwen/Qwen-Image-2.1", "./qwen-image-2.1"),
    "qimage-turbo": ("Qwen/Qwen-Image-2.1-Turbo", "./qwen-image-2.1-turbo"),
}

USAGE_TEXT = """\
FLUX.2-klein Model Downloader

Usage: python download_model.py MODEL [--token TOKEN] [--output-dir DIR]

Available models:

  4b            Distilled 4B (4 steps, fast, ~16 GB disk)
  4b-base       Base 4B (50 steps, CFG, higher quality, ~16 GB disk)
  9b            Distilled 9B (4 steps, higher quality, non-commercial, ~30 GB disk)
  9b-base       Base 9B (50 steps, CFG, highest quality, non-commercial, ~30 GB disk)
  zimage-turbo  Z-Image-Turbo 6B (8 NFE / 9 scheduler steps, Apache 2.0, ~22 GB disk)
  flux3         FLUX.3 Image Klein 5.6B (16 steps, Kommunity license, ~21 GB disk)
  flux3-base    FLUX.3 Image Klein Base 5.6B (50 steps, CFG, Kommunity license, ~21 GB disk)
  qimage        Qwen-Image 2.1 (40 steps, Qwen Research license: non-commercial, ~32 GB disk)
  qimage-turbo  Qwen-Image 2.1 Turbo (8 steps, Qwen Research license: non-commercial, ~32 GB disk)

By default this implementation uses mmap() so inference is often
possible with less RAM than the model size.

If this is your first time, we suggest downloading the "4b" model:
  python download_model.py 4b"""


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith('-'):
        print(USAGE_TEXT)
        return 1

    parser = argparse.ArgumentParser(
        description='Download FLUX.2-klein model files from HuggingFace'
    )
    parser.add_argument(
        'model',
        choices=list(MODELS.keys()),
        help='Model to download (4b, 4b-base, 9b, 9b-base)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory (default: auto based on model type)'
    )
    parser.add_argument(
        '--token', '-t',
        default=None,
        help='HuggingFace authentication token (for gated models like 9B)'
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed")
        print("Install with: pip install huggingface_hub")
        return 1

    # Determine token: CLI arg > env var
    token = args.token
    if not token:
        import os
        token = os.environ.get('HF_TOKEN')

    repo_id, default_dir = MODELS[args.model]
    output_dir = Path(args.output_dir if args.output_dir else default_dir)

    print(f"FLUX.2 Model Downloader")
    print("================================")
    print()
    print(f"Repository: {repo_id}")
    print(f"Output dir: {output_dir}")
    if token:
        print(f"Auth: using token")
    print()

    # Files to download - VAE, transformer, Qwen3 text encoder, and model_index.json
    # (Qwen-Image also needs its scheduler config and keeps the tokenizer in processor/)
    patterns = [
        "model_index.json",
        "modular_model_index.json",
        "vae/*.safetensors",
        "vae/*.json",
        "transformer/*.safetensors",
        "transformer/*.json",
        "text_encoder/*",
        "tokenizer/*",
        "scheduler/*.json",
        "processor/*",
    ]

    print("Downloading files...")
    print("(This may take a while depending on your connection)")
    print()

    try:
        model_dir = snapshot_download(
            repo_id,
            local_dir=str(output_dir),
            allow_patterns=patterns,
            ignore_patterns=["*.bin", "*.pt", "*.pth"],  # Skip pytorch format
            token=token,
        )
        print()
        print("Download complete!")
        print(f"Model saved to: {model_dir}")
        print()

        # Show file sizes
        vae_path = output_dir / "vae" / "diffusion_pytorch_model.safetensors"
        tf_path = output_dir / "transformer" / "diffusion_pytorch_model.safetensors"
        te_path = output_dir / "text_encoder"

        total_size = 0
        if vae_path.exists():
            vae_size = vae_path.stat().st_size
            total_size += vae_size
            print(f"  VAE:          {vae_size / 1024 / 1024:.1f} MB")
        if tf_path.exists():
            tf_size = tf_path.stat().st_size
            total_size += tf_size
            print(f"  Transformer:  {tf_size / 1024 / 1024 / 1024:.2f} GB")
        if te_path.exists():
            te_size = sum(f.stat().st_size for f in te_path.rglob("*") if f.is_file())
            total_size += te_size
            print(f"  Text encoder: {te_size / 1024 / 1024 / 1024:.2f} GB")

        if total_size > 0:
            print(f"  Total:        {total_size / 1024 / 1024 / 1024:.2f} GB")
        print()
        print("Usage:")
        print(f"  ./flux -d {output_dir} -p \"your prompt\" -o output.png")
        print()

    except Exception as e:
        error_msg = str(e)
        print(f"Error downloading: {e}")
        print()
        if '401' in error_msg or '403' in error_msg or 'auth' in error_msg.lower():
            print("Authentication required. For gated models (like 9B):")
            print("  1. Accept the license at https://huggingface.co/black-forest-labs/" +
                  repo_id.split('/')[-1])
            print("  2. Get your token from https://huggingface.co/settings/tokens")
            print(f"  3. Run: python download_model.py {args.model} --token YOUR_TOKEN")
            print("  Or set the HF_TOKEN env var")
        else:
            print("If you need to authenticate, run:")
            print("  huggingface-cli login")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
