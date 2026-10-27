#!/usr/bin/env python3
"""
Iris test runner - verifies inference correctness against reference images.
Usage: python3 run_test.py [--binary PATH] [--full]
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Test cases: uses mean_diff threshold to allow bf16 precision differences
# while still catching actual bugs. Observed mean_diff values:
#   - 64x64: ~3.4, 512x512: ~1.7, img2img: ~6-17 (varies due to GPU non-determinism)
# Threshold of 20 accounts for GPU floating-point non-determinism while
# still catching catastrophic failures (wrong image would have mean_diff > 50).
# Optional: "input" for img2img tests
TESTS = [
    {
        "name": "64x64 quick test (2 steps)",
        "prompt": "A fluffy orange cat sitting on a windowsill",
        "seed": 42,
        "steps": 2,
        "width": 64,
        "height": 64,
        "reference": "test_vectors/reference_2step_64x64_seed42.png",
        "mean_diff_threshold": 20,
    },
    {
        "name": "512x512 full test (4 steps)",
        "prompt": "A red apple on a wooden table",
        "seed": 123,
        "steps": 4,
        "width": 512,
        "height": 512,
        "reference": "test_vectors/reference_4step_512x512_seed123.png",
        "mean_diff_threshold": 20,
    },
    {
        "name": "256x256 img2img test (4 steps)",
        "prompt": "A colorful oil painting of a cat",
        "seed": 456,
        "steps": 4,
        "width": 256,
        "height": 256,
        "input": "test_vectors/img2img_input_256x256.png",
        "reference": "test_vectors/reference_img2img_256x256_seed456.png",
        "mean_diff_threshold": 20,
    },
]

TESTS.extend([
    {
        "name": "FLUX.2 reference resolution independent of output",
        "prompt": "Create an exact copy of the input image.",
        "seed": 42, "steps": 2, "width": 64, "height": 64,
        "input": "test_vectors/img2img_input_256x256.png",
        "extra_args": ["--allow-large-images"],
        "expect_stderr": "Reference image: 256x256 (output 64x64)",
    },
    {
        "name": "FLUX.2 rejects impossible single-reference manual budget",
        "prompt": "Create an exact copy of the input image.",
        "seed": 42, "steps": 1, "width": 1920, "height": 1280,
        "input": "test_vectors/img2img_input_256x256.png",
        "extra_args": ["--allow-large-images"],
        "env": {"IRIS_USE_MANUAL_SDPA": "1"},
        "expect_error": "Output exceeds the 4 GiB attention budget",
        "reject_stderr": "VAE working buffers",
    },
    {
        "name": "FLUX.2 rejects impossible multi-reference manual budget",
        "prompt": "Create an exact copy of the input images.",
        "seed": 42, "steps": 1, "width": 1920, "height": 1280,
        "input": "test_vectors/img2img_input_256x256.png",
        "extra_args": ["--allow-large-images", "-i", "test_vectors/img2img_input_256x256.png"],
        "env": {"IRIS_USE_MANUAL_SDPA": "1"},
        "expect_error": "Output exceeds the 4 GiB attention budget",
        "reject_stderr": "VAE working buffers",
    },
    dict(TESTS[0], name="FLUX.2 manual attention numerical parity",
         env={"IRIS_USE_MANUAL_SDPA": "1"}),
    {
        "name": "FLUX.2 multiple references with native attention",
        "prompt": "Create an exact copy of the input images.",
        "seed": 42, "steps": 1, "width": 64, "height": 64,
        "input": "test_vectors/img2img_input_256x256.png",
        "extra_args": ["--allow-large-images", "-i", "test_vectors/img2img_input_256x256.png"],
        "expect_stderr": "VAE working buffers for 256x256: 0.38 GiB",
    },
])


# Full-only tests: these are slow and require visual inspection.
FULL_TESTS = [
    {
        "name": "1024x1024 img2img with attention budget shrinking (4 steps)",
        "prompt": "A blue sports car parked on a rainy city street at night",
        "seed": 99,
        "steps": 4,
        "width": 1024,
        "height": 1024,
        "expect_stderr": "reference image resized",
        "env": {"IRIS_USE_MANUAL_SDPA": "1"},
        "visual_check": "a blue sports car on a rainy city street at night, "
                        "output is 1024x1024",
    },
]

# Optional Z-Image smoke test.
# This runs only if a Z-Image model directory is auto-detected (or provided).
ZIMAGE_SMOKE_TEST = {
    "name": "Z-Image smoke test (2 steps, 256x256)",
    "prompt": "A simple geometric logo",
    "seed": 7,
    "steps": 2,
    "width": 256,
    "height": 256,
}


# Optional FLUX.3 tests, run only if a FLUX.3 model directory is found.
# The references were produced by debug/flux3_reference.py (official
# diffusers pipeline, float32) starting from the noise iris generates for
# seed 42.
FLUX3_TESTS = [
    {
        "name": "FLUX.3 Klein test (4 steps, 256x256)",
        "prompt": "A fluffy orange cat sitting on a windowsill",
        "seed": 42,
        "steps": 4,
        "width": 256,
        "height": 256,
        "reference": "test_vectors/reference_flux3_4step_256x256_seed42.png",
        "mean_diff_threshold": 20,
    },
    {
        "name": "FLUX.3 Klein edit test (4 steps, 256x256)",
        "prompt": "Turn the cat into a watercolor painting",
        "seed": 42,
        "steps": 4,
        "width": 256,
        "height": 256,
        "input": "test_vectors/img2img_input_256x256.png",
        "reference": "test_vectors/reference_flux3_edit_4step_256x256_seed42.png",
        "mean_diff_threshold": 20,
    },
]

# Exercise request-sized buffers against the same numerical references.
FLUX3_TESTS.extend([
    dict(FLUX3_TESTS[0], name="FLUX.3 request-sized txt2img buffers",
         extra_args=["--allow-large-images"],
         expect_stderr="VAE working buffers for 256x256: 0.38 GiB"),
    dict(FLUX3_TESTS[1], name="FLUX.3 request-sized reference buffers",
         extra_args=["--allow-large-images"]),
    dict(FLUX3_TESTS[0], name="FLUX.3 default ceiling rejects large output",
         width=1808, height=64, expect_error="Image dimensions exceed maximum (1792x1792)"),
    {
        "name": "FLUX.3 opt-in output above 1792 pixels",
        "prompt": "A simple geometric logo", "seed": 42, "steps": 1,
        "width": 1808, "height": 64,
        "extra_args": ["--allow-large-images"],
        "expect_stderr": "VAE working buffers for 1808x64: 0.66 GiB",
    },
    dict(FLUX3_TESTS[0], name="FLUX.3 rejects overflowing VAE tensor size",
         width=4096, height=4096, extra_args=["--allow-large-images"],
         expect_error="Image dimensions exceed supported VAE tensor size"),
    {
        "name": "FLUX.3 reference larger than output with request-sized buffers",
        "prompt": "Create an exact copy of the input image.", "seed": 42, "steps": 1,
        "width": 64, "height": 64, "input": "test_vectors/img2img_input_256x256.png",
        "extra_args": ["--allow-large-images"],
        "expect_stderr": "VAE working buffers for 256x256: 0.38 GiB",
    },
    {
        "name": "FLUX.3 multiple references reuse one VAE capacity",
        "prompt": "Create an exact copy of the input image.", "seed": 42, "steps": 1,
        "width": 64, "height": 64, "input": "test_vectors/img2img_input_256x256.png",
        "extra_args": ["--allow-large-images", "-i", "test_vectors/img2img_input_256x256.png"],
        "expect_stderr": "VAE working buffers for 256x256: 0.38 GiB",
    },
    {
        "name": "FLUX.3 quiet mode suppresses memory estimates",
        "prompt": "A simple geometric logo", "seed": 42, "steps": 1,
        "width": 64, "height": 64,
        "extra_args": ["--allow-large-images", "--quiet"],
        "reject_stderr": "VAE working buffers",
    },
])


def is_flux3_model_dir(model_dir: Path) -> bool:
    """Return True if model_dir looks like a FLUX.3 model directory."""
    config = model_dir / "transformer" / "config.json"
    try:
        return "Flux3Transformer2DModel" in config.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def is_zimage_model_dir(model_dir: Path) -> bool:
    """Return True if model_dir looks like a Z-Image model directory."""
    if not model_dir.is_dir():
        return False

    model_index = model_dir / "model_index.json"
    if not model_index.exists():
        return False

    try:
        text = model_index.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False

    return ("ZImagePipeline" in text) or ("Z-Image" in text)


def detect_model_dir(explicit_dir: Optional[str], common_names, matches) -> Optional[Path]:
    """Find a model directory for which matches(dir) is true."""
    if explicit_dir:
        p = Path(explicit_dir)
        return p if matches(p) else None

    # Common local names first.
    for name in common_names:
        if matches(Path(name)):
            return Path(name)

    # Fallback: scan direct subdirectories.
    for p in sorted(Path(".").iterdir()):
        if p.is_dir() and matches(p):
            return p

    return None


def run_test(binary: str, test: dict, model_dir: str) -> tuple[bool, str]:
    """Run a single test case. Returns (passed, message)."""
    if "output" in test:
        output_path = test["output"]
    else:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

    cmd = [
        binary,
        "-d", model_dir,
        "-p", test["prompt"],
        "--seed", str(test["seed"]),
        "--steps", str(test["steps"]),
        "-W", str(test["width"]),
        "-H", str(test["height"]),
        "-o", output_path,
    ]
    cmd.extend(test.get("extra_args", []))

    # Add input image for img2img tests
    if "input" in test:
        cmd.extend(["-i", test["input"]])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                env={**os.environ, **test.get("env", {})})
        if "expect_error" in test:
            if result.returncode != 1:
                return False, f"expected clean rejection (exit 1), got {result.returncode}"
            if test["expect_error"] not in result.stderr:
                return False, f"expected '{test['expect_error']}', got: {result.stderr}"
            if "reject_stderr" in test and test["reject_stderr"] in result.stderr:
                return False, f"unexpected '{test['reject_stderr']}' before rejection"
            return True, "request rejected as expected"
        if result.returncode != 0:
            return False, f"process exited with code {result.returncode}: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "timeout (300s)"
    except FileNotFoundError:
        return False, f"binary not found: {binary}"

    if "expect_stderr" in test and test["expect_stderr"] not in result.stderr:
        return False, f"expected '{test['expect_stderr']}' in stderr but not found"
    if "reject_stderr" in test and test["reject_stderr"] in result.stderr:
        return False, f"unexpected '{test['reject_stderr']}' in stderr"

    # If the test has no reference image, it's a visual-check-only test.
    if "reference" not in test:
        # Verify output was created and has expected dimensions.
        try:
            out = Image.open(output_path)
        except Exception as e:
            return False, f"failed to load output: {e}"
        if out.width != test["width"] or out.height != test["height"]:
            return False, (f"wrong output size: {out.width}x{out.height}, "
                           f"expected {test['width']}x{test['height']}")

        return True, f"output saved to {output_path}"

    # Compare images against reference
    try:
        ref = np.array(Image.open(test["reference"]))
        out = np.array(Image.open(output_path))
    except Exception as e:
        return False, f"failed to load images: {e}"

    if ref.shape != out.shape:
        return False, f"shape mismatch: ref={ref.shape}, out={out.shape}"

    diff = np.abs(ref.astype(float) - out.astype(float))
    max_diff = diff.max()
    mean_diff = diff.mean()

    threshold = test["mean_diff_threshold"]
    if mean_diff <= threshold:
        return True, f"mean_diff={mean_diff:.2f}, max_diff={max_diff:.0f}"
    else:
        return False, f"mean_diff={mean_diff:.2f} > {threshold} (max={max_diff:.0f})"


def main():
    parser = argparse.ArgumentParser(description="Run Iris inference tests")
    parser.add_argument("--flux-binary", default="./iris", help="Path to iris binary")
    parser.add_argument("--model-dir", default="flux-klein-4b", help="Path to model")
    parser.add_argument("--zimage-model-dir", default=None,
                        help="Optional Z-Image model dir (auto-detected if omitted)")
    parser.add_argument("--flux3-model-dir", default=None,
                        help="Optional FLUX.3 model dir (auto-detected if omitted)")
    parser.add_argument("--quick", action="store_true", help="Run only the quick 64x64 test")
    parser.add_argument("--full", action="store_true",
                        help="Also run slow tests that require visual inspection")
    args = parser.parse_args()

    if args.quick:
        tests_to_run = TESTS[:1]
    else:
        tests_to_run = list(TESTS)
    full_tests_to_run = list(FULL_TESTS) if args.full else []

    # Optional Z-Image and FLUX.3 coverage: run only in non-quick mode.
    scheduled_tests: list[tuple[dict, str]] = [(t, args.model_dir) for t in tests_to_run]
    optional = [
        ("Z-Image", "--zimage-model-dir", args.zimage_model_dir, [ZIMAGE_SMOKE_TEST],
         ("zimage-turbo", "zimage", "Z-Image-Turbo"), is_zimage_model_dir),
        ("FLUX.3", "--flux3-model-dir", args.flux3_model_dir, FLUX3_TESTS,
         ("flux-3-klein", "flux-3-klein-base"), is_flux3_model_dir),
    ]
    for label, flag, explicit, tests, names, matches in optional:
        if args.quick:
            break
        model_dir = detect_model_dir(explicit, names, matches)
        if model_dir:
            print(f"Detected {label} model dir: {model_dir}")
            scheduled_tests.extend((test, str(model_dir)) for test in tests)
        elif explicit:
            print(f"Warning: {flag} '{explicit}' is not a valid {label} model dir")
            print(f"Skipping optional {label} test.")
        else:
            print(f"No {label} model dir detected; skipping optional {label} test.")

    total = len(scheduled_tests) + len(full_tests_to_run)
    print(f"Running {total} test(s)...\n")

    passed = 0
    failed = 0
    visual_checks = []

    for i, (test, model_dir) in enumerate(scheduled_tests, 1):
        print(f"[{i}/{total}] {test['name']}...")
        ok, msg = run_test(args.flux_binary, test, model_dir)

        if ok:
            print(f"    PASS: {msg}")
            passed += 1
        else:
            print(f"    FAIL: {msg}")
            failed += 1

    for j, test in enumerate(full_tests_to_run, len(scheduled_tests) + 1):
        print(f"[{j}/{total}] {test['name']}...")

        # Step 1: Generate a reference image to use as img2img input.
        ref_path = "/tmp/iris_test_ref_1024.png"
        print(f"    Step 1: Generating 1024x1024 reference image...")
        ref_cmd = [
            args.flux_binary, "-d", args.model_dir,
            "-p", "A red sports car parked on a sunny city street",
            "--seed", "42", "--steps", "4",
            "-W", "1024", "-H", "1024", "-o", ref_path,
        ]
        try:
            r = subprocess.run(ref_cmd, capture_output=True, text=True,
                               timeout=300)
            if r.returncode != 0:
                print(f"    FAIL: could not generate reference: {r.stderr}")
                failed += 1
                continue
        except Exception as e:
            print(f"    FAIL: {e}")
            failed += 1
            continue
        print(f"    Step 1: Done ({ref_path})")

        # Step 2: Run img2img with the reference — this should trigger
        # the attention budget shrinking and print a resize note.
        output_path = "/tmp/iris_test_img2img_1024.png"
        print(f"    Step 2: Running img2img with attention budget "
              f"shrinking (reference should be auto-resized)...")
        test_with_input = dict(test)
        test_with_input["input"] = ref_path
        test_with_input["output"] = output_path
        ok, msg = run_test(args.flux_binary, test_with_input, args.model_dir)

        if ok:
            print(f"    Step 2: Done ({output_path})")
            print(f"    PASS: {msg}")
            passed += 1
            if "visual_check" in test:
                visual_checks.append((test["name"], output_path,
                                      test["visual_check"]))
        else:
            print(f"    FAIL: {msg}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")

    if visual_checks:
        print("\n--- Visual verification needed ---")
        for name, out_file, description in visual_checks:
            print(f"  {name}:")
            print(f"    Open: {out_file}")
            print(f"    Check: {description}")

    if not args.full and not args.quick:
        print("\nTo run a more complete (and slow) test, run with --full")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
