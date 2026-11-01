#!/bin/bash
set -e

show_usage() {
    echo "FLUX.2-klein Model Downloader"
    echo ""
    echo "Usage: $0 MODEL [--token TOKEN]"
    echo ""
    echo "Available models:"
    echo ""
    echo "  4b            Distilled 4B (4 steps, fast, ~16 GB disk)"
    echo "  4b-base       Base 4B (50 steps, CFG, higher quality, ~16 GB disk)"
    echo "  9b            Distilled 9B (4 steps, higher quality, non-commercial, ~30 GB disk)"
    echo "  9b-base       Base 9B (50 steps, CFG, highest quality, non-commercial, ~30 GB disk)"
    echo "  zimage-turbo  Z-Image-Turbo 6B (8 NFE / 9 scheduler steps, Apache 2.0, ~22 GB disk)"
    echo "  flux3         FLUX.3 Image Klein 5.6B (16 steps, Kommunity license, ~21 GB disk)"
    echo "  flux3-base    FLUX.3 Image Klein Base 5.6B (50 steps, CFG, Kommunity license, ~21 GB disk)"
    echo "  qimage        Qwen-Image 2.1 (40 steps, Qwen Research license: non-commercial, ~32 GB disk)"
    echo "  qimage-turbo  Qwen-Image 2.1 Turbo (8 steps, Qwen Research license: non-commercial, ~32 GB disk)"
    echo ""
    echo "By default this implementation uses mmap() so inference is often"
    echo "possible with less RAM than the model size."
    echo ""
    echo "If this is your first time, we suggest downloading the \"4b\" model:"
    echo "  $0 4b"
    exit 1
}

# Need at least one argument
if [ $# -lt 1 ]; then
    show_usage
fi

# First positional argument is the model name
MODEL="$1"
shift

# Map model name to repo and output directory
case "$MODEL" in
    4b)
        REPO="FLUX.2-klein-4B"
        OUT="./flux-klein-4b"
        ;;
    4b-base)
        REPO="FLUX.2-klein-base-4B"
        OUT="./flux-klein-4b-base"
        ;;
    9b)
        REPO="FLUX.2-klein-9B"
        OUT="./flux-klein-9b"
        ;;
    9b-base)
        REPO="FLUX.2-klein-base-9B"
        OUT="./flux-klein-9b-base"
        ;;
    zimage-turbo)
        ORG="Tongyi-MAI"
        REPO="Z-Image-Turbo"
        OUT="./zimage-turbo"
        ;;
    flux3)
        REPO="flux-3-image-klein"
        OUT="./flux-3-klein"
        FLUX3=1
        ;;
    flux3-base)
        REPO="flux-3-image-klein-base"
        OUT="./flux-3-klein-base"
        FLUX3=1
        ;;
    qimage)
        ORG="Qwen"
        REPO="Qwen-Image-2.1"
        OUT="./qwen-image-2.1"
        QIMAGE=1
        ;;
    qimage-turbo)
        ORG="Qwen"
        REPO="Qwen-Image-2.1-Turbo"
        OUT="./qwen-image-2.1-turbo"
        QIMAGE=1
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo ""
        show_usage
        ;;
esac

# Parse remaining arguments
TOKEN=""
while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            TOKEN="$2"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            show_usage
            ;;
    esac
    shift
done

# Try to find token from environment
if [ -z "$TOKEN" ] && [ -n "$HF_TOKEN" ]; then
    TOKEN="$HF_TOKEN"
fi

if [ -n "$TOKEN" ]; then
    echo "Using authentication token"
fi

echo "Downloading $REPO..."

ORG="${ORG:-black-forest-labs}"
BASE="https://huggingface.co/$ORG/$REPO/resolve/main"

# Helper function to download with optional auth
dl() {
    local rc=0
    if [ -n "$TOKEN" ]; then
        curl -fL -H "Authorization: Bearer $TOKEN" -o "$1" "$2" || rc=$?
    else
        curl -fL -o "$1" "$2" || rc=$?
    fi
    if [ $rc -ne 0 ]; then
        echo ""
        echo "Error: failed to download $(basename "$1")"
        echo "URL: $2"
        echo ""
        if [ -z "$TOKEN" ]; then
            echo "This may be a gated model that requires authentication."
            echo "  1. Accept the license at https://huggingface.co/$ORG/$REPO"
            echo "  2. Get your token from https://huggingface.co/settings/tokens"
            echo "  3. Run: $0 $MODEL --token YOUR_TOKEN"
            echo "  Or set the HF_TOKEN env var"
        else
            echo "Authentication failed (HTTP 403). Possible causes:"
            echo "  - Token may be invalid or expired"
            echo "  - You may need to accept the license first:"
            echo "    https://huggingface.co/$ORG/$REPO"
            echo "  - The repository name may not exist (check spelling)"
        fi
        exit 1
    fi
}

if [ -n "$QIMAGE" ]; then
    # Qwen-Image: list the repository's files and fetch what iris reads
    # (the text encoder is sharded in one repository and a single file in
    # the other, and the tokenizer sits in processor/).
    FILES=$(curl -fsL ${TOKEN:+-H "Authorization: Bearer $TOKEN"} \
        "https://huggingface.co/api/models/$ORG/$REPO" | python3 -c "
import json, re, sys
for s in json.load(sys.stdin)['siblings']:
    f = s['rfilename']
    if re.match(r'(model_index\.json|scheduler/.*\.json|(transformer|vae|text_encoder)/.*\.(json|safetensors)|(processor|tokenizer)/.*)$', f):
        print(f)
")
    if [ -z "$FILES" ]; then
        echo "Error: could not list the files of $ORG/$REPO"
        exit 1
    fi
    for f in $FILES; do
        mkdir -p "$OUT/$(dirname "$f")"
        dl "$OUT/$f" "$BASE/$f"
    done
    echo "Done. -> $OUT"
    exit 0
fi

if [ -n "$FLUX3" ]; then
    # FLUX.3 ships a modular pipeline index and keeps the tokenizer in text_encoder/
    mkdir -p "$OUT"/{text_encoder,transformer,vae}
    dl "$OUT/modular_model_index.json" "$BASE/modular_model_index.json"
    TOK_DIR="text_encoder"
    TOK_FILES="chat_template.json merges.txt tokenizer.json tokenizer_config.json vocab.json"
else
    mkdir -p "$OUT"/{text_encoder,tokenizer,transformer,vae}
    # model_index.json (needed for autodetection)
    dl "$OUT/model_index.json" "$BASE/model_index.json"
    TOK_DIR="tokenizer"
    TOK_FILES="added_tokens.json chat_template.jinja merges.txt special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json"
fi

# text_encoder (Qwen3 - ~8GB for 4B, ~16GB for 9B)
dl "$OUT/text_encoder/config.json" "$BASE/text_encoder/config.json"
dl "$OUT/text_encoder/generation_config.json" "$BASE/text_encoder/generation_config.json"
dl "$OUT/text_encoder/model.safetensors.index.json" "$BASE/text_encoder/model.safetensors.index.json"

# Discover and download all safetensors shards from the index
SHARDS=$(python3 -c "
import json, sys
try:
    with open('$OUT/text_encoder/model.safetensors.index.json') as f:
        idx = json.load(f)
    shards = sorted(set(idx['weight_map'].values()))
    for s in shards:
        print(s)
except:
    # Fallback: assume 2 shards
    print('model-00001-of-00002.safetensors')
    print('model-00002-of-00002.safetensors')
" 2>/dev/null)

for shard in $SHARDS; do
    dl "$OUT/text_encoder/$shard" "$BASE/text_encoder/$shard"
done

# tokenizer
for f in $TOK_FILES; do
    dl "$OUT/$TOK_DIR/$f" "$BASE/$TOK_DIR/$f"
done

# transformer
dl "$OUT/transformer/config.json" "$BASE/transformer/config.json"

# Try to download transformer index (sharded models like 9B)
# Fall back to single file for non-sharded models (4B)
TF_INDEX="$OUT/transformer/diffusion_pytorch_model.safetensors.index.json"
curl -fL ${TOKEN:+-H "Authorization: Bearer $TOKEN"} -o "$TF_INDEX" \
    "$BASE/transformer/diffusion_pytorch_model.safetensors.index.json" 2>/dev/null || rm -f "$TF_INDEX"

if [ -f "$TF_INDEX" ]; then
    # Sharded: discover and download all shards
    TF_SHARDS=$(python3 -c "
import json
with open('$TF_INDEX') as f:
    idx = json.load(f)
shards = sorted(set(idx['weight_map'].values()))
for s in shards:
    print(s)
" 2>/dev/null)
    for shard in $TF_SHARDS; do
        dl "$OUT/transformer/$shard" "$BASE/transformer/$shard"
    done
else
    # Single file (4B distilled/base)
    dl "$OUT/transformer/diffusion_pytorch_model.safetensors" "$BASE/transformer/diffusion_pytorch_model.safetensors"
fi

# vae (~168 MB)
dl "$OUT/vae/config.json" "$BASE/vae/config.json"
dl "$OUT/vae/diffusion_pytorch_model.safetensors" "$BASE/vae/diffusion_pytorch_model.safetensors"

echo "Done. -> $OUT"
