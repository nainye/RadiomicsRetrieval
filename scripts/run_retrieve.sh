#!/usr/bin/env bash
# Retrieve top-k similar samples from a RadiomicsRetrieval DB.
#
# Edit the variables below for your search, then run:
#     ./run_retrieve.sh
# Any extra flags are forwarded to retrieve.py.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../source"

# ---- Search mode -----------------------------------------------------------
# One of: img | rad | shape | firstorder | hist | texture | glcm | glszm | feature
#   img         image embedding cosine
#   rad         all 72 radiomics features → transtab embedding cosine
#   shape       Shape features (14)
#   firstorder  First-order / histogram features (18)        (alias: hist)
#   texture     GLCM ∪ GLSZM (40)
#   glcm        GLCM only (24)
#   glszm       GLSZM only (16)
#   feature     a single feature, set FEATURE_NAME below
MODE=img

# Used only when MODE=feature. Must match an entry in feature_names.json
# inside the DB (e.g. Shape_Elongation, Hist_Entropy, GLCM_Contrast, ...).
FEATURE_NAME=Shape_Elongation

# ---- Query -----------------------------------------------------------------
# Pick exactly ONE of (A) or (B). Leave the other blank.
#
# (A) Use a sample already in the DB:
QUERY_ID=

# (B) External sample — fill these and clear QUERY_ID above.
#     APE is only needed when MODE=img.
IMAGE=/workspace/RadiomicsRetrieval/data/NSCLC/images/LUNG1-001_1.nii.gz
SEG=/workspace/RadiomicsRetrieval/data/NSCLC/labels/LUNG1-001_1.nii.gz
APE=/workspace/RadiomicsRetrieval/data/NSCLC/apes_npy/LUNG1-001_1.npy
# Optional: ground-truth label of the external query (enables label_match ✓).
# One of: SCC | LCC | ADC | NOS | NaN
QUERY_LABEL=LCC

# ---- DB & checkpoint -------------------------------------------------------
DB_DIR=/workspace/RadiomicsRetrieval/results/checkpoint-lung/db
CKPT_DIR=/workspace/RadiomicsRetrieval/results/checkpoint-lung

# ---- Output ----------------------------------------------------------------
TOP_K=10
# Leave blank to only print results; set a path to also dump JSON.
OUTPUT_JSON=

# ---- Runtime ---------------------------------------------------------------
DEVICE=cuda:0
SEED=42
BATCH_SIZE=64
# Set to 1 to skip the per-mode embedding cache on disk.
NO_CACHE=0
# Set to 1 to force re-forwarding the DB even if a cache exists.
REBUILD_CACHE=0
# Set to 1 to skip the auxiliary raw-radiomics-cosine and label-match columns.
NO_AUX=0

# ---------------------------------------------------------------------------
ARGS=(
    --mode      "${MODE}"
    --db-dir    "${DB_DIR}"
    --ckpt-dir  "${CKPT_DIR}"
    --top-k     "${TOP_K}"
    --device    "${DEVICE}"
    --seed      "${SEED}"
    --batch-size "${BATCH_SIZE}"
)

[ "${MODE}" = "feature" ] && ARGS+=( --feature-name "${FEATURE_NAME}" )

if [ -n "${QUERY_ID:-}" ]; then
    ARGS+=( --query-id "${QUERY_ID}" )
else
    [ -z "${IMAGE:-}" ] || [ -z "${SEG:-}" ] && {
        echo "Set QUERY_ID, or set both IMAGE and SEG (and APE for MODE=img)." >&2
        exit 1
    }
    ARGS+=( --image "${IMAGE}" --seg "${SEG}" )
    [ "${MODE}" = "img" ] && ARGS+=( --ape "${APE}" )
    [ -n "${QUERY_LABEL:-}" ] && ARGS+=( --query-label "${QUERY_LABEL}" )
fi

[ -n "${OUTPUT_JSON:-}" ] && ARGS+=( --output-json "${OUTPUT_JSON}" )
[ "${NO_CACHE}" = "1" ]      && ARGS+=( --no-cache )
[ "${REBUILD_CACHE}" = "1" ] && ARGS+=( --rebuild-cache )
[ "${NO_AUX}" = "1" ]        && ARGS+=( --no-aux )

python3 retrieve.py "${ARGS[@]}" "$@"
