#!/usr/bin/env bash
# Build the RadiomicsRetrieval DB from a jsonl of samples.
#
# Edit the variables below to point at your own data / checkpoint, then run:
#     ./run_build_db.sh
# Any extra flags are forwarded to build_db.py, e.g.:
#     ./run_build_db.sh --limit 5

set -euo pipefail
# Resolve to repo's source/ directory regardless of the caller's cwd.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../source"

# ---- Inputs ----------------------------------------------------------------
# jsonl listing the samples to embed; each line must contain "id", "radiomics", "label".
JSONL=/workspace/RadiomicsRetrieval/data/NSCLC/train.jsonl

# Root that contains the three sample folders: images/, labels/, apes_npy/.
# Files are looked up as <DATA_ROOT>/{images,labels,apes_npy}/<id>.{nii.gz,npy}.
DATA_ROOT=/workspace/RadiomicsRetrieval/data/NSCLC

# Per-feature [min, max] used to normalize radiomics features before transtab.
# Computed from the train split; rarely needs to change.
MIN_MAX_JSON=${DATA_ROOT}/radiomics_features_min_max.json

# ---- Model checkpoint ------------------------------------------------------
# Parent folder containing the trained checkpoint directories.
RESULT_DIR=/workspace/RadiomicsRetrieval/results

# Subfolder under RESULT_DIR with the trained image-encoder weights.
MODEL_NAME=checkpoint-lung

# Full ckpt path — derived from the two above.
CKPT_DIR=${RESULT_DIR}/${MODEL_NAME}

# ---- Output ----------------------------------------------------------------
# Where the DB is written:
#   <OUTPUT_DIR>/img_embeddings/<id>.npy           image embedding per sample
#   <OUTPUT_DIR>/radiomics_normalized.json         {<id>: {feature: norm_value}}
#   <OUTPUT_DIR>/feature_names.json                ordered 72 feature names
#   <OUTPUT_DIR>/gt_labels.json                    {<id>: 'SCC'|'LCC'|'ADC'|'NOS'|'NaN'}
#   <OUTPUT_DIR>/radiomics_features_min_max.json   copy, used by retrieve.py
OUTPUT_DIR=${CKPT_DIR}/db

# ---- Runtime ---------------------------------------------------------------
# torch device — e.g. cuda:0 / cuda:1 / cpu
DEVICE=cuda:0

# Random seed (only affects the rare empty-bbox-center fallback).
SEED=42

# ---------------------------------------------------------------------------
python3 build_db.py \
    --jsonl        "${JSONL}" \
    --data-root    "${DATA_ROOT}" \
    --min-max-json "${MIN_MAX_JSON}" \
    --ckpt-dir     "${CKPT_DIR}" \
    --output-dir   "${OUTPUT_DIR}" \
    --device       "${DEVICE}" \
    --seed         "${SEED}" \
    "$@"
