#!/usr/bin/env bash
set -euo pipefail

# One-click pipeline:
# 1) Put your video at:   examples/data/user_video_drop/input.mp4
# 2) Run this script:     bash examples/train_from_video.sh <scene_name>
#
# It will:
# - extract frames with ffmpeg
# - run COLMAP SfM
# - launch gsplat training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCENE_NAME="${1:-my_scene}"
INPUT_DIR="${SCRIPT_DIR}/data/user_video_drop"
INPUT_VIDEO="${INPUT_DIR}/input.mp4"

DATA_ROOT="${SCRIPT_DIR}/data/user_capture"
RESULT_ROOT="${SCRIPT_DIR}/results/user_capture"
DATA_DIR="${DATA_ROOT}/${SCENE_NAME}"
RESULT_DIR="${RESULT_ROOT}/${SCENE_NAME}"

IMAGES_DIR="${DATA_DIR}/images"
SPARSE_DIR="${DATA_DIR}/sparse"
DB_PATH="${DATA_DIR}/database.db"

FPS="${FPS:-2}"
DATA_FACTOR="${DATA_FACTOR:-4}"
PORT="${PORT:-8080}"
MAX_STEPS="${MAX_STEPS:-30000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

echo "[INFO] Scene: ${SCENE_NAME}"
echo "[INFO] Input video: ${INPUT_VIDEO}"
echo "[INFO] Data dir: ${DATA_DIR}"
echo "[INFO] Result dir: ${RESULT_DIR}"

if [[ ! -f "${INPUT_VIDEO}" ]]; then
  echo "[ERROR] Input video not found."
  echo "[ERROR] Please put your video at: ${INPUT_VIDEO}"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[ERROR] ffmpeg not found."
  exit 1
fi

if ! command -v colmap >/dev/null 2>&1; then
  echo "[ERROR] colmap not found."
  exit 1
fi

mkdir -p "${IMAGES_DIR}" "${SPARSE_DIR}" "${RESULT_DIR}"

echo "[INFO] Cleaning previous extracted frames and COLMAP DB for this scene."
rm -f "${DB_PATH}"
find "${IMAGES_DIR}" -type f \( -name '*.jpg' -o -name '*.png' \) -delete || true
find "${SPARSE_DIR}" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + || true

echo "[INFO] Extracting frames from video (fps=${FPS}) ..."
ffmpeg -y -i "${INPUT_VIDEO}" -vf "fps=${FPS}" -q:v 2 "${IMAGES_DIR}/%06d.jpg" >/dev/null 2>&1

FRAME_COUNT="$(find "${IMAGES_DIR}" -type f -name '*.jpg' | wc -l | tr -d ' ')"
if [[ "${FRAME_COUNT}" -lt 20 ]]; then
  echo "[ERROR] Too few frames extracted: ${FRAME_COUNT}. Increase video length or FPS."
  exit 1
fi
echo "[INFO] Extracted ${FRAME_COUNT} frames."

echo "[INFO] Running COLMAP feature extraction..."
if ! colmap feature_extractor \
  --database_path "${DB_PATH}" \
  --image_path "${IMAGES_DIR}" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV \
  --SiftExtraction.use_gpu 1; then
  echo "[WARN] GPU feature extraction failed, retrying on CPU."
  colmap feature_extractor \
    --database_path "${DB_PATH}" \
    --image_path "${IMAGES_DIR}" \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.use_gpu 0
fi

echo "[INFO] Running COLMAP matching..."
if ! colmap exhaustive_matcher \
  --database_path "${DB_PATH}" \
  --SiftMatching.use_gpu 1; then
  echo "[WARN] GPU matching failed, retrying on CPU."
  colmap exhaustive_matcher \
    --database_path "${DB_PATH}" \
    --SiftMatching.use_gpu 0
fi

echo "[INFO] Running COLMAP mapper..."
colmap mapper \
  --database_path "${DB_PATH}" \
  --image_path "${IMAGES_DIR}" \
  --output_path "${SPARSE_DIR}"

if [[ ! -d "${SPARSE_DIR}/0" ]]; then
  FIRST_MODEL_DIR="$(find "${SPARSE_DIR}" -mindepth 1 -maxdepth 1 -type d | head -n 1 || true)"
  if [[ -n "${FIRST_MODEL_DIR}" ]]; then
    ln -sfn "${FIRST_MODEL_DIR}" "${SPARSE_DIR}/0"
  else
    echo "[ERROR] COLMAP mapper did not produce sparse model."
    exit 1
  fi
fi

echo "[INFO] Starting gsplat training..."
cd "${SCRIPT_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python simple_trainer.py default \
  --data_dir "${DATA_DIR}" \
  --data_factor "${DATA_FACTOR}" \
  --result_dir "${RESULT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --port "${PORT}" \
  ${TRAIN_EXTRA_ARGS}
