#!/usr/bin/env bash
# Generic offline text-to-3D entry: bash run_infer.sh /input/prompts.jsonl /output
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "用法: bash run_infer.sh /input/prompts.jsonl /output" >&2
  exit 2
fi

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
PIPELINE_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
INPUT_JSONL="$1"
OUTPUT_ROOT="$2"
mkdir -p "${OUTPUT_ROOT}/_logs"
exec > >(tee -a "${OUTPUT_ROOT}/_logs/run.log") 2>&1
trap 'result=$?; echo "[infer] 退出码=${result}"; exit "${result}"' EXIT

BUNDLE_ROOT="${BUNDLE_ROOT:-$(cd "${PIPELINE_DIR}/.." && pwd)}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-${BUNDLE_ROOT}/weights}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="${HF_HOME:-${OUTPUT_ROOT}/_hf_cache}"
export TORCH_HOME="${TORCH_HOME:-${WEIGHTS_ROOT}/torch_home}"
export QWEN38_MODEL="${QWEN38_MODEL:-${WEIGHTS_ROOT}/Qwen3.8-27B}"
export QWEN_IMAGE_MODEL="${QWEN_IMAGE_MODEL:-${WEIGHTS_ROOT}/Qwen-Image-2512}"
export PIXAL_MODEL="${PIXAL_MODEL:-${WEIGHTS_ROOT}/Pixal3D}"
export DINO_MODEL="${DINO_MODEL:-${WEIGHTS_ROOT}/dinov3-vitl16}"
export RMBG_MODEL="${RMBG_MODEL:-${WEIGHTS_ROOT}/RMBG-2.0}"
export MOGE_MODEL="${MOGE_MODEL:-${WEIGHTS_ROOT}/moge-2-vitl/model.pt}"
export NAF_SOURCE_DIR="${NAF_SOURCE_DIR:-${BUNDLE_ROOT}/NAF}"
export PIXAL_ROOT="${PIXAL_ROOT:-${BUNDLE_ROOT}/Pixal3D-master}"
INPUT_HASH="$(sha256sum "${INPUT_JSONL}" | cut -d ' ' -f 1)"
export DATA_ROOT="${OUTPUT_ROOT}/_input/${INPUT_HASH}"
export WORK_ROOT="${OUTPUT_ROOT}/_work/${INPUT_HASH}"
export SUBMISSION_ROOT="${OUTPUT_ROOT}"
export SUBMISSION_ZIP="${OUTPUT_ROOT}/_logs/submission.zip"
export TEAM_ID="${TEAM_ID:-local}"
export TEAM_NAME="${TEAM_NAME:-text3d}"
export ROUND="${ROUND:-demo}"

for required in \
  "${QWEN38_MODEL}/config.json" \
  "${QWEN_IMAGE_MODEL}/model_index.json" \
  "${PIXAL_MODEL}/pipeline.json" \
  "${DINO_MODEL}/config.json" \
  "${RMBG_MODEL}/config.json" \
  "${MOGE_MODEL}" \
  "${NAF_SOURCE_DIR}/hubconf.py" \
  "${TORCH_HOME}/hub/checkpoints/naf_release.pth"; do
  if [[ ! -f "${required}" ]]; then
    echo "[infer] 缺少离线依赖: ${required}" >&2
    exit 1
  fi
done

PREPARE_PYTHON="${PREPARE_PYTHON:-${CUBE3D_PYTHON:-python3}}"
"${PREPARE_PYTHON}" "${PIPELINE_DIR}/verify_offline_assets.py"
export EXPECTED_COUNT
EXPECTED_COUNT="$("${PREPARE_PYTHON}" "${PIPELINE_DIR}/prepare_prompts_jsonl.py" "${INPUT_JSONL}" "${DATA_ROOT}")"
echo "[infer] 题目数=${EXPECTED_COUNT}，权重目录=${WEIGHTS_ROOT}"

for stage in plans-new references glbs postprocess render package; do
  echo "[infer] 开始 ${stage}"
  if ! bash "${PIPELINE_DIR}/run_pipeline.sh" "${stage}" > >(tee "${OUTPUT_ROOT}/_logs/${stage}.log") 2>&1; then
    echo "[infer] ${stage} 失败，详见 _logs/${stage}.log" >&2
    exit 1
  fi
done
echo "[infer] 完成，manifest=${OUTPUT_ROOT}/manifest.json"
