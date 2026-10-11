#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="${BUNDLE_ROOT:-$(cd "${PIPELINE_DIR}/.." && pwd)}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-${BUNDLE_ROOT}/weights}"

DATA_ROOT="${DATA_ROOT:-${PIPELINE_DIR}/data}"
WORK_ROOT="${WORK_ROOT:-${PIPELINE_DIR}/work}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-${PIPELINE_DIR}/submission}"
SUBMISSION_ZIP="${SUBMISSION_ZIP:-${PIPELINE_DIR}/submission.zip}"
EXPECTED_COUNT="${EXPECTED_COUNT:-1}"

CUBE3D_PYTHON="${CUBE3D_PYTHON:-python3}"
QWEN_IMAGE_PYTHON="${QWEN_IMAGE_PYTHON:-python3}"
PIXAL_PYTHON="${PIXAL_PYTHON:-python3}"
BLENDER_BIN="${BLENDER_BIN:-blender}"

QWEN38_MODEL="${QWEN38_MODEL:-${WEIGHTS_ROOT}/Qwen3.8-27B}"
QWEN_IMAGE_MODEL="${QWEN_IMAGE_MODEL:-${WEIGHTS_ROOT}/Qwen-Image-2512}"
PIXAL_ROOT="${PIXAL_ROOT:-${BUNDLE_ROOT}/Pixal3D-master}"
PIXAL_MODEL="${PIXAL_MODEL:-${WEIGHTS_ROOT}/Pixal3D}"
PIXAL_RESOLUTION="${PIXAL_RESOLUTION:-1536}"
# Manufacturing repair now belongs to stage 04.  Keep the older Pixal-side
# printable remesh available as an opt-in, but avoid doing two SDF rebuilds.
PIXAL_PRINTABLE="${PIXAL_PRINTABLE:-0}"
PIXAL_MIN_WALL_MM="${PIXAL_MIN_WALL_MM:-1.2}"
PIXAL_PRINT_VOXEL_MM="${PIXAL_PRINT_VOXEL_MM:-0.6}"
PIXAL_MIN_COMPONENT_FACES="${PIXAL_MIN_COMPONENT_FACES:-50}"
POST_MIN_COMPONENT_FACES="${POST_MIN_COMPONENT_FACES:-50}"
POST_ORIENTATION_MODE="${POST_ORIENTATION_MODE:-semantic}"
POST_BOTTOM_MAX_TILT_DEG="${POST_BOTTOM_MAX_TILT_DEG:-15}"
POST_BASE_CUT_MM="${POST_BASE_CUT_MM:-0.4}"
POST_VOXEL_MM="${POST_VOXEL_MM:-0.6}"
POST_SDF_OFFSET_MM="${POST_SDF_OFFSET_MM:-0.8}"
POST_MAX_VOXEL_AXIS="${POST_MAX_VOXEL_AXIS:-384}"
TEXTURE_RESOLUTION="${TEXTURE_RESOLUTION:-2048}"
SKIP_TEXTURE_BAKE="${SKIP_TEXTURE_BAKE:-0}"
EMBED_UV_TEXTURE="${EMBED_UV_TEXTURE:-0}"
ONLY_PROMPT_ID="${ONLY_PROMPT_ID:-}"
REFERENCE_CANDIDATES="${REFERENCE_CANDIDATES:-3}"
REFERENCE_SELECTOR="${REFERENCE_SELECTOR:-vlm}"
TEAM_ID="${TEAM_ID:-local}"
TEAM_NAME="${TEAM_NAME:-text3d}"
ROUND="${ROUND:-demo}"
export HF_HOME="${HF_HOME:-${BUNDLE_ROOT}/.cache/huggingface}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

ONLY_PROMPT_ARGS=()
if [[ -n "${ONLY_PROMPT_ID}" ]]; then
  ONLY_PROMPT_ARGS=(--prompt-id "${ONLY_PROMPT_ID}")
fi

TEXTURE_BAKE_ARGS=(--texture-resolution "${TEXTURE_RESOLUTION}")
if [[ "${SKIP_TEXTURE_BAKE}" == "1" ]]; then
  TEXTURE_BAKE_ARGS+=(--skip-texture-bake)
fi
if [[ "${EMBED_UV_TEXTURE}" == "1" ]]; then
  TEXTURE_BAKE_ARGS+=(--embed-uv-texture)
fi

PIXAL_PRINTABLE_ARGS=()
if [[ "${PIXAL_PRINTABLE}" == "1" ]]; then
  PIXAL_PRINTABLE_ARGS=(
    --printable
    --min-wall-mm "${PIXAL_MIN_WALL_MM}"
    --print-voxel-mm "${PIXAL_PRINT_VOXEL_MM}"
    --min-component-faces "${PIXAL_MIN_COMPONENT_FACES}"
  )
fi

run_stage() {
  local stage="$1"
  case "${stage}" in
    plans)
      "${CUBE3D_PYTHON}" "${PIPELINE_DIR}/01_generate_plans.py" \
        --data-root "${DATA_ROOT}" \
        --work-root "${WORK_ROOT}" \
        --expected-count "${EXPECTED_COUNT}" \
        --gpus 0,1 \
        --model "${QWEN38_MODEL}" \
        "${OVERWRITE_ARGS[@]}"
      ;;
    plans-new)
      "${CUBE3D_PYTHON}" "${PIPELINE_DIR}/01_generate_plans_new.py" \
        --data-root "${DATA_ROOT}" \
        --work-root "${WORK_ROOT}" \
        --expected-count "${EXPECTED_COUNT}" \
        --gpus 0,1 \
        --model "${QWEN38_MODEL}" \
        "${OVERWRITE_ARGS[@]}"
      ;;
    recover-plans)
      "${CUBE3D_PYTHON}" "${PIPELINE_DIR}/01_generate_plans.py" \
        --data-root "${DATA_ROOT}" \
        --work-root "${WORK_ROOT}" \
        --expected-count "${EXPECTED_COUNT}" \
        --model "${QWEN38_MODEL}" \
        --recover-responses
      ;;
    references|images)
      # This Python process exits before the selector starts, so Qwen-Image's
      # two-GPU allocation is fully released before Qwen3.8 is loaded.
      "${QWEN_IMAGE_PYTHON}" "${PIPELINE_DIR}/02_generate_references.py" \
        --data-root "${DATA_ROOT}" \
        --work-root "${WORK_ROOT}" \
        --expected-count "${EXPECTED_COUNT}" \
        --gpus 0,1 \
        --model "${QWEN_IMAGE_MODEL}" \
        --candidates "${REFERENCE_CANDIDATES}" \
        "${OVERWRITE_ARGS[@]}"
      "${CUBE3D_PYTHON}" "${PIPELINE_DIR}/02_select_references.py" \
        --data-root "${DATA_ROOT}" \
        --work-root "${WORK_ROOT}" \
        --expected-count "${EXPECTED_COUNT}" \
        --gpus 0,1 \
        --model "${QWEN38_MODEL}" \
        --selector "${REFERENCE_SELECTOR}" \
        "${OVERWRITE_ARGS[@]}"
      ;;
    glbs)
      "${PIXAL_PYTHON}" "${PIPELINE_DIR}/03_generate_raw_glbs.py" \
        --data-root "${DATA_ROOT}" \
        --work-root "${WORK_ROOT}" \
        --expected-count "${EXPECTED_COUNT}" \
        --pixal-root "${PIXAL_ROOT}" \
        --model "${PIXAL_MODEL}" \
        --resolution "${PIXAL_RESOLUTION}" \
        --gpus 0,1 \
        "${PIXAL_PRINTABLE_ARGS[@]}" \
        "${ONLY_PROMPT_ARGS[@]}" \
        "${OVERWRITE_ARGS[@]}"
      ;;
    postprocess)
      "${PIXAL_PYTHON}" "${PIPELINE_DIR}/04_postprocess_glbs.py" \
        --data-root "${DATA_ROOT}" \
        --work-root "${WORK_ROOT}" \
        --expected-count "${EXPECTED_COUNT}" \
        --min-component-faces "${POST_MIN_COMPONENT_FACES}" \
        --orientation-mode "${POST_ORIENTATION_MODE}" \
        --bottom-plane-max-tilt-deg "${POST_BOTTOM_MAX_TILT_DEG}" \
        --base-cut-mm "${POST_BASE_CUT_MM}" \
        --voxel-mm "${POST_VOXEL_MM}" \
        --sdf-offset-mm "${POST_SDF_OFFSET_MM}" \
        --max-voxel-axis "${POST_MAX_VOXEL_AXIS}" \
        "${ONLY_PROMPT_ARGS[@]}" \
        "${OVERWRITE_ARGS[@]}"
      ;;
    render)
      "${BLENDER_BIN}" --background \
        --python "${PIPELINE_DIR}/05_render_blender.py" \
        -- \
        --work-root "${WORK_ROOT}" \
        "${TEXTURE_BAKE_ARGS[@]}" \
        "${ONLY_PROMPT_ARGS[@]}" \
        "${OVERWRITE_ARGS[@]}"
      ;;
    package)
      "${PIXAL_PYTHON}" "${PIPELINE_DIR}/06_package_submission.py" \
        --data-root "${DATA_ROOT}" \
        --work-root "${WORK_ROOT}" \
        --submission-root "${SUBMISSION_ROOT}" \
        --zip-path "${SUBMISSION_ZIP}" \
        --expected-count "${EXPECTED_COUNT}" \
        --team-id "${TEAM_ID}" \
        --team-name "${TEAM_NAME}" \
        --round "${ROUND}"
      ;;
    *)
      echo "未知阶段: ${stage}" >&2
      echo "可用阶段: plans, plans-new, recover-plans, references, glbs, postprocess, render, package, all" >&2
      return 2
      ;;
  esac
}

STAGE="${1:-all}"
if [[ "${STAGE}" == "all" ]]; then
  for part in plans-new references glbs postprocess render package; do
    echo "[pipeline] 开始阶段: ${part}"
    run_stage "${part}"
  done
else
  run_stage "${STAGE}"
fi
