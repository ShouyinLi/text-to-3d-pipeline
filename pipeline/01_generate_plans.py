#!/usr/bin/env python3
"""Load Qwen3.8 once and generate plan.json for all benchmark prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_WORK_ROOT,
    PipelineError,
    atomic_write_json,
    discover_tasks,
    extract_json_object,
    load_json,
    task_dir,
    update_status,
    validate_plan,
)


SYSTEM_PROMPT = """你是文本转3D管线的规划器。你的回答会直接交给文生图模型和几何后处理程序。
只输出一个合法 JSON 对象，不要输出 Markdown、解释或思考过程。

目标：把一道中文3D打印题改写成适合单张图片重建的中文产品参考图提示词。必须保留原始物体、数量、孔槽、文字和结构约束，不得凭空改变功能。

参考图规则：
1. 单个完整产品居中且完全入镜，浅灰或白色纯背景，柔和均匀工作室光照。
2. 使用最能显示关键结构的四分之三视角和轻微俯视；孔、槽、开口、底座必须真实可见。
3. 禁止人物、手、尺寸箭头、说明文字、复杂场景、爆炸图、多视图拼图、悬浮零件、玻璃和镜面材质。
4. single_component=true 时明确所有部件连续相连；stable_base=true 时明确底面平整并接触地面。
5. 多组件物体统一显示装配完成状态，不使用爆炸图。
6. target_longest_mm 用于最终模型的均匀缩放。若题目给出整体外廓上限，取不超过该上限的安全值；不要把插槽宽度、孔径、卡片宽度等局部尺寸误当成整体尺寸。没有整体尺寸时，根据常见桌面FDM物体选择 80 到 120 毫米。
7. 理解题目的3D打印意图：target_object 只描述需要打印的目标本体；must_not_show 必须列出仅用于说明用途、但绝不能进入参考图的环境物体（例如手、书本、线缆、扫把杆、卡片、桌板）。
8. must_show 列出从单张图中必须清楚看到并可核对的几何特征，包括数量、孔槽、文字、开口、连接和稳定底面；printability 列出需要被图像明确表达的简单、结实、连续、免支撑等制造特性。

必须严格输出这些字段：
{
  "t2i_prompt_zh": "完整的正向提示词",
  "negative_prompt": "负向提示词",
  "view": "相机视角说明",
  "component_mode": "single_connected 或 assembled_multi_part",
  "target_longest_mm": 100,
  "size_reason": "选择该整体尺寸的简短依据",
  "reference_intent": {
    "target_object": "只包含待打印物体本体的简短名称与形态",
    "must_show": ["必须清楚可见的关键几何1", "关键几何2"],
    "must_not_show": ["不能出现在图中的使用场景物体1"],
    "printability": ["连续实体", "结构简单结实"],
    "presentation_state": "完整装配状态或题目要求的开启状态"
  }
}
"""

REQUIRED_PLAN_SCHEMA_VERSION = 2
PLAN_RESPONSE_FILENAME = "plan_response.txt"
DEFAULT_MAX_NEW_TOKENS = 768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--expected-count", type=int, default=40)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    parser.add_argument(
        "--recover-responses",
        action="store_true",
        help=f"不加载模型，从 logs/{PLAN_RESPONSE_FILENAME} 重新解析并写出 plan.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def recover_saved_responses(args: argparse.Namespace, tasks: list[dict]) -> int:
    """Recover plans from already generated model responses without any GPU work."""
    failures: list[str] = []
    recovered = 0
    upgraded = 0
    for task in tasks:
        prompt_id = task["prompt_id"]
        output_dir = task_dir(args.work_root, prompt_id)
        response_path = output_dir / "logs" / PLAN_RESPONSE_FILENAME
        if not response_path.is_file():
            failures.append(prompt_id)
            print(f"  缺少: {response_path}", file=sys.stderr)
            continue
        try:
            response = response_path.read_text(encoding="utf-8")
            plan = validate_plan(extract_json_object(response), task)
            atomic_write_json(output_dir / "plan.json", plan)
            if plan.get("plan_schema_version", 0) >= REQUIRED_PLAN_SCHEMA_VERSION:
                upgraded += 1
            update_status(
                args.work_root,
                prompt_id,
                "plan",
                ok=True,
                seconds=0.0,
                model=args.model,
                recovered_from=str(response_path),
            )
            recovered += 1
        except Exception as exc:
            failures.append(prompt_id)
            print(f"  {prompt_id} 恢复失败: {exc}", file=sys.stderr)
    if failures:
        print(f"响应恢复失败 {len(failures)} 题: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(
        f"已从现有模型响应恢复 {recovered} 个 plan.json，其中新版意图计划 {upgraded} 个、"
        f"旧版计划 {recovered - upgraded} 个；未加载 Qwen3.8。"
    )
    return 0


def plan_is_current(path: Path) -> bool:
    """A current plan has the explicit intent schema required by this version."""
    if not path.is_file():
        return False
    try:
        plan = load_json(path)
    except PipelineError:
        return False
    return plan.get("plan_schema_version", 0) >= REQUIRED_PLAN_SCHEMA_VERSION


def main() -> int:
    args = parse_args()
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise PipelineError("--gpus 必须指定两张不同的 GPU，例如 0,1")

    tasks = discover_tasks(args.data_root, args.expected_count)
    if args.recover_responses:
        return recover_saved_responses(args, tasks)
    pending = [
        task
        for task in tasks
        if args.overwrite
        or not plan_is_current(
            task_dir(args.work_root, task["prompt_id"]) / "plan.json"
        )
    ]
    if not pending:
        print(f"全部 {len(tasks)} 个 plan.json 已存在，无需加载模型。")
        return 0

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise PipelineError(f"需要恰好两张可见 GPU，当前为 {torch.cuda.device_count()} 张")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    max_memory: dict[int, str] = {}
    for device_id in range(2):
        free_bytes, _ = torch.cuda.mem_get_info(device_id)
        usable_gib = int(free_bytes / 1024**3) - 2
        if usable_gib <= 0:
            raise PipelineError(f"cuda:{device_id} 可用显存不足")
        max_memory[device_id] = f"{usable_gib}GiB"

    print(f"加载 {args.model}，待处理 {len(pending)}/{len(tasks)} 题")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="balanced",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    model.eval()

    placement = Counter(str(device) for device in model.hf_device_map.values())
    normalized_devices = {
        f"cuda:{device}" if str(device).isdigit() else str(device)
        for device in model.hf_device_map.values()
    }
    if not {"cuda:0", "cuda:1"}.issubset(normalized_devices):
        raise PipelineError(f"Qwen3.8 未分布到两张 GPU: {placement}")
    if "cpu" in normalized_devices or "disk" in normalized_devices:
        raise PipelineError(f"Qwen3.8 发生 CPU/磁盘卸载: {placement}")
    print(f"模型分片: {placement}")

    failures: list[str] = []
    for index, task in enumerate(pending, start=1):
        prompt_id = task["prompt_id"]
        output_dir = task_dir(args.work_root, prompt_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        request = {
            key: value
            for key, value in task.items()
            if not key.startswith("_")
        }
        user_text = "请为下面题目生成规划 JSON：\n" + json.dumps(
            request, ensure_ascii=False, indent=2
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]
        started = time.monotonic()
        print(f"[{index}/{len(pending)}] 生成 {prompt_id}/plan.json", flush=True)
        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                enable_thinking=False,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
            response = processor.decode(new_tokens, skip_special_tokens=True)
            (output_dir / "logs").mkdir(exist_ok=True)
            (output_dir / "logs" / PLAN_RESPONSE_FILENAME).write_text(
                response + "\n", encoding="utf-8"
            )
            plan = validate_plan(extract_json_object(response), task)
            if plan.get("plan_schema_version", 0) < REQUIRED_PLAN_SCHEMA_VERSION:
                raise PipelineError(
                    f"{prompt_id} 返回 plan schema v{plan.get('plan_schema_version', 0)}，"
                    f"当前生成器要求至少 v{REQUIRED_PLAN_SCHEMA_VERSION}"
                )
            atomic_write_json(output_dir / "plan.json", plan)
            update_status(
                args.work_root,
                prompt_id,
                "plan",
                ok=True,
                seconds=round(time.monotonic() - started, 3),
                model=args.model,
            )
        except Exception as exc:
            failures.append(prompt_id)
            update_status(
                args.work_root,
                prompt_id,
                "plan",
                ok=False,
                seconds=round(time.monotonic() - started, 3),
                error=str(exc),
            )
            print(f"  失败: {exc}", file=sys.stderr, flush=True)

    if failures:
        print(f"plan 阶段失败 {len(failures)} 题: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"plan 阶段完成，共 {len(tasks)} 题。进程退出后释放两张 GPU。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
