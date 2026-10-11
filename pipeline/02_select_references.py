#!/usr/bin/env python3
"""Use Qwen3.8 vision intent matching to select one reference candidate per task."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_WORK_ROOT,
    PipelineError,
    atomic_write_json,
    discover_tasks,
    extract_json_object,
    load_json,
    stable_json_hash,
    task_dir,
    update_status,
)
from intent_prompt_and_selection import (
    constraint_instructions,
    normalized_geometry_spec,
    normalized_reference_intent,
    normalized_semantic_identity,
)


SELECTOR_POLICY_VERSION = "intent_visual_selector_v1"

SELECTOR_SYSTEM_PROMPT = """你是3D打印单图重建管线的参考图筛选器。你会看到一道题和若干按编号排列的候选图。
目标不是选择最漂亮的图，而是选择最适合后续图转3D、最忠实题意、最容易打印的图。

依次执行以下规则：
1. 硬性否决：目标物体错误；把造型隐喻字面化成真实叶子、动物、植物、桥梁等错误主体；关键数量/文字/孔槽错误；缺少题目要求的结构；出现使用场景物体或配件；主体被裁切；多物体/拼图/爆炸图；结构悬浮或明显不可制造。
2. 题意与关键结构 40 分：物体身份、数量、功能几何、连接和题目特定约束。
3. 单图重建适配 25 分：三维厚度可判断，关键正面、侧面、顶面同时可读，遮挡少，轮廓完整。
4. 目标隔离与构图 20 分：只显示目标本体，纯背景，主体占画面适中且完整入镜。
5. 3D打印清晰度 10 分：结构简单结实、连续、稳定，没有无意义薄片和碎细节。
6. 成像质量 5 分：全景深、单色哑光、光照均匀、背景干净。

只输出一个合法 JSON 对象，不要输出 Markdown 或额外解释：
{
  "selected_index": 0,
  "candidate_scores": [
    {"index": 0, "total": 0, "hard_failures": ["若无则为空数组"], "summary": "简短判断"}
  ],
  "reason": "为何入选，必须提到关键题意和淘汰原因"
}
selected_index 必须是实际候选编号。若所有候选都有缺陷，仍选硬伤最少的一张，并在 reason 中说明。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--expected-count", type=int, default=40)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--selector", choices=("vlm", "heuristic"), default="vlm")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selection_signature(manifest: dict[str, Any], args: argparse.Namespace) -> str:
    return stable_json_hash(
        {
            "policy": SELECTOR_POLICY_VERSION,
            "generation_signature": manifest.get("generation_signature"),
            "selector": args.selector,
            "model": args.model if args.selector == "vlm" else None,
        }
    )


def selection_is_current(output_dir: Path, signature: str) -> bool:
    selection_path = output_dir / "reference_candidates" / "selection.json"
    reference_path = output_dir / "reference.png"
    if not selection_path.is_file() or not reference_path.is_file():
        return False
    try:
        return load_json(selection_path).get("selection_signature") == signature
    except PipelineError:
        return False


def heuristic_choice(candidates: list[dict[str, Any]]) -> int:
    def score(item: dict[str, Any]) -> float:
        metrics = item.get("metrics")
        if not isinstance(metrics, dict):
            return float("-inf")
        value = metrics.get("heuristic_score")
        return float(value) if isinstance(value, (int, float)) else float("-inf")

    return int(max(candidates, key=score)["index"])


def validate_model_choice(response: dict[str, Any], valid_indices: set[int]) -> int:
    selected = response.get("selected_index")
    if not isinstance(selected, int) or isinstance(selected, bool):
        raise PipelineError("筛选模型的 selected_index 不是整数")
    if selected not in valid_indices:
        raise PipelineError(
            f"筛选模型返回不存在的候选 {selected}，可选编号为 {sorted(valid_indices)}"
        )
    return selected


def build_selector_messages(
    task: dict[str, Any], plan: dict[str, Any], output_dir: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    intent = normalized_reference_intent(plan)
    semantic = normalized_semantic_identity(plan)
    geometry = normalized_geometry_spec(plan)
    compact_task = {
        "prompt_id": task["prompt_id"],
        "category": task["category"],
        "原始题目": task["prompt"],
        "题目意图": task.get("prompt_intent", plan.get("prompt_intent", "3D打印")),
        "目标本体": intent["target_object"],
        "必须看见": intent["must_show"],
        "禁止场景物体": intent["must_not_show"],
        "打印要求": intent["printability"],
        "呈现状态": intent["presentation_state"],
        "相机视角": plan.get("view"),
        "组件模式": plan.get("component_mode"),
        "约束视觉化": constraint_instructions(plan),
        "语义消歧": semantic,
        "可观察几何与装配拓扑": geometry,
    }
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "请按系统评分规则筛选。题意如下：\n"
            + json.dumps(compact_task, ensure_ascii=False, indent=2)
            + "\n下面图片按候选编号依次给出。",
        }
    ]
    for candidate in manifest["candidates"]:
        candidate_path = output_dir / candidate["relative_path"]
        content.extend(
            (
                {
                    "type": "text",
                    "text": (
                        f"候选 {candidate['index']}；自动构图指标："
                        + json.dumps(candidate.get("metrics", {}), ensure_ascii=False)
                    ),
                },
                {"type": "image", "image": str(candidate_path.resolve())},
            )
        )
    content.append({"type": "text", "text": "现在只输出规定的 JSON。"})
    return [
        {"role": "system", "content": [{"type": "text", "text": SELECTOR_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    tasks = discover_tasks(args.data_root, args.expected_count)
    pending: list[tuple[dict, dict, dict, str]] = []
    for task in tasks:
        output_dir = task_dir(args.work_root, task["prompt_id"])
        plan_path = output_dir / "plan.json"
        manifest_path = output_dir / "reference_candidates" / "manifest.json"
        if not plan_path.is_file():
            raise PipelineError(f"缺少 {plan_path}，请先运行 plan 阶段")
        if not manifest_path.is_file():
            raise PipelineError(f"缺少 {manifest_path}，请先生成候选参考图")
        plan = load_json(plan_path)
        manifest = load_json(manifest_path)
        candidates = manifest.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise PipelineError(f"{manifest_path} 至少需要两个候选")
        for candidate in candidates:
            if not isinstance(candidate, dict) or not (
                output_dir / candidate.get("relative_path", "")
            ).is_file():
                raise PipelineError(f"{manifest_path} 含缺失或非法候选文件")
        signature = selection_signature(manifest, args)
        if args.overwrite or not selection_is_current(output_dir, signature):
            pending.append((task, plan, manifest, signature))

    if not pending:
        print(f"全部 {len(tasks)} 张 reference.png 已由当前筛选策略选定。")
        return 0

    processor = None
    model = None
    torch = None
    if args.selector == "vlm":
        gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
        if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
            raise PipelineError("--gpus 必须指定两张不同的 GPU，例如 0,1")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))

        import torch as torch_module
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        torch = torch_module
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
        print(f"加载 {args.model} 进行多图题意筛选，待处理 {len(pending)}/{len(tasks)} 题")
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
    fallback_count = 0
    for index, (task, plan, manifest, signature) in enumerate(pending, start=1):
        prompt_id = task["prompt_id"]
        output_dir = task_dir(args.work_root, prompt_id)
        candidate_dir = output_dir / "reference_candidates"
        candidates = manifest["candidates"]
        valid_indices = {int(item["index"]) for item in candidates}
        started = time.monotonic()
        print(f"[{index}/{len(pending)}] 筛选 {prompt_id}/reference.png", flush=True)
        method = args.selector
        warning = None
        raw_response = ""
        parsed_response: dict[str, Any] = {}
        try:
            if args.selector == "vlm":
                assert processor is not None and model is not None and torch is not None
                try:
                    messages = build_selector_messages(task, plan, output_dir, manifest)
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
                    raw_response = processor.decode(new_tokens, skip_special_tokens=True)
                    parsed_response = extract_json_object(raw_response)
                    selected_index = validate_model_choice(parsed_response, valid_indices)
                except Exception as exc:
                    warning = f"VLM 筛选失败，已用构图指标回退: {exc}"
                    print(f"  警告: {warning}", file=sys.stderr, flush=True)
                    selected_index = heuristic_choice(candidates)
                    method = "heuristic_fallback"
                    fallback_count += 1
            else:
                selected_index = heuristic_choice(candidates)

            selected = next(
                item for item in candidates if int(item["index"]) == selected_index
            )
            source_path = output_dir / selected["relative_path"]
            atomic_copy(source_path, output_dir / "reference.png")
            if raw_response:
                (output_dir / "logs").mkdir(exist_ok=True)
                (output_dir / "logs" / "reference_selection_response.txt").write_text(
                    raw_response + "\n", encoding="utf-8"
                )
            selection_record = {
                "prompt_id": prompt_id,
                "policy_version": SELECTOR_POLICY_VERSION,
                "selection_signature": signature,
                "generation_signature": manifest.get("generation_signature"),
                "method": method,
                "selector_model": args.model if args.selector == "vlm" else None,
                "selected_index": selected_index,
                "selected_relative_path": selected["relative_path"],
                "selected_metrics": selected.get("metrics", {}),
                "model_judgement": parsed_response,
            }
            if warning:
                selection_record["warning"] = warning
            atomic_write_json(candidate_dir / "selection.json", selection_record)
            update_status(
                args.work_root,
                prompt_id,
                "reference",
                ok=True,
                seconds=round(time.monotonic() - started, 3),
                method=method,
                selected_index=selected_index,
                selection_signature=signature,
                warning=warning,
            )
        except Exception as exc:
            failures.append(prompt_id)
            update_status(
                args.work_root,
                prompt_id,
                "reference",
                ok=False,
                seconds=round(time.monotonic() - started, 3),
                error=str(exc),
            )
            print(f"  失败: {exc}", file=sys.stderr, flush=True)

    if failures:
        print(f"参考图筛选失败 {len(failures)} 题: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(
        f"参考图筛选完成，共选定 {len(pending)} 张 reference.png；"
        f"VLM 回退 {fallback_count} 题。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
