#!/usr/bin/env python3
"""Load Qwen-Image once and generate multiple intent-aware candidates per plan."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_WORK_ROOT,
    PipelineError,
    atomic_write_json,
    discover_tasks,
    load_json,
    task_dir,
    update_status,
)
from intent_prompt_and_selection import (
    PROMPT_POLICY_VERSION,
    candidate_seed,
    compose_negative_prompt,
    compose_reference_prompt,
    generation_signature,
    measure_candidate,
)


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(os.environ.get("QWEN_IMAGE_MODEL", str(BUNDLE_ROOT / "weights" / "Qwen-Image-2512")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--expected-count", type=int, default=40)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--width", type=int, default=1328)
    parser.add_argument("--height", type=int, default=1328)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--max-memory-per-gpu", default="42GiB")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def expected_signature(args: argparse.Namespace, plan: dict) -> str:
    return generation_signature(
        plan,
        candidate_count=args.candidates,
        width=args.width,
        height=args.height,
        steps=args.steps,
        true_cfg_scale=args.true_cfg_scale,
        model=str(args.model),
    )


def candidates_are_current(output_dir: Path, signature: str, count: int) -> bool:
    manifest_path = output_dir / "reference_candidates" / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = load_json(manifest_path)
    except PipelineError:
        return False
    if manifest.get("generation_signature") != signature:
        return False
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != count:
        return False
    return all((output_dir / item.get("relative_path", "")).is_file() for item in candidates)


def main() -> int:
    args = parse_args()
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise PipelineError("--gpus 必须指定两张不同的 GPU，例如 0,1")
    if args.candidates < 2 or args.candidates > 8:
        raise PipelineError("--candidates 必须在 [2, 8]，推荐 3")
    if not args.model.is_dir():
        raise PipelineError(f"Qwen-Image 模型目录不存在: {args.model}")

    tasks = discover_tasks(args.data_root, args.expected_count)
    pending: list[tuple[dict, dict, str]] = []
    for task in tasks:
        output_dir = task_dir(args.work_root, task["prompt_id"])
        plan_path = output_dir / "plan.json"
        if not plan_path.exists():
            raise PipelineError(f"缺少 {plan_path}，请先运行 plan 阶段")
        plan = load_json(plan_path)
        # Legacy plans predate prompt_intent; recover it from the benchmark
        # task in memory so the improved reference stage still understands the
        # manufacturing intent without rewriting the user's plan.json.
        plan.setdefault("prompt_intent", task.get("prompt_intent", "3D打印物体"))
        signature = expected_signature(args, plan)
        if args.overwrite or not candidates_are_current(
            output_dir, signature, args.candidates
        ):
            pending.append((task, plan, signature))

    if not pending:
        print(f"全部 {len(tasks)} 题的候选参考图已是当前版本，无需加载 Qwen-Image。")
        return 0

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))

    import torch
    from diffusers import QwenImagePipeline

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise PipelineError(f"需要恰好两张可见 GPU，当前为 {torch.cuda.device_count()} 张")

    total_images = len(pending) * args.candidates
    print(
        f"加载 {args.model}，待处理 {len(pending)}/{len(tasks)} 题，"
        f"每题 {args.candidates} 个候选，共 {total_images} 张"
    )
    pipe = QwenImagePipeline.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={0: args.max_memory_per_gpu, 1: args.max_memory_per_gpu},
    )
    print(f"模型分布: {pipe.hf_device_map}")

    failures: list[str] = []
    generated_images = 0
    for task_index, (task, plan, signature) in enumerate(pending, start=1):
        prompt_id = task["prompt_id"]
        output_dir = task_dir(args.work_root, prompt_id)
        candidate_dir = output_dir / "reference_candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        print(
            f"[{task_index}/{len(pending)}] 生成 {prompt_id} 的 {args.candidates} 个候选",
            flush=True,
        )
        try:
            negative_prompt = compose_negative_prompt(plan)
            records: list[dict] = []
            prompt_log: list[str] = []
            for candidate_index in range(args.candidates):
                prompt = compose_reference_prompt(plan, candidate_index)
                seed = candidate_seed(int(plan["seed"]), candidate_index)
                output_path = candidate_dir / f"candidate_{candidate_index:02d}.png"
                print(
                    f"  候选 {candidate_index + 1}/{args.candidates}，seed={seed}",
                    flush=True,
                )
                generator = torch.Generator(device="cuda:0").manual_seed(seed)
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=args.width,
                    height=args.height,
                    num_inference_steps=args.steps,
                    true_cfg_scale=args.true_cfg_scale,
                    generator=generator,
                ).images[0]
                if image.width <= 0 or image.height <= 0:
                    raise PipelineError("Qwen-Image 返回空图像")
                image.save(output_path)
                metrics = measure_candidate(output_path)
                records.append(
                    {
                        "index": candidate_index,
                        "relative_path": str(output_path.relative_to(output_dir)),
                        "seed": seed,
                        "focus_prompt": prompt,
                        "metrics": metrics,
                    }
                )
                prompt_log.append(
                    f"===== candidate_{candidate_index:02d} / seed {seed} =====\n"
                    f"{prompt}\n\nNEGATIVE:\n{negative_prompt}\n"
                )
                generated_images += 1

            (output_dir / "logs").mkdir(exist_ok=True)
            (output_dir / "logs" / "reference_prompts.txt").write_text(
                "\n".join(prompt_log), encoding="utf-8"
            )
            atomic_write_json(
                candidate_dir / "manifest.json",
                {
                    "prompt_id": prompt_id,
                    "policy_version": PROMPT_POLICY_VERSION,
                    "generation_signature": signature,
                    "model": str(args.model),
                    "width": args.width,
                    "height": args.height,
                    "steps": args.steps,
                    "true_cfg_scale": args.true_cfg_scale,
                    "negative_prompt": negative_prompt,
                    "candidates": records,
                },
            )
            update_status(
                args.work_root,
                prompt_id,
                "reference_candidates",
                ok=True,
                seconds=round(time.monotonic() - started, 3),
                model=str(args.model),
                count=args.candidates,
                generation_signature=signature,
            )
        except Exception as exc:
            failures.append(prompt_id)
            update_status(
                args.work_root,
                prompt_id,
                "reference_candidates",
                ok=False,
                seconds=round(time.monotonic() - started, 3),
                error=str(exc),
            )
            print(f"  失败: {exc}", file=sys.stderr, flush=True)

    if failures:
        print(
            f"候选参考图阶段失败 {len(failures)} 题: {', '.join(failures)}",
            file=sys.stderr,
        )
        return 1
    print(
        f"候选参考图阶段完成，本次生成 {generated_images} 张。"
        "Qwen-Image 进程退出后释放两张 GPU。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
