#!/usr/bin/env python3
"""Convert the generic JSONL interface to the pipeline's task layout."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def prepare(input_path: Path, data_root: Path) -> int:
    if not input_path.is_file():
        raise ValueError(f"输入文件不存在: {input_path}")
    tasks: list[tuple[str, dict]] = []
    seen: set[str] = set()
    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                task = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行 JSON 无效: {exc}") from exc
            if not isinstance(task, dict):
                raise ValueError(f"第 {line_number} 行必须是 JSON 对象")
            prompt_id = task.get("prompt_id")
            prompt = task.get("prompt")
            if not isinstance(prompt_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", prompt_id):
                raise ValueError(f"第 {line_number} 行 prompt_id 无效: {prompt_id!r}")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"第 {line_number} 行 prompt 不能为空")
            if prompt_id in seen:
                raise ValueError(f"第 {line_number} 行 prompt_id 重复: {prompt_id}")
            if "category" in task and not isinstance(task["category"], str):
                raise ValueError(f"第 {line_number} 行 category 必须是字符串")
            if "constraints" in task and not isinstance(task["constraints"], dict):
                raise ValueError(f"第 {line_number} 行 constraints 必须是对象")
            # Preserve every supplied field; the two defaults serve the existing
            # planner's task contract when the JSONL input has only ID/text.
            task.setdefault("category", "unknown")
            task.setdefault("constraints", {})
            tasks.append((prompt_id, task))
            seen.add(prompt_id)
    if not tasks:
        raise ValueError("输入文件没有题目")
    data_root.mkdir(parents=True, exist_ok=True)
    for prompt_id, task in tasks:
        target = data_root / "input" / prompt_id / "prompt.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(tasks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("data_root", type=Path)
    args = parser.parse_args()
    try:
        print(prepare(args.input, args.data_root))
    except (OSError, ValueError) as exc:
        print(f"输入校验失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
