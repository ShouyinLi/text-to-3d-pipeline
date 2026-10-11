#!/usr/bin/env python3
"""Fail before GPU initialization if a required local model shard is missing."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def check_index_tree(root: Path) -> None:
    for index in root.rglob("*.safetensors.index.json"):
        data = json.loads(index.read_text(encoding="utf-8"))
        names = set(data.get("weight_map", {}).values())
        if not names:
            raise ValueError(f"空权重索引: {index}")
        for name in names:
            shard = index.parent / name
            if not shard.is_file() or shard.stat().st_size == 0:
                raise ValueError(f"缺少权重分片: {shard}")


def main() -> int:
    try:
        roots = {
            "Qwen3.8": Path(os.environ["QWEN38_MODEL"]),
            "Qwen-Image": Path(os.environ["QWEN_IMAGE_MODEL"]),
            "Pixal3D": Path(os.environ["PIXAL_MODEL"]),
            "DINOv3": Path(os.environ["DINO_MODEL"]),
            "RMBG-2.0": Path(os.environ["RMBG_MODEL"]),
        }
        for label, root in roots.items():
            if not root.is_dir():
                raise ValueError(f"{label} 目录不存在: {root}")
            check_index_tree(root)
        pixal_config = json.loads((roots["Pixal3D"] / "pipeline.json").read_text(encoding="utf-8"))
        for relative in pixal_config["args"]["models"].values():
            for extension in (".json", ".safetensors"):
                weight_file = roots["Pixal3D"] / (relative + extension)
                if not weight_file.is_file() or weight_file.stat().st_size == 0:
                    raise ValueError(f"Pixal3D 子模型缺失: {weight_file}")
        for label, filename in (
            ("DINOv3", "model.safetensors"),
            ("RMBG-2.0", "model.safetensors"),
        ):
            weight_file = roots[label] / filename
            if not weight_file.is_file() or weight_file.stat().st_size == 0:
                raise ValueError(f"{label} 权重缺失: {weight_file}")
        moge = Path(os.environ["MOGE_MODEL"])
        naf = Path(os.environ["TORCH_HOME"]) / "hub/checkpoints/naf_release.pth"
        for weight_file in (moge, naf):
            if not weight_file.is_file() or weight_file.stat().st_size == 0:
                raise ValueError(f"权重缺失: {weight_file}")
        print("离线权重文件与分片索引检查通过")
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"离线权重预检失败: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
