#!/usr/bin/env python3
"""Shared filesystem helpers for the minimal text-to-3D pipeline."""

from __future__ import annotations

import json
import hashlib
import os
import re
import tempfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PIPELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(PIPELINE_ROOT / "data")))
DEFAULT_WORK_ROOT = Path(os.environ.get("WORK_ROOT", str(PIPELINE_ROOT / "work")))


class PipelineError(RuntimeError):
    """Raised when an input or stage output violates the pipeline contract."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"无法读取 JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"JSON 顶层必须是对象: {path}")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def discover_tasks(data_root: Path, expected_count: int | None = 40) -> list[dict[str, Any]]:
    paths = sorted(data_root.glob("*/*/prompt.json"))
    if expected_count is not None and len(paths) != expected_count:
        raise PipelineError(
            f"期望发现 {expected_count} 道题，实际在 {data_root} 发现 {len(paths)} 道"
        )

    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    required = {"prompt_id", "category", "prompt", "constraints"}
    for path in paths:
        task = load_json(path)
        missing = sorted(required - task.keys())
        if missing:
            raise PipelineError(f"{path} 缺少字段: {', '.join(missing)}")
        prompt_id = task["prompt_id"]
        if not isinstance(prompt_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", prompt_id):
            raise PipelineError(f"非法 prompt_id: {prompt_id!r} ({path})")
        if prompt_id in seen:
            raise PipelineError(f"重复 prompt_id: {prompt_id}")
        if not isinstance(task["constraints"], dict):
            raise PipelineError(f"constraints 必须是对象: {path}")
        seen.add(prompt_id)
        task["_source_path"] = str(path)
        tasks.append(task)
    return sorted(tasks, key=lambda item: item["prompt_id"])


def task_dir(work_root: Path, prompt_id: str) -> Path:
    return work_root / prompt_id


def stable_seed(prompt_id: str) -> int:
    """Return a deterministic positive 31-bit seed."""
    return zlib.crc32(prompt_id.encode("utf-8")) & 0x7FFFFFFF


def stable_json_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for JSON-serializable pipeline inputs."""
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def update_status(work_root: Path, prompt_id: str, stage: str, **fields: Any) -> None:
    path = task_dir(work_root, prompt_id) / "status.json"
    current: dict[str, Any] = {}
    if path.exists():
        current = load_json(path)
    stages = current.setdefault("stages", {})
    stages[stage] = {"updated_at": utc_now_iso(), **fields}
    current["prompt_id"] = prompt_id
    atomic_write_json(path, current)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the largest decodable JSON object from an LLM response.

    A response may contain nested objects (for example ``reference_intent``).
    Scanning for every opening brace therefore finds both the complete answer
    and its inner objects.  The complete top-level answer has the largest
    decoded span, while selecting the last match incorrectly returns the last
    nested object.
    """
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append((end, match.start(), value))
    if not candidates:
        raise PipelineError("模型回答中没有可解析的 JSON 对象")
    # max() also prefers the later object when two candidates have equal span,
    # preserving the useful old behavior for responses containing an example
    # followed by the actual answer.
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def validate_plan(plan: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    prompt_id = task["prompt_id"]
    required_strings = (
        "t2i_prompt_zh",
        "negative_prompt",
        "view",
        "component_mode",
        "size_reason",
    )
    for key in required_strings:
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            raise PipelineError(f"{prompt_id} 的 plan 缺少非空字符串字段 {key}")

    target = plan.get("target_longest_mm")
    if not isinstance(target, (int, float)) or isinstance(target, bool):
        raise PipelineError(f"{prompt_id} 的 target_longest_mm 必须是数字")
    target = float(target)
    if not 10.0 <= target <= 300.0:
        raise PipelineError(
            f"{prompt_id} 的 target_longest_mm={target} 超出允许范围 [10, 300]"
        )

    raw_semantic = plan.get("semantic_identity")
    raw_geometry = plan.get("geometry_spec")
    has_semantic_schema = raw_semantic is not None or raw_geometry is not None
    semantic_identity: dict[str, Any] | None = None
    geometry_spec: dict[str, Any] | None = None

    def required_nested_text(container: dict[str, Any], group: str, key: str) -> str:
        value = container.get(key)
        if not isinstance(value, str) or not value.strip():
            raise PipelineError(f"{prompt_id} 的 {group}.{key} 必须是非空字符串")
        return value.strip()

    def required_nested_list(
        container: dict[str, Any], group: str, key: str
    ) -> list[str]:
        if key not in container or not isinstance(container[key], list):
            raise PipelineError(f"{prompt_id} 的 {group}.{key} 必须是字符串数组")
        values = [
            item.strip()
            for item in container[key]
            if isinstance(item, str) and item.strip()
        ]
        if len(values) != len(container[key]):
            raise PipelineError(f"{prompt_id} 的 {group}.{key} 包含空值或非字符串")
        return values

    if has_semantic_schema:
        if not isinstance(raw_semantic, dict) or not isinstance(raw_geometry, dict):
            raise PipelineError(
                f"{prompt_id} 的 semantic_identity 和 geometry_spec 必须同时为对象"
            )
        semantic_identity = {
            "artifact_head_noun": required_nested_text(
                raw_semantic, "semantic_identity", "artifact_head_noun"
            ),
            "literal_identity": required_nested_text(
                raw_semantic, "semantic_identity", "literal_identity"
            ),
            "shape_metaphors": required_nested_list(
                raw_semantic, "semantic_identity", "shape_metaphors"
            ),
            "usage_objects": required_nested_list(
                raw_semantic, "semantic_identity", "usage_objects"
            ),
            "forbidden_literalizations": required_nested_list(
                raw_semantic, "semantic_identity", "forbidden_literalizations"
            ),
        }
        geometry_spec = {
            "primary_volumes": required_nested_list(
                raw_geometry, "geometry_spec", "primary_volumes"
            ),
            "functional_surfaces": required_nested_list(
                raw_geometry, "geometry_spec", "functional_surfaces"
            ),
            "assembly_topology": required_nested_list(
                raw_geometry, "geometry_spec", "assembly_topology"
            ),
            "count_constraints": required_nested_list(
                raw_geometry, "geometry_spec", "count_constraints"
            ),
            "thickness_and_support": required_nested_list(
                raw_geometry, "geometry_spec", "thickness_and_support"
            ),
            "must_be_visible": required_nested_list(
                raw_geometry, "geometry_spec", "must_be_visible"
            ),
            "presentation_state": required_nested_text(
                raw_geometry, "geometry_spec", "presentation_state"
            ),
        }

    raw_intent = plan.get("reference_intent")
    has_explicit_reference_intent = isinstance(raw_intent, dict)
    if raw_intent is None:
        raw_intent = {}
    if not isinstance(raw_intent, dict):
        raise PipelineError(f"{prompt_id} 的 reference_intent 必须是对象")

    def optional_text(key: str, fallback: str) -> str:
        value = raw_intent.get(key)
        return value.strip() if isinstance(value, str) and value.strip() else fallback

    def optional_text_list(key: str) -> list[str]:
        value = raw_intent.get(key, [])
        if not isinstance(value, list):
            raise PipelineError(f"{prompt_id} 的 reference_intent.{key} 必须是数组")
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]

    if semantic_identity is not None and geometry_spec is not None:
        default_target = semantic_identity["literal_identity"]
        default_must_show = (
            geometry_spec["primary_volumes"]
            + geometry_spec["functional_surfaces"]
            + geometry_spec["assembly_topology"]
            + geometry_spec["count_constraints"]
            + geometry_spec["must_be_visible"]
        )
        default_must_not_show = (
            semantic_identity["usage_objects"]
            + semantic_identity["forbidden_literalizations"]
        )
        default_printability = geometry_spec["thickness_and_support"]
        default_state = geometry_spec["presentation_state"]
    else:
        default_target = task["prompt"]
        default_must_show = []
        default_must_not_show = []
        default_printability = []
        default_state = "完整装配状态"

    normalized = {
        "plan_schema_version": (
            3
            if semantic_identity is not None
            else 2
            if has_explicit_reference_intent
            else 1
        ),
        "prompt_id": prompt_id,
        "category": task["category"],
        "source_prompt": task["prompt"],
        "prompt_intent": task.get("prompt_intent", "3D打印物体"),
        "constraints": task["constraints"],
        "t2i_prompt_zh": plan["t2i_prompt_zh"].strip(),
        "negative_prompt": plan["negative_prompt"].strip(),
        "view": plan["view"].strip(),
        "component_mode": plan["component_mode"].strip(),
        "target_longest_mm": target,
        "size_reason": plan["size_reason"].strip(),
        # Older plan.json files did not contain reference_intent.  Keeping
        # conservative fallbacks makes the improved image stage usable without
        # forcing all existing plans to be regenerated first.
        "reference_intent": {
            "target_object": optional_text("target_object", default_target),
            "must_show": optional_text_list("must_show") or default_must_show,
            "must_not_show": optional_text_list("must_not_show")
            or default_must_not_show,
            "printability": optional_text_list("printability")
            or default_printability,
            "presentation_state": optional_text("presentation_state", default_state),
        },
        "seed": stable_seed(prompt_id),
    }
    if semantic_identity is not None and geometry_spec is not None:
        normalized["semantic_identity"] = semantic_identity
        normalized["geometry_spec"] = geometry_spec
    return normalized
