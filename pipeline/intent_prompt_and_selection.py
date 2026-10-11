#!/usr/bin/env python3
"""Intent-aware reference prompting and lightweight candidate measurements.

This module is deliberately model-independent: both Qwen-Image generation and
Qwen3.8 visual selection use the exact same normalized intent contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common import stable_json_hash


PROMPT_POLICY_VERSION = "intent_reference_v2_semantic_geometry"

UNIFIED_VISUAL_SUFFIX = (
    "只显示待打印的目标物体本身，不展示它正在被使用；单个完整目标居中且全部入镜，四周留有清楚余量；"
    "全景深，整个物体从前到后都清晰锐利，不得虚化或散景；统一单色哑光浅灰色工程塑料原型，"
    "用明暗清楚表达几何起伏；纯白色无缝背景，柔和均匀的棚拍光和轻微接触阴影。"
    "画面效果类似干净的电商商品主图，但无品牌、无标签、无水印、无装饰文字。"
)

PRINT_INTENT_SUFFIX = (
    "这是供单图重建后进行FDM 3D打印的结构参考图：结构优先，形体简单、清楚、结实；"
    "减少无功能的细碎装饰、薄片、尖刺和深度无法判断的纹理；连接关系、实体厚度、开口、孔槽、"
    "接触地面的底座必须明确可读，不能依靠阴影或颜色假装几何。"
)

GENERIC_CONTEXT_OBJECTS = (
    "人物，手，身体，桌面场景，房间，墙面，窗户，植物，包装盒，展示台，尺寸线，标注箭头，"
    "说明文字，品牌，水印，多视图拼图，爆炸图，重复产品，悬浮零件，支撑材料，"
    "正在配套使用的卡片、书本、纸张、线缆、笔、扫把杆、垃圾袋、抹布、电池、螺丝和工具"
)

CANDIDATE_FOCUS = (
    "轮廓与整体比例优先，关键结构同时清楚可见。",
    "孔、槽、开口、连接和组件边界优先，但仍保持完整商品主图构图。",
    "可打印性与稳定落地关系优先，避免复杂装饰和脆弱细节。",
    "功能几何的数量与相对位置优先，保持形体简洁。",
)


def _clean_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def normalized_reference_intent(plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize new and legacy plans to one reference-image intent schema."""
    raw = plan.get("reference_intent")
    if not isinstance(raw, dict):
        raw = {}
    target = raw.get("target_object")
    if not isinstance(target, str) or not target.strip():
        target = plan.get("source_prompt") or plan.get("t2i_prompt_zh") or "待打印物体"
    state = raw.get("presentation_state")
    if not isinstance(state, str) or not state.strip():
        state = "完整装配状态"
    return {
        "target_object": target.strip(),
        "must_show": _clean_text_list(raw.get("must_show")),
        "must_not_show": _clean_text_list(raw.get("must_not_show")),
        "printability": _clean_text_list(raw.get("printability")),
        "presentation_state": state.strip(),
    }


def normalized_semantic_identity(plan: dict[str, Any]) -> dict[str, Any] | None:
    raw = plan.get("semantic_identity")
    if not isinstance(raw, dict):
        return None
    head_noun = raw.get("artifact_head_noun")
    literal_identity = raw.get("literal_identity")
    if not isinstance(head_noun, str) or not head_noun.strip():
        return None
    if not isinstance(literal_identity, str) or not literal_identity.strip():
        return None
    return {
        "artifact_head_noun": head_noun.strip(),
        "literal_identity": literal_identity.strip(),
        "shape_metaphors": _clean_text_list(raw.get("shape_metaphors")),
        "usage_objects": _clean_text_list(raw.get("usage_objects")),
        "forbidden_literalizations": _clean_text_list(
            raw.get("forbidden_literalizations")
        ),
    }


def normalized_geometry_spec(plan: dict[str, Any]) -> dict[str, Any] | None:
    raw = plan.get("geometry_spec")
    if not isinstance(raw, dict):
        return None
    keys = (
        "primary_volumes",
        "functional_surfaces",
        "assembly_topology",
        "count_constraints",
        "thickness_and_support",
        "must_be_visible",
    )
    result = {key: _clean_text_list(raw.get(key)) for key in keys}
    state = raw.get("presentation_state")
    result["presentation_state"] = (
        state.strip() if isinstance(state, str) and state.strip() else "完整装配状态"
    )
    return result


def constraint_instructions(plan: dict[str, Any]) -> list[str]:
    """Translate every relevant plan constraint into explicit visual language."""
    constraints = plan.get("constraints")
    if not isinstance(constraints, dict):
        constraints = {}
    rules: list[str] = []
    component_mode = str(plan.get("component_mode", "")).strip()
    if component_mode == "single_connected" or constraints.get("single_component") is True:
        rules.append("单一连续实体，所有局部均有真实材料相连，不得画成多个分离件")
    elif component_mode == "assembled_multi_part" or constraints.get("single_component") is False:
        rules.append("多组件以完整装配状态呈现，组件接缝和连接方式清楚，但禁止爆炸图和分离摆放")
    if constraints.get("no_floating_parts") is True:
        rules.append("任何局部都不得悬空或孤立，所有构件具有明确实体连接或装配支承")
    if constraints.get("stable_base") is True:
        rules.append("正常打印朝向，最低底面平整、水平并接触地面，画面能看出稳定放置")
    if constraints.get("watertight_required") is True:
        rules.append("表面是连续封闭的可制造实体，仅保留题目明确要求的孔槽和开口")
    if constraints.get("size_constrained") is True:
        rules.append("严格保留题目要求的数量、相对比例与尺寸关系，不添加会混淆外廓的附件")
    return rules


def compose_reference_prompt(plan: dict[str, Any], candidate_index: int = 0) -> str:
    """Build the final Qwen-Image prompt from every stage-one intent field."""
    intent = normalized_reference_intent(plan)
    semantic = normalized_semantic_identity(plan)
    geometry = normalized_geometry_spec(plan)
    must_show = intent["must_show"] or ["题目指定的全部功能几何、数量和连接关系"]
    printability = intent["printability"] or ["结构简单清楚", "厚度可信", "适合FDM打印"]
    constraints = constraint_instructions(plan) or ["保持连续、稳定、可制造的实体结构"]
    must_not_show = intent["must_not_show"] or ["任何仅用于说明用途的环境、配件或被承载物"]
    focus = CANDIDATE_FOCUS[candidate_index % len(CANDIDATE_FOCUS)]
    sections: list[str] = []
    if semantic is not None:
        sections.extend(
            [
                f"【人工制品中心名词，最高优先级】{semantic['artifact_head_noun']}",
                f"【物体实际身份】这是人工制造的{semantic['literal_identity']}",
            ]
        )
        if semantic["shape_metaphors"]:
            sections.append(
                "【造型隐喻，只修饰指定部位】"
                + "；".join(semantic["shape_metaphors"])
                + f"；整体始终必须识别为{semantic['artifact_head_noun']}"
            )
        if semantic["forbidden_literalizations"]:
            sections.append(
                "【禁止把造型字面化】不得生成"
                + "、".join(semantic["forbidden_literalizations"])
            )
    else:
        sections.append(f"【目标物体】{intent['target_object']}")
    if geometry is not None:
        geometry_sections = (
            ("主体实体体块", geometry["primary_volumes"]),
            ("功能表面", geometry["functional_surfaces"]),
            ("数量与装配拓扑", geometry["assembly_topology"] + geometry["count_constraints"]),
            ("实体厚度与支承", geometry["thickness_and_support"]),
            ("视角中必须可见", geometry["must_be_visible"]),
        )
        sections.extend(
            f"【{title}】" + "；".join(items)
            for title, items in geometry_sections
            if items
        )
    sections.extend([
        f"【任务背景意图】{str(plan.get('prompt_intent', '生成可实际3D打印的物体')).strip()}；最终产物需要单图重建并实际打印",
        f"【原题几何】{str(plan.get('t2i_prompt_zh', '')).strip()}",
        f"【相机视角，必须执行】{str(plan.get('view', '')).strip()}",
        f"【组件呈现】{str(plan.get('component_mode', '')).strip()}；{intent['presentation_state']}",
        "【必须清楚看见】" + "；".join(must_show),
        "【禁止出现在画面】" + "；".join(must_not_show),
        "【题目约束的视觉化】" + "；".join(constraints),
        "【3D打印意图】" + "；".join(printability),
        f"【本候选侧重】{focus}",
        PRINT_INTENT_SUFFIX,
        UNIFIED_VISUAL_SUFFIX,
    ])
    return "\n".join(section for section in sections if section.strip())


def compose_negative_prompt(plan: dict[str, Any]) -> str:
    intent = normalized_reference_intent(plan)
    semantic = normalized_semantic_identity(plan)
    dynamic = "，".join(intent["must_not_show"])
    parts = [str(plan.get("negative_prompt", "")).strip(), GENERIC_CONTEXT_OBJECTS]
    if dynamic:
        parts.append("题目特定禁用物体：" + dynamic)
    if semantic is not None:
        if semantic["usage_objects"]:
            parts.append("不得展示使用对象：" + "，".join(semantic["usage_objects"]))
        if semantic["forbidden_literalizations"]:
            parts.append(
                "不得把造型生成成真实物体："
                + "，".join(semantic["forbidden_literalizations"])
            )
    parts.append(
        "使用场景，道具，彩色材质，多种材质，透明材质，玻璃，金属镜面，景深虚化，散景，"
        "强烈投影，纯正面图，纯顶视图，裁切，出框，过小主体，复杂纹理，概念草图，线稿"
    )
    return "，".join(part for part in parts if part)


def candidate_seed(base_seed: int, candidate_index: int) -> int:
    """Deterministic, separated 31-bit seeds for multiple candidates."""
    return (int(base_seed) + candidate_index * 1_000_003) & 0x7FFFFFFF


def generation_signature(
    plan: dict[str, Any],
    *,
    candidate_count: int,
    width: int,
    height: int,
    steps: int,
    true_cfg_scale: float,
    model: str,
) -> str:
    return stable_json_hash(
        {
            "policy": PROMPT_POLICY_VERSION,
            "plan": plan,
            "candidate_count": candidate_count,
            "width": width,
            "height": height,
            "steps": steps,
            "true_cfg_scale": true_cfg_scale,
            "model": model,
        }
    )


def measure_candidate(path: Path) -> dict[str, Any]:
    """Measure framing/background/sharpness for logging and selector fallback."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width, _ = rgb.shape
    band = max(2, int(min(height, width) * 0.035))
    border = np.concatenate(
        (
            rgb[:band].reshape(-1, 3),
            rgb[-band:].reshape(-1, 3),
            rgb[:, :band].reshape(-1, 3),
            rgb[:, -band:].reshape(-1, 3),
        ),
        axis=0,
    )
    background = np.median(border, axis=0)
    border_distance = np.linalg.norm(border - background, axis=1)
    threshold = max(18.0, float(np.percentile(border_distance, 95)) + 8.0)
    distance = np.linalg.norm(rgb - background, axis=2)
    mask = distance > threshold
    occupancy = float(mask.mean())
    if mask.any():
        ys, xs = np.nonzero(mask)
        margins = {
            "left": float(xs.min() / width),
            "right": float((width - 1 - xs.max()) / width),
            "top": float(ys.min() / height),
            "bottom": float((height - 1 - ys.max()) / height),
        }
    else:
        margins = {key: 0.0 for key in ("left", "right", "top", "bottom")}
    gray = rgb.mean(axis=2)
    gradient = (
        float(np.abs(np.diff(gray, axis=0)).mean())
        + float(np.abs(np.diff(gray, axis=1)).mean())
    ) / 2.0
    border_std = float(border.std())
    occupancy_score = max(0.0, 1.0 - abs(occupancy - 0.34) / 0.34)
    margin_score = min(1.0, min(margins.values()) / 0.045) if mask.any() else 0.0
    background_score = max(0.0, 1.0 - border_std / 42.0)
    sharpness_score = min(1.0, gradient / 9.0)
    heuristic_score = 100.0 * (
        0.38 * occupancy_score
        + 0.25 * margin_score
        + 0.22 * background_score
        + 0.15 * sharpness_score
    )
    return {
        "width": width,
        "height": height,
        "foreground_occupancy": round(occupancy, 5),
        "margins": {key: round(value, 5) for key, value in margins.items()},
        "border_std": round(border_std, 4),
        "mean_gradient": round(gradient, 4),
        "heuristic_score": round(heuristic_score, 3),
    }
