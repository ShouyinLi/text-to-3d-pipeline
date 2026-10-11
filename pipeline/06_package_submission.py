#!/usr/bin/env python3
"""Create metadata, process notes, manifest and submission.zip."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_WORK_ROOT,
    PipelineError,
    atomic_write_json,
    discover_tasks,
    load_json,
    task_dir,
)


PIPELINE_ROOT = Path(__file__).resolve().parent
DEFAULT_SUBMISSION_ROOT = PIPELINE_ROOT / "submission"
DEFAULT_ZIP_PATH = PIPELINE_ROOT / "submission.zip"
RENDER_NAMES = ("front.png", "side.png", "top.png", "perspective.png")


def validate_final_model(model_path: Path, target_mm: float) -> dict:
    """Require raw-index watertightness, then repeat checks after position weld."""
    try:
        import pymeshlab
    except ImportError as exc:
        raise PipelineError("最终模型检验需要 pymeshlab") from exc
    mesh_set = pymeshlab.MeshSet()
    mesh_set.load_new_mesh(str(model_path))
    vertices_before_weld = int(mesh_set.current_mesh().vertex_number())
    raw_measures = mesh_set.get_topological_measures()
    mesh_set.meshing_remove_duplicate_vertices()
    mesh_set.meshing_remove_duplicate_faces()
    mesh_set.meshing_remove_null_faces()
    mesh_set.meshing_remove_unreferenced_vertices()
    measures = mesh_set.get_topological_measures()
    mesh_set.compute_selection_by_self_intersections_per_face()
    mesh = mesh_set.current_mesh()
    self_intersections = int(np.count_nonzero(mesh.face_selection_array()))
    vertices = np.asarray(mesh.vertex_matrix(), dtype=np.float64)
    bounds = np.vstack((vertices.min(axis=0), vertices.max(axis=0)))
    extents = bounds[1] - bounds[0]
    valid = (
        bool(raw_measures["is_mesh_two_manifold"])
        and int(raw_measures["boundary_edges"]) == 0
        and int(raw_measures["non_two_manifold_edges"]) == 0
        and int(raw_measures["non_two_manifold_vertices"]) == 0
        and int(raw_measures["connected_components_number"]) == 1
        and bool(measures["is_mesh_two_manifold"])
        and int(measures["boundary_edges"]) == 0
        and int(measures["non_two_manifold_edges"]) == 0
        and int(measures["non_two_manifold_vertices"]) == 0
        and int(measures["connected_components_number"]) == 1
        and self_intersections == 0
        and abs(float(bounds[0, 2])) <= 1e-4
        and np.isclose(float(extents.max()), target_mm, rtol=1e-5, atol=1e-3)
    )
    stats = {
        "vertices_before_uv_seam_weld": vertices_before_weld,
        "vertices_after_uv_seam_weld": int(mesh.vertex_number()),
        "watertight_before_weld": int(raw_measures["boundary_edges"]) == 0,
        "two_manifold_before_weld": bool(raw_measures["is_mesh_two_manifold"]),
        "boundary_edges_before_weld": int(raw_measures["boundary_edges"]),
        "nonmanifold_edges_before_weld": int(
            raw_measures["non_two_manifold_edges"]
        ),
        "nonmanifold_vertices_before_weld": int(
            raw_measures["non_two_manifold_vertices"]
        ),
        "components_before_weld": int(raw_measures["connected_components_number"]),
        "watertight_after_weld": int(measures["boundary_edges"]) == 0,
        "two_manifold_after_weld": bool(measures["is_mesh_two_manifold"]),
        "nonmanifold_edges_after_weld": int(measures["non_two_manifold_edges"]),
        "nonmanifold_vertices_after_weld": int(
            measures["non_two_manifold_vertices"]
        ),
        "components_after_weld": int(measures["connected_components_number"]),
        "self_intersecting_faces_after_weld": self_intersections,
        "bounds_mm": bounds.tolist(),
        "extents_mm": extents.tolist(),
    }
    if not valid:
        raise PipelineError(f"{model_path} 最终模型制造检验失败: {stats}")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--submission-root", type=Path, default=DEFAULT_SUBMISSION_ROOT)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--expected-count", type=int, default=40)
    parser.add_argument("--team-id", default="team_001")
    parser.add_argument("--team-name", default="baseline_team")
    parser.add_argument("--round", default="preliminary")
    return parser.parse_args()


def stage_seconds(status: dict) -> float:
    total = 0.0
    for stage in status.get("stages", {}).values():
        if stage.get("ok"):
            total += float(stage.get("seconds", 0.0))
    return total


def make_process_markdown(task: dict, plan: dict, status: dict) -> str:
    total_minutes = stage_seconds(status) / 60.0
    raw_stage = status.get("stages", {}).get("raw_glb", {})
    post = status.get("stages", {}).get("postprocess", {})
    candidate_count = status.get("stages", {}).get("reference_candidates", {}).get("count", "unknown")
    return f"""题目编号：{task['prompt_id']}
使用模型：Qwen3.8-27B、Qwen-Image-2512、Pixal3D
主要改进：中文约束改写为单图重建参考图；多候选参考图筛选；保留语义朝上的制造定向；底面校正与切平；Voxel/SDF 水密严格流形单体化。
生成参数：Qwen-Image 候选数={candidate_count}；Pixal3D resolution={raw_stage.get('resolution', 'unknown')}；目标最长边={plan['target_longest_mm']} mm。
随机种子：{plan['seed']}
生成耗时：{total_minutes:.3f} 分钟
后处理流程：Scene 展开合并与小碎片清理；朝向模式 {post.get('orientation_mode', 'unknown')}，选择竖直轴 {post.get('selected_up_axis', 'unknown')}；底面拟合与毫米缩放；以 {post.get('sdf', {}).get('actual_voxel_mm', 'unknown')} mm 体素完成 SDF 实体化；按需执行有符号 SDF 流形清理；最终保持拓扑地压平底面、令 Zmin=0，并在焊接重合顶点后验证水密、非流形边/顶点、退化面、自相交和组件数量。
失败样例与修正：本题最终管线成功，无额外候选重试。
是否使用商业API：否
数据来源说明：使用公开预训练模型生成；输入由用户通过 prompts.jsonl 提供；未使用商业 API。
"""


def main() -> int:
    args = parse_args()
    tasks = discover_tasks(args.data_root, args.expected_count)
    args.submission_root.mkdir(parents=True, exist_ok=True)
    manifest_tasks: list[dict] = []
    archive_files: list[tuple[Path, str]] = []

    for task in tasks:
        prompt_id = task["prompt_id"]
        source_dir = task_dir(args.work_root, prompt_id)
        model_path = source_dir / "model.glb"
        plan_path = source_dir / "plan.json"
        status_path = source_dir / "status.json"
        missing = [path for path in (model_path, plan_path, status_path) if not path.exists()]
        missing += [source_dir / "renders" / name for name in RENDER_NAMES if not (source_dir / "renders" / name).exists()]
        if missing:
            raise PipelineError(
                f"{prompt_id} 缺少提交文件: {', '.join(str(path) for path in missing)}"
            )
        plan = load_json(plan_path)
        status = load_json(status_path)
        for required_stage in ("plan", "reference", "raw_glb", "postprocess", "render"):
            if not status.get("stages", {}).get(required_stage, {}).get("ok"):
                raise PipelineError(f"{prompt_id} 的阶段 {required_stage} 未成功")

        final_model_validation = validate_final_model(
            model_path, float(plan["target_longest_mm"])
        )

        destination = args.submission_root / prompt_id
        renders_destination = destination / "renders"
        renders_destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, destination / "model.glb")
        for name in RENDER_NAMES:
            shutil.copy2(source_dir / "renders" / name, renders_destination / name)

        total_minutes = stage_seconds(status) / 60.0
        metadata = {
            "prompt_id": prompt_id,
            "primary_model": "model.glb",
            "unit": "mm",
            "base_model": "Qwen3.8-27B + Qwen-Image-2512 + Pixal3D",
            "seed": int(plan["seed"]),
            "generation_time_minutes": round(total_minutes, 3),
            "postprocess": (
                "flatten and merge scene meshes, semantic Z-up orientation, "
                "base leveling and topology-preserving floor flattening, "
                "uniform millimeter scaling, unsigned SDF solidification, "
                "adaptive signed-SDF manifold cleanup, single-body enforcement, "
                "ground to Z=0 and raw-index plus welded strict topology validation"
            ),
            "final_textured_model_validation": final_model_validation,
            "commercial_api_used": False,
        }
        atomic_write_json(destination / "metadata.json", metadata)
        (destination / "process.md").write_text(
            make_process_markdown(task, plan, status), encoding="utf-8"
        )
        manifest_tasks.append(
            {"prompt_id": prompt_id, "path": prompt_id, "primary_model": "model.glb"}
        )
        for relative in (
            "model.glb",
            "metadata.json",
            "process.md",
            *(f"renders/{name}" for name in RENDER_NAMES),
        ):
            archive_files.append((destination / relative, f"{prompt_id}/{relative}"))

    china_tz = timezone(timedelta(hours=8))
    manifest = {
        "team_id": args.team_id,
        "team_name": args.team_name,
        "round": args.round,
        "created_at": datetime.now(china_tz).isoformat(timespec="seconds"),
        "tasks": manifest_tasks,
    }
    manifest_path = args.submission_root / "manifest.json"
    atomic_write_json(manifest_path, manifest)

    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.write(manifest_path, "manifest.json")
        for source, archive_name in archive_files:
            archive.write(source, archive_name)

    with zipfile.ZipFile(args.zip_path, "r") as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise PipelineError(f"ZIP CRC 检查失败: {bad_file}")
        expected_members = 1 + len(tasks) * 7
        if len(archive.namelist()) != expected_members:
            raise PipelineError(
                f"ZIP 文件数量错误: {len(archive.namelist())} != {expected_members}"
            )

    print(f"提交目录: {args.submission_root}")
    print(f"提交压缩包: {args.zip_path} ({args.zip_path.stat().st_size / 1024**2:.2f} MiB)")
    print(f"题目数量: {len(manifest_tasks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
