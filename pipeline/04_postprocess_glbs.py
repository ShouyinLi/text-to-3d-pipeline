#!/usr/bin/env python3
"""Turn Pixal3D raw GLBs into grounded, single-body printable GLBs.

The manufacturing pass deliberately rebuilds topology.  Original texture UVs
therefore cannot survive; their appearance is sampled to vertex colors on the
final SDF surface instead.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_WORK_ROOT,
    PipelineError,
    discover_tasks,
    load_json,
    task_dir,
    update_status,
)


AXIS_CANDIDATES = {
    "+Z": np.array([0.0, 0.0, 1.0]),
    "-Z": np.array([0.0, 0.0, -1.0]),
    "+X": np.array([1.0, 0.0, 0.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]),
    "-Y": np.array([0.0, -1.0, 0.0]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--expected-count", type=int, default=40)
    parser.add_argument(
        "--min-component-faces",
        type=int,
        default=50,
        help="SDF 前后删除小于此面数的明显碎片，默认50",
    )
    parser.add_argument(
        "--orientation-mode",
        choices=("semantic", "auto-print"),
        default="semantic",
        help=(
            "竖直方向策略：semantic 保留参考图/Pixal3D 的 +Z 语义方向（默认）；"
            "auto-print 才允许按六轴接触与悬垂评分改变朝向"
        ),
    )
    parser.add_argument(
        "--bottom-plane-max-tilt-deg",
        type=float,
        default=15.0,
        help="六轴定向后允许底面拟合纠正的最大倾角，默认15度",
    )
    parser.add_argument(
        "--base-cut-mm",
        type=float,
        default=0.4,
        help="在实体化前切除的底部厚度，默认0.4mm；0表示禁用",
    )
    parser.add_argument(
        "--voxel-mm",
        type=float,
        default=0.6,
        help="Voxel/SDF 重建的目标体素尺寸，默认0.6mm",
    )
    parser.add_argument(
        "--sdf-offset-mm",
        type=float,
        default=0.8,
        help="无符号距离场外扩半径；可桥接约其两倍宽度的微缝，默认0.8mm",
    )
    parser.add_argument(
        "--max-voxel-axis",
        type=int,
        default=384,
        help="单轴最大体素数；大模型会自动增大实际体素，默认384",
    )
    parser.add_argument(
        "--no-color-transfer",
        action="store_true",
        help="不从原纹理近邻采样顶点色，可降低内存占用",
    )
    parser.add_argument(
        "--prompt-id",
        action="append",
        default=[],
        help="只处理指定 prompt_id；可重复传入，默认处理全部任务",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.min_component_faces < 0:
        parser.error("--min-component-faces 必须非负")
    if not 0.0 <= args.bottom_plane_max_tilt_deg <= 45.0:
        parser.error("--bottom-plane-max-tilt-deg 必须在 [0, 45] 内")
    if args.base_cut_mm < 0:
        parser.error("--base-cut-mm 必须非负")
    if args.voxel_mm <= 0 or args.sdf_offset_mm <= 0:
        parser.error("--voxel-mm 和 --sdf-offset-mm 必须为正数")
    if args.max_voxel_axis < 32:
        parser.error("--max-voxel-axis 必须至少为32")
    return args


def _validate_source_mesh(mesh: trimesh.Trimesh, path: Path, name: str) -> None:
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise PipelineError(f"{path} 包含空网格 {name}")
    if not np.isfinite(mesh.vertices).all():
        raise PipelineError(f"{path} 的顶点包含 NaN 或 Inf: {name}")
    if mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
        raise PipelineError(f"{path} 的面索引越界: {name}")


def load_and_flatten(path: Path) -> list[trimesh.Trimesh]:
    """Load every scene node and bake its graph transform into a mesh copy."""
    try:
        scene = trimesh.load(str(path), force="scene", process=False)
    except Exception as exc:
        raise PipelineError(f"GLB 无法加载: {path}: {exc}") from exc
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(scene)

    geometries: list[trimesh.Trimesh] = []
    for node_name in list(scene.graph.nodes_geometry):
        transform, geometry_name = scene.graph[node_name]
        source = scene.geometry.get(geometry_name)
        if not isinstance(source, trimesh.Trimesh):
            continue
        mesh = source.copy()
        _validate_source_mesh(mesh, path, str(geometry_name))
        mesh.apply_transform(np.asarray(transform, dtype=np.float64))
        geometries.append(mesh)
    if not geometries:
        raise PipelineError(f"{path} 不包含非空三角网格")
    return geometries


def _vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """Convert texture/face/vertex visuals to one RGBA color per vertex."""
    try:
        direct = getattr(mesh.visual, "vertex_colors", None)
        if direct is not None:
            colors = np.asarray(direct, dtype=np.uint8)
        else:
            colors = np.asarray(mesh.visual.to_color().vertex_colors, dtype=np.uint8)
        if colors.shape == (len(mesh.vertices), 4):
            colors = colors.copy()
            # The manufactured body is opaque; zero-alpha texels at UV seams
            # must not make reconstructed faces disappear.
            colors[:, 3] = 255
            return colors
    except Exception:
        pass
    return np.tile(np.array([[200, 200, 200, 255]], dtype=np.uint8), (len(mesh.vertices), 1))


def merge_geometries(
    geometries: list[trimesh.Trimesh], *, preserve_colors: bool
) -> trimesh.Trimesh:
    """Merge scene meshes into one Trimesh without welding unrelated vertices."""
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    offset = 0
    for mesh in geometries:
        vertex_array = np.asarray(mesh.vertices, dtype=np.float64)
        vertices.append(vertex_array)
        faces.append(np.asarray(mesh.faces, dtype=np.int64) + offset)
        if preserve_colors:
            colors.append(_vertex_colors(mesh))
        offset += len(vertex_array)
    kwargs: dict[str, Any] = {}
    if preserve_colors:
        kwargs["vertex_colors"] = np.vstack(colors)
    merged = trimesh.Trimesh(
        vertices=np.vstack(vertices),
        faces=np.vstack(faces),
        process=False,
        validate=False,
        **kwargs,
    )
    if len(merged.vertices) == 0 or len(merged.faces) == 0:
        raise PipelineError("Scene 合并后得到空网格")
    return merged


def _load_pymeshlab():
    try:
        import pymeshlab
    except ImportError as exc:
        raise PipelineError("制造后处理需要 pymeshlab") from exc
    return pymeshlab


def _pymeshlab_mesh(mesh: trimesh.Trimesh, pymeshlab_module):
    return pymeshlab_module.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    )


def _from_pymeshlab(mesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertex_matrix(), dtype=np.float64).copy(),
        faces=np.asarray(mesh.face_matrix(), dtype=np.int64).copy(),
        process=False,
        validate=False,
    )


def remove_small_fragments(mesh: trimesh.Trimesh, min_faces: int) -> tuple[trimesh.Trimesh, dict]:
    """Weld duplicates and remove obvious disconnected face fragments."""
    pymeshlab = _load_pymeshlab()
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(_pymeshlab_mesh(mesh, pymeshlab), "merged_raw")
    before_vertices = int(mesh_set.current_mesh().vertex_number())
    before_faces = int(mesh_set.current_mesh().face_number())
    mesh_set.meshing_remove_duplicate_vertices()
    mesh_set.meshing_remove_duplicate_faces()
    mesh_set.meshing_remove_null_faces()
    mesh_set.meshing_remove_unreferenced_vertices()
    if min_faces > 0:
        mesh_set.meshing_remove_connected_component_by_face_number(
            mincomponentsize=int(min_faces), removeunref=True
        )
    cleaned = _from_pymeshlab(mesh_set.current_mesh())
    if len(cleaned.vertices) == 0 or len(cleaned.faces) == 0:
        raise PipelineError("删除小碎片后网格为空")
    return cleaned, {
        "before_vertices": before_vertices,
        "before_faces": before_faces,
        "after_vertices": int(len(cleaned.vertices)),
        "after_faces": int(len(cleaned.faces)),
        "removed_faces": before_faces - int(len(cleaned.faces)),
    }


def _surface_centroid(mesh: trimesh.Trimesh) -> np.ndarray:
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    valid = np.isfinite(areas) & (areas > 1e-16)
    if valid.any():
        return np.average(centers[valid], axis=0, weights=areas[valid])
    return np.asarray(mesh.centroid, dtype=np.float64)


def _support_hull_metrics(points_xy: np.ndarray, center_xy: np.ndarray) -> tuple[float, float, bool]:
    """Return support polygon area, signed COM margin and containment."""
    if len(points_xy) < 3:
        return 0.0, -1.0, False
    if len(points_xy) > 30_000:
        stride = max(1, len(points_xy) // 30_000)
        points_xy = points_xy[::stride]
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points_xy)
    except Exception:
        return 0.0, -1.0, False
    equations = np.asarray(hull.equations, dtype=np.float64)
    denominators = np.linalg.norm(equations[:, :2], axis=1)
    distances = -(
        equations[:, :2] @ np.asarray(center_xy, dtype=np.float64)
        + equations[:, 2]
    ) / np.maximum(denominators, 1e-12)
    margin = float(distances.min())
    return float(hull.volume), margin, bool(margin >= -1e-10)


def _axis_metrics(mesh: trimesh.Trimesh, transform: np.ndarray, label: str) -> dict:
    rotation = np.asarray(transform[:3, :3], dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64) @ rotation.T
    normals = np.asarray(mesh.face_normals, dtype=np.float64) @ rotation.T
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    z = vertices[:, 2]
    bounds = np.vstack((vertices.min(axis=0), vertices.max(axis=0)))
    extents = bounds[1] - bounds[0]
    longest = max(float(extents.max()), 1e-12)
    bed_band = max(longest * 0.006, float(extents[2]) * 0.004)
    face_z_max = z[np.asarray(mesh.faces)].max(axis=1)
    near_bed = face_z_max <= float(z.min()) + bed_band
    flat = np.abs(normals[:, 2]) >= math.cos(math.radians(25.0))
    contact = near_bed & flat
    projected_contact = float(np.sum(areas[contact] * np.abs(normals[contact, 2])))

    if contact.any():
        support_indices = np.unique(np.asarray(mesh.faces)[contact].reshape(-1))
    else:
        support_indices = np.flatnonzero(z <= float(z.min()) + bed_band)
    center = _surface_centroid(mesh) @ rotation.T
    support_area, stability_margin, stable = _support_hull_metrics(
        vertices[support_indices, :2], center[:2]
    )

    severe_cutoff = math.cos(math.radians(50.0))
    severity = np.clip(
        (-normals[:, 2] - severe_cutoff) / (1.0 - severe_cutoff), 0.0, 1.0
    )
    overhang_area = float(np.sum(areas[~near_bed] * severity[~near_bed]))
    total_area = max(float(areas.sum()), 1e-12)
    footprint_box_area = max(float(extents[0] * extents[1]), 1e-12)
    footprint_diagonal = max(float(np.linalg.norm(extents[:2])), 1e-12)
    return {
        "label": label,
        "contact_ratio": projected_contact / total_area,
        "support_fill_ratio": min(1.0, support_area / footprint_box_area),
        "stable": stable,
        "stability_margin_ratio": stability_margin / footprint_diagonal,
        "overhang_ratio": overhang_area / total_area,
        "height_ratio": float(extents[2]) / longest,
    }


def _normalized(values: list[float], *, higher_is_better: bool) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    low = float(array.min())
    high = float(array.max())
    if high - low <= 1e-12:
        result = np.full(len(array), 0.5, dtype=np.float64)
    else:
        result = (array - low) / (high - low)
    return result if higher_is_better else 1.0 - result


def choose_vertical_axis(
    mesh: trimesh.Trimesh, orientation_mode: str = "semantic"
) -> tuple[np.ndarray, str, list[dict]]:
    """Choose model up without confusing a large back panel for the base.

    Stage 03 already maps Pixal3D coordinates so the reference image's
    semantic vertical is +Z. Product renders commonly contain a back plate,
    sign face, or wall much larger than the true feet. A six-axis contact-area
    maximizer therefore cannot be the default: it lays such objects on their
    backs. Semantic mode keeps all six-axis metrics for diagnostics, while only
    explicit ``auto-print`` mode may override reference +Z.
    """
    if orientation_mode not in {"semantic", "auto-print"}:
        raise PipelineError(f"未知竖直方向策略: {orientation_mode}")
    transforms: list[np.ndarray] = []
    metrics: list[dict] = []
    for label, source_up in AXIS_CANDIDATES.items():
        transform = trimesh.geometry.align_vectors(source_up, [0.0, 0.0, 1.0])
        if transform is None:
            transform = np.eye(4)
        transform = np.asarray(transform, dtype=np.float64)
        transforms.append(transform)
        metrics.append(_axis_metrics(mesh, transform, label))

    contact = _normalized([m["contact_ratio"] for m in metrics], higher_is_better=True)
    support = _normalized([m["support_fill_ratio"] for m in metrics], higher_is_better=True)
    margin = _normalized([m["stability_margin_ratio"] for m in metrics], higher_is_better=True)
    overhang = _normalized([m["overhang_ratio"] for m in metrics], higher_is_better=False)
    height = _normalized([m["height_ratio"] for m in metrics], higher_is_better=False)
    for index, item in enumerate(metrics):
        semantic_prior = 1.0 if item["label"] == "+Z" else 0.0
        stable_bonus = 1.0 if item["stable"] else 0.0
        score = (
            0.36 * contact[index]
            + 0.18 * support[index]
            + 0.18 * margin[index]
            + 0.13 * overhang[index]
            + 0.07 * height[index]
            + 0.05 * stable_bonus
            + 0.03 * semantic_prior
        )
        item["score"] = round(float(score), 8)
        for key in (
            "contact_ratio",
            "support_fill_ratio",
            "stability_margin_ratio",
            "overhang_ratio",
            "height_ratio",
        ):
            item[key] = round(float(item[key]), 8)
    semantic_index = next(
        index for index, item in enumerate(metrics) if item["label"] == "+Z"
    )
    if orientation_mode == "semantic":
        best_index = semantic_index
        selection_reason = "reference_semantic_up"
    else:
        best_index = int(np.argmax([item["score"] for item in metrics]))
        selection_reason = "highest_print_score"
        if best_index != semantic_index:
            alternative = metrics[best_index]
            semantic = metrics[semantic_index]
            stable_base_evidence = (
                alternative["stable"]
                and alternative["contact_ratio"] >= 0.002
                and alternative["support_fill_ratio"] >= 0.02
            )
            # A vertical back plate can have excellent contact/fill ratios when
            # laid flat. It is not a credible base unless the COM projection is
            # also contained by the support polygon.
            strong_flat_base_evidence = (
                alternative["stable"]
                and alternative["contact_ratio"] >= 0.008
                and alternative["support_fill_ratio"] >= 0.04
                and alternative["score"] >= semantic["score"] + 0.20
            )
            if not (stable_base_evidence or strong_flat_base_evidence):
                best_index = semantic_index
                selection_reason = "semantic_z_up_guard"
            else:
                selection_reason = "strong_alternative_base_evidence"
    for index, item in enumerate(metrics):
        item["selected"] = index == best_index
        if index == best_index:
            item["selection_reason"] = selection_reason
    return transforms[best_index], metrics[best_index]["label"], metrics


def fit_bottom_plane_correction(
    mesh: trimesh.Trimesh, max_tilt_degrees: float
) -> tuple[np.ndarray, dict]:
    """RANSAC-fit a nearly horizontal supporting plane and level it."""
    if max_tilt_degrees <= 0:
        return np.eye(4), {"applied": False, "reason": "disabled"}
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) > 60_000:
        stride = max(1, len(vertices) // 60_000)
        sample = vertices[::stride]
    else:
        sample = vertices
    extents = np.ptp(sample, axis=0)
    longest = float(extents.max())
    height = float(extents[2])
    if longest <= 0 or height <= 0:
        return np.eye(4), {"applied": False, "reason": "zero_extent"}
    low = sample[sample[:, 2] <= float(sample[:, 2].min()) + 0.20 * height]
    if len(low) > 20_000:
        low = low[:: max(1, len(low) // 20_000)]
    if len(low) < 30:
        return np.eye(4), {"applied": False, "reason": "too_few_bottom_points"}

    tolerance = longest * 0.0025
    max_tilt = math.radians(max_tilt_degrees)
    rng = np.random.default_rng(0)
    best_inliers: np.ndarray | None = None
    best_count = 0
    iterations = min(128, max(48, len(low) // 100))
    for _ in range(iterations):
        points = low[rng.choice(len(low), size=3, replace=False)]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            continue
        normal /= norm
        if normal[2] < 0:
            normal *= -1.0
        tilt = math.acos(float(np.clip(normal[2], -1.0, 1.0)))
        if tilt > max_tilt:
            continue
        offset = float(np.dot(normal, points[0]))
        # A valid base is a supporting plane: almost all sampled points lie above it.
        all_signed = sample @ normal - offset
        if float(np.quantile(all_signed, 0.01)) < -2.0 * tolerance:
            continue
        inliers = np.abs(low @ normal - offset) <= tolerance
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
    minimum_inliers = max(30, int(0.03 * len(low)))
    if best_inliers is None or best_count < minimum_inliers:
        # Diffusion meshes often represent a visually planar base with noisy,
        # densely tessellated relief. Point RANSAC then sees no exact plane.
        # Recover only its *orientation* from broad, near-horizontal faces in
        # the lower quarter; the later SDF/floor pass creates the exact plane.
        centers = np.asarray(mesh.triangles_center, dtype=np.float64)
        normals = np.asarray(mesh.face_normals, dtype=np.float64).copy()
        areas = np.asarray(mesh.area_faces, dtype=np.float64)
        normals[normals[:, 2] < 0] *= -1.0
        normal_cone = min(40.0, max(30.0, 2.5 * max_tilt_degrees))
        candidates = (
            (centers[:, 2] <= float(sample[:, 2].min()) + 0.25 * height)
            & (normals[:, 2] >= math.cos(math.radians(normal_cone)))
        )
        candidate_area = float(areas[candidates].sum())
        total_area = max(float(areas.sum()), 1e-12)
        if np.count_nonzero(candidates) >= 30 and candidate_area > 0:
            coverage = np.ptp(centers[candidates, :2], axis=0) / np.maximum(
                extents[:2], 1e-12
            )
            normal = np.sum(
                normals[candidates] * areas[candidates, None], axis=0
            )
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm > 1e-12:
                normal /= normal_norm
                tilt = math.acos(float(np.clip(normal[2], -1.0, 1.0)))
                enough_evidence = (
                    candidate_area / total_area >= 0.005
                    and float(np.min(coverage)) >= 0.25
                    and tilt <= max_tilt
                )
                if enough_evidence:
                    transform = trimesh.geometry.align_vectors(
                        normal, [0.0, 0.0, 1.0]
                    )
                    if transform is None:
                        transform = np.eye(4)
                    return np.asarray(transform, dtype=np.float64), {
                        "applied": bool(math.degrees(tilt) > 1e-4),
                        "method": "area_weighted_low_surface_normals",
                        "tilt_degrees": round(math.degrees(tilt), 6),
                        "candidate_faces": int(np.count_nonzero(candidates)),
                        "candidate_area_ratio": round(
                            candidate_area / total_area, 6
                        ),
                        "xy_coverage_ratio": coverage.tolist(),
                        "ransac_best_inliers": best_count,
                    }
        return np.eye(4), {
            "applied": False,
            "reason": "no_dominant_support_orientation",
            "best_inliers": best_count,
            "bottom_points": int(len(low)),
        }

    plane_points = low[best_inliers]
    center = plane_points.mean(axis=0)
    covariance = np.cov(plane_points - center, rowvar=False)
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    if normal[2] < 0:
        normal *= -1.0
    tilt = math.acos(float(np.clip(normal[2], -1.0, 1.0)))
    if tilt > max_tilt:
        return np.eye(4), {"applied": False, "reason": "refined_tilt_too_large"}
    transform = trimesh.geometry.align_vectors(normal, [0.0, 0.0, 1.0])
    if transform is None:
        transform = np.eye(4)
    return np.asarray(transform, dtype=np.float64), {
        "applied": bool(math.degrees(tilt) > 1e-4),
        "tilt_degrees": round(math.degrees(tilt), 6),
        "inliers": int(best_count),
        "bottom_points": int(len(low)),
        "inlier_ratio": round(best_count / len(low), 6),
    }


def principal_xy_yaw(mesh: trimesh.Trimesh) -> float:
    """Align the longest horizontal principal axis to +X while keeping Z-up."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) > 200_000:
        vertices = vertices[:: max(1, len(vertices) // 200_000)]
    xy = vertices[:, :2] - vertices[:, :2].mean(axis=0, keepdims=True)
    covariance = np.cov(xy, rowvar=False)
    if not np.isfinite(covariance).all() or np.max(np.abs(covariance)) < 1e-12:
        return 0.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = math.atan2(float(direction[1]), float(direction[0]))
    while angle > math.pi / 2:
        angle -= math.pi
    while angle <= -math.pi / 2:
        angle += math.pi
    return -angle


def _ground_and_center(mesh: trimesh.Trimesh) -> np.ndarray:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center_xy = (bounds[0, :2] + bounds[1, :2]) / 2.0
    translation = np.array([-center_xy[0], -center_xy[1], -bounds[0, 2]])
    mesh.apply_translation(translation)
    return translation


def robust_base_cut(mesh: trimesh.Trimesh, cut_mm: float) -> tuple[trimesh.Trimesh, float]:
    """Clip the bottom with an exact plane; SDF subsequently closes the open cap."""
    if cut_mm <= 0:
        return mesh.copy(), 0.0
    height = float(mesh.extents[2])
    effective_cut = min(float(cut_mm), 0.05 * height)
    if effective_cut <= 1e-8:
        return mesh.copy(), 0.0
    try:
        # Use the array-level clipper.  Trimesh's higher-level
        # ``slice_mesh_plane`` imports Shapely even when ``cap=False``; the
        # array implementation has no optional polygon dependency and the SDF
        # stage below intentionally closes this planar opening.
        vertices, faces, _ = trimesh.intersections.slice_faces_plane(
            vertices=np.asarray(mesh.vertices, dtype=np.float64),
            faces=np.asarray(mesh.faces, dtype=np.int64),
            plane_normal=np.array([0.0, 0.0, 1.0]),
            plane_origin=np.array([0.0, 0.0, effective_cut]),
        )
        clipped = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
            validate=False,
        )
    except Exception as exc:
        raise PipelineError(f"底面平面裁切失败: {exc}") from exc
    if not isinstance(clipped, trimesh.Trimesh) or len(clipped.faces) == 0:
        raise PipelineError("底面平面裁切产生空网格")
    clipped.apply_translation([0.0, 0.0, -effective_cut])
    return clipped, effective_cut


def restore_flat_floor_after_sdf(
    mesh: trimesh.Trimesh, cut_mm: float
) -> tuple[trimesh.Trimesh, dict]:
    """Flatten a bottom vertex band without opening the watertight SDF surface."""
    height = float(mesh.extents[2])
    maximum_cut = min(max(float(cut_mm), 0.0), 0.03 * height)
    attempts: list[dict] = []
    best: tuple[trimesh.Trimesh, dict] | None = None
    # The zero-depth attempt guarantees that floor flattening never damages an
    # otherwise valid SDF shell merely to force a wider contact patch.
    factors = (1.0, 0.75, 0.5, 0.25, 0.0) if maximum_cut > 0 else (0.0,)
    for factor in factors:
        effective_cut = maximum_cut * factor
        flattened = mesh.copy()
        vertices = np.asarray(flattened.vertices, dtype=np.float64).copy()
        plane_z = float(vertices[:, 2].min()) + effective_cut
        projected = vertices[:, 2] <= plane_z
        vertices[projected, 2] = plane_z
        flattened.vertices = vertices
        flattened.apply_translation([0.0, 0.0, -plane_z])

        # Coincident positions are what slicers and Blender weld in practice.
        # Validate that geometry, not only the original GLB vertex indices.
        welded = flattened.copy()
        vertices_before_weld = int(len(welded.vertices))
        welded.merge_vertices(merge_tex=True, merge_norm=True)
        topology = mesh_topology_stats(welded)
        strict_topology = strict_surface_stats(welded)
        areas = np.asarray(flattened.area_faces, dtype=np.float64)
        area_tolerance = max(float(flattened.area) * 1e-14, 1e-14)
        collapsed_faces = int(np.count_nonzero(areas <= area_tolerance))
        attempt = {
            "cut_mm": float(effective_cut),
            "projected_vertices": int(np.count_nonzero(projected)),
            "welded_vertices": vertices_before_weld - int(len(welded.vertices)),
            "collapsed_faces": collapsed_faces,
            "welded_topology": topology,
            "strict_topology": strict_topology,
        }
        attempts.append(attempt)
        valid = (
            topology["watertight"]
            and topology["winding_consistent"]
            and topology["boundary_edges"] == 0
            and topology["nonmanifold_edges"] == 0
            and topology["component_count"] == 1
            and strict_surface_is_valid(strict_topology)
        )
        if valid:
            candidate_stats = {
                "applied": bool(effective_cut > 0),
                "method": "adaptive_topology_preserving_floor_projection",
                **attempt,
                "attempts": attempts,
            }
            best = welded, candidate_stats
            if collapsed_faces == 0:
                return best

    if best is not None:
        return best
    raise PipelineError(f"所有底面压平深度在焊接后都不是水密流形: {attempts}")


def sdf_solidify(
    mesh: trimesh.Trimesh,
    *,
    voxel_mm: float,
    sdf_offset_mm: float,
    max_voxel_axis: int,
    min_component_faces: int,
) -> tuple[trimesh.Trimesh, dict]:
    """Unsigned-distance resampling: thicken, bridge micro-gaps and watertight."""
    pymeshlab = _load_pymeshlab()
    longest = float(mesh.extents.max())
    actual_voxel = max(float(voxel_mm), longest / float(max_voxel_axis))
    pure_value = getattr(pymeshlab, "PureValue", None)
    if pure_value is None:
        pure_value = pymeshlab.AbsoluteValue

    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(_pymeshlab_mesh(mesh, pymeshlab), "cut_for_sdf")
    mesh_set.generate_resampled_uniform_mesh(
        cellsize=pure_value(actual_voxel),
        offset=pure_value(float(sdf_offset_mm)),
        mergeclosevert=True,
        discretize=False,
        multisample=True,
        absdist=True,
    )
    mesh_set.meshing_remove_duplicate_vertices()
    mesh_set.meshing_remove_duplicate_faces()
    mesh_set.meshing_remove_null_faces()
    mesh_set.meshing_remove_unreferenced_vertices()
    if min_component_faces > 0:
        mesh_set.meshing_remove_connected_component_by_face_number(
            mincomponentsize=int(min_component_faces), removeunref=True
        )
    rebuilt = _from_pymeshlab(mesh_set.current_mesh())
    if len(rebuilt.faces) == 0:
        raise PipelineError("Voxel/SDF 实体化产生空网格")
    return rebuilt, {
        "requested_voxel_mm": float(voxel_mm),
        "actual_voxel_mm": float(actual_voxel),
        "sdf_offset_mm": float(sdf_offset_mm),
        "vertices_before_single_body": int(len(rebuilt.vertices)),
        "faces_before_single_body": int(len(rebuilt.faces)),
    }


def signed_sdf_cleanup(
    mesh: trimesh.Trimesh,
    *,
    voxel_mm: float,
    max_voxel_axis: int,
    min_component_faces: int,
) -> tuple[trimesh.Trimesh, dict]:
    """Zero-offset signed SDF pass for pinches and self-intersections."""
    pymeshlab = _load_pymeshlab()
    longest = float(mesh.extents.max())
    actual_voxel = max(float(voxel_mm), longest / float(max_voxel_axis))
    pure_value = getattr(pymeshlab, "PureValue", None)
    if pure_value is None:
        pure_value = pymeshlab.AbsoluteValue

    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(_pymeshlab_mesh(mesh, pymeshlab), "manifold_cleanup")
    mesh_set.generate_resampled_uniform_mesh(
        cellsize=pure_value(actual_voxel),
        offset=pure_value(0.0),
        mergeclosevert=True,
        discretize=False,
        multisample=True,
        absdist=False,
    )
    mesh_set.meshing_remove_duplicate_vertices()
    mesh_set.meshing_remove_duplicate_faces()
    mesh_set.meshing_remove_null_faces()
    mesh_set.meshing_remove_unreferenced_vertices()
    if min_component_faces > 0:
        mesh_set.meshing_remove_connected_component_by_face_number(
            mincomponentsize=int(min_component_faces), removeunref=True
        )
    cleaned = _from_pymeshlab(mesh_set.current_mesh())
    if len(cleaned.faces) == 0:
        raise PipelineError("有符号 SDF 流形清理产生空网格")
    cleaned, body_stats = keep_largest_body(cleaned)
    cleaned, weld_stats = weld_for_topology(cleaned)
    return cleaned, {
        "actual_voxel_mm": actual_voxel,
        "body_selection": body_stats,
        "weld": weld_stats,
    }


def keep_largest_body(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict]:
    """Enforce one connected printable body, keeping the largest enclosed volume."""
    components = list(mesh.split(only_watertight=False))
    if not components:
        raise PipelineError("连通域分解没有得到有效组件")

    def component_key(component: trimesh.Trimesh) -> tuple[int, float, float]:
        volume = abs(float(component.volume)) if component.is_watertight else 0.0
        # Face count is the robust primary key here.  A large temporarily open
        # body has reported volume zero and must not lose to a tiny closed speck.
        return int(len(component.faces)), volume, float(component.area)

    components.sort(key=component_key, reverse=True)
    largest = components[0].copy()
    if largest.is_watertight and float(largest.volume) < 0:
        largest.invert()
    return largest, {
        "components_before_single_body": int(len(components)),
        "removed_components": int(len(components) - 1),
        "kept_component_faces": int(len(largest.faces)),
    }


def mesh_topology_stats(mesh: trimesh.Trimesh) -> dict:
    inverse = np.asarray(mesh.edges_unique_inverse, dtype=np.int64)
    counts = np.bincount(inverse, minlength=len(mesh.edges_unique))
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    area_tolerance = max(float(mesh.area) * 1e-14, 1e-14)
    return {
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "degenerate_faces": int(np.count_nonzero(areas <= area_tolerance)),
        "component_count": int(mesh.body_count),
    }


def weld_for_topology(
    mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, dict]:
    """Weld coincident positions before topology checks, as slicers do."""
    welded = mesh.copy()
    vertices_before = int(len(welded.vertices))
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    return welded, {
        "vertices_before_weld": vertices_before,
        "vertices_after_weld": int(len(welded.vertices)),
        "coincident_vertices_merged": vertices_before - int(len(welded.vertices)),
    }


def strict_surface_stats(mesh: trimesh.Trimesh) -> dict:
    """MeshLab checks vertex-manifoldness and geometric self-intersections."""
    pymeshlab = _load_pymeshlab()
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(_pymeshlab_mesh(mesh, pymeshlab), "strict_validation")
    measures = mesh_set.get_topological_measures()
    mesh_set.compute_selection_by_self_intersections_per_face()
    selected = np.asarray(
        mesh_set.current_mesh().face_selection_array(), dtype=bool
    )
    return {
        "two_manifold": bool(measures["is_mesh_two_manifold"]),
        "nonmanifold_vertices": int(measures["non_two_manifold_vertices"]),
        "nonmanifold_edges_meshlab": int(measures["non_two_manifold_edges"]),
        "self_intersecting_faces": int(np.count_nonzero(selected)),
        "boundary_edges_meshlab": int(measures["boundary_edges"]),
        "components_meshlab": int(measures["connected_components_number"]),
    }


def strict_surface_is_valid(stats: dict) -> bool:
    return bool(
        stats["two_manifold"]
        and stats["nonmanifold_vertices"] == 0
        and stats["nonmanifold_edges_meshlab"] == 0
        and stats["self_intersecting_faces"] == 0
        and stats["boundary_edges_meshlab"] == 0
        and stats["components_meshlab"] == 1
    )


def repair_sdf_topology(
    mesh: trimesh.Trimesh,
    *,
    voxel_mm: float,
    max_voxel_axis: int,
    min_component_faces: int,
) -> tuple[trimesh.Trimesh, dict]:
    """Remove rare SDF pinch edges and directionally close their tiny holes."""
    repaired, initial_weld = weld_for_topology(mesh)
    before = mesh_topology_stats(repaired)
    strict_before = strict_surface_stats(repaired)
    if (
        before["watertight"]
        and before["nonmanifold_edges"] == 0
        and before["degenerate_faces"] == 0
        and before["component_count"] == 1
        and before["winding_consistent"]
        and strict_surface_is_valid(strict_before)
    ):
        return repaired, {
            "applied": bool(initial_weld["coincident_vertices_merged"]),
            "method": "weld_coincident_vertices",
            "initial_weld": initial_weld,
            "before": before,
            "strict_before": strict_before,
            "after": before,
            "strict_after": strict_before,
        }

    # The unsigned solidification can create pinch contacts even though its
    # geometric surface is suitable for signed resampling. Try the zero-offset
    # signed SDF before deleting faces: it removes non-manifold pinches,
    # degenerate triangles and self-intersections without creating branched
    # boundary loops that are impossible to cap as simple polygons.
    signed_cleanup_error: str | None = None
    try:
        signed, signed_stats = signed_sdf_cleanup(
            repaired,
            voxel_mm=voxel_mm,
            max_voxel_axis=max_voxel_axis,
            min_component_faces=min_component_faces,
        )
        signed_topology = mesh_topology_stats(signed)
        signed_strict = strict_surface_stats(signed)
        if (
            signed_topology["watertight"]
            and signed_topology["nonmanifold_edges"] == 0
            and signed_topology["degenerate_faces"] == 0
            and signed_topology["component_count"] == 1
            and signed_topology["winding_consistent"]
            and strict_surface_is_valid(signed_strict)
        ):
            return signed, {
                "applied": True,
                "method": "signed_sdf_manifold_cleanup",
                "initial_weld": initial_weld,
                "before": before,
                "strict_before": strict_before,
                "signed_sdf": signed_stats,
                "after": signed_topology,
                "strict_after": signed_strict,
            }
    except Exception as exc:
        signed_cleanup_error = str(exc)

    repair_source = repaired.copy()
    retry_history: list[dict] = []

    def adaptive_resdf() -> tuple[trimesh.Trimesh, dict] | None:
        nonlocal repaired
        retry_offset = 0.75 * float(voxel_mm)
        for retry_index in range(1, 4):
            repaired, retry_sdf_stats = sdf_solidify(
                repaired,
                voxel_mm=voxel_mm,
                sdf_offset_mm=retry_offset,
                max_voxel_axis=max_voxel_axis,
                min_component_faces=min_component_faces,
            )
            repaired, retry_body_stats = keep_largest_body(repaired)
            repaired, retry_weld_stats = weld_for_topology(repaired)
            retry_topology = mesh_topology_stats(repaired)
            retry_strict = strict_surface_stats(repaired)
            retry_history.append(
                {
                    "pass": retry_index,
                    "sdf": retry_sdf_stats,
                    "body_selection": retry_body_stats,
                    "weld": retry_weld_stats,
                    "topology": retry_topology,
                    "strict_topology": retry_strict,
                }
            )
            if (
                retry_topology["watertight"]
                and retry_topology["nonmanifold_edges"] == 0
                and retry_topology["degenerate_faces"] == 0
                and retry_topology["component_count"] == 1
                and retry_topology["winding_consistent"]
                and strict_surface_is_valid(retry_strict)
            ):
                return repaired, {
                    "applied": True,
                    "method": "adaptive_resdf",
                    "before": before,
                    "retry_offset_mm": retry_offset,
                    "retry_history": retry_history,
                    "after": retry_topology,
                    "strict_after": retry_strict,
                }
        return None

    # Several coincident expanded sheets can meet at one edge.  For complex
    # pinches, another small-offset UDF pass is more reliable than opening a
    # branched boundary and trying to triangulate it manually.
    if before["nonmanifold_edges"] > 2 or before["boundary_edges"] > 0:
        adaptive_result = adaptive_resdf()
        if adaptive_result is not None:
            return adaptive_result

    removed_nonmanifold_faces = 0
    if before["nonmanifold_edges"] > 0:
        pymeshlab = _load_pymeshlab()
        mesh_set = pymeshlab.MeshSet()
        mesh_set.add_mesh(_pymeshlab_mesh(repaired, pymeshlab), "sdf_nonmanifold")
        faces_before = int(mesh_set.current_mesh().face_number())
        mesh_set.meshing_repair_non_manifold_edges(method="Remove Faces")
        mesh_set.meshing_remove_unreferenced_vertices()
        repaired = _from_pymeshlab(mesh_set.current_mesh())
        removed_nonmanifold_faces = faces_before - int(len(repaired.faces))

    after_removal = mesh_topology_stats(repaired)
    cap_faces_added = 0
    if after_removal["boundary_edges"] > 0:
        try:
            repaired, cap_faces_added = fill_residual_boundary_loops(
                repaired, max_total_edges=10_000
            )
        except PipelineError:
            repaired = repair_source.copy()
            adaptive_result = adaptive_resdf()
            if adaptive_result is not None:
                return adaptive_result
            raise
    repaired, body_stats = keep_largest_body(repaired)
    repaired, final_weld = weld_for_topology(repaired)
    after = mesh_topology_stats(repaired)
    if (
        not after["watertight"]
        or after["nonmanifold_edges"] != 0
        or after["degenerate_faces"] != 0
    ):
        raise PipelineError(
            "SDF 拓扑修复失败: "
            f"before={before}, after_removal={after_removal}, after={after}, "
            f"signed_cleanup_error={signed_cleanup_error}"
        )
    if after["component_count"] != 1 or not after["winding_consistent"]:
        raise PipelineError(f"SDF 拓扑修复后组件或绕序异常: {after}")
    strict_after = strict_surface_stats(repaired)
    if not strict_surface_is_valid(strict_after):
        signed, signed_stats = signed_sdf_cleanup(
            repaired,
            voxel_mm=voxel_mm,
            max_voxel_axis=max_voxel_axis,
            min_component_faces=min_component_faces,
        )
        signed_topology = mesh_topology_stats(signed)
        signed_strict = strict_surface_stats(signed)
        if strict_surface_is_valid(signed_strict):
            return signed, {
                "applied": True,
                "method": "local_repair_then_signed_sdf_cleanup",
                "initial_weld": initial_weld,
                "before": before,
                "strict_before": strict_before,
                "local_after": after,
                "local_strict_after": strict_after,
                "signed_sdf": signed_stats,
                "after": signed_topology,
                "strict_after": signed_strict,
            }
        raise PipelineError(
            f"SDF 严格流形修复失败: topology={signed_topology}, strict={signed_strict}"
        )
    return repaired, {
        "applied": True,
        "method": "local_nonmanifold_face_repair",
        "before": before,
        "strict_before": strict_before,
        "removed_nonmanifold_faces": removed_nonmanifold_faces,
        "cap_faces_added": cap_faces_added,
        "body_selection": body_stats,
        "initial_weld": initial_weld,
        "final_weld": final_weld,
        "retry_history": retry_history,
        "after": after,
        "strict_after": strict_after,
    }


def fill_residual_boundary_loops(
    mesh: trimesh.Trimesh, *, max_total_edges: int = 2_000
) -> tuple[trimesh.Trimesh, int]:
    """Fan-fill small boundary loops left by MeshLab's planar hole closer."""
    inverse = np.asarray(mesh.edges_unique_inverse, dtype=np.int64)
    counts = np.bincount(inverse, minlength=len(mesh.edges_unique))
    boundary_ids = np.flatnonzero(counts == 1)
    if len(boundary_ids) == 0:
        return mesh, 0
    if len(boundary_ids) > max_total_edges:
        raise PipelineError(f"封底后仍有过大的边界，共 {len(boundary_ids)} 条边")

    directed_edges: list[tuple[int, int]] = []
    all_directed = np.asarray(mesh.edges, dtype=np.int64)
    for unique_id in boundary_ids:
        occurrence = int(np.flatnonzero(inverse == unique_id)[0])
        # A closing face must traverse the shared edge opposite to the existing face.
        start, end = all_directed[occurrence]
        directed_edges.append((int(end), int(start)))

    outgoing: dict[int, list[int]] = {}
    for start, end in directed_edges:
        outgoing.setdefault(start, []).append(end)
    if any(len(ends) != 1 for ends in outgoing.values()):
        raise PipelineError("残余边界不是可封闭的简单环")

    unused = set(directed_edges)
    loops: list[list[int]] = []
    while unused:
        start_edge = next(iter(unused))
        start = start_edge[0]
        loop = [start]
        current = start
        for _ in range(len(directed_edges) + 1):
            next_vertex = outgoing[current][0]
            edge = (current, next_vertex)
            if edge not in unused:
                if next_vertex == start:
                    break
                raise PipelineError("残余边界环发生重复或断裂")
            unused.remove(edge)
            current = next_vertex
            if current == start:
                break
            loop.append(current)
        if current != start or len(loop) < 3:
            raise PipelineError("无法闭合残余边界环")
        loops.append(loop)

    vertices = np.asarray(mesh.vertices, dtype=np.float64).tolist()
    faces = np.asarray(mesh.faces, dtype=np.int64).tolist()
    faces_added = 0
    for loop in loops:
        if len(loop) == 3:
            faces.append(loop)
            faces_added += 1
            continue
        center_index = len(vertices)
        vertices.append(np.asarray(mesh.vertices)[loop].mean(axis=0).tolist())
        for index, vertex in enumerate(loop):
            faces.append([vertex, loop[(index + 1) % len(loop)], center_index])
            faces_added += 1
    repaired = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
        validate=False,
    )
    return repaired, faces_added


def transfer_vertex_colors(source: trimesh.Trimesh, target: trimesh.Trimesh) -> dict:
    """Nearest-neighbour transfer from baked raw texture samples to SDF vertices."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise PipelineError("顶点色转移需要 scipy") from exc
    source_vertices = np.asarray(source.vertices, dtype=np.float64)
    source_colors = _vertex_colors(source)
    maximum_sources = 750_000
    stride = max(1, math.ceil(len(source_vertices) / maximum_sources))
    sampled_vertices = source_vertices[::stride]
    sampled_colors = source_colors[::stride]
    tree = cKDTree(sampled_vertices)
    distances, indices = tree.query(
        np.asarray(target.vertices, dtype=np.float64), k=1, workers=-1
    )
    target.visual.vertex_colors = sampled_colors[np.asarray(indices, dtype=np.int64)]
    return {
        "appearance": "nearest_vertex_color_from_raw_texture",
        "color_source_vertices": int(len(sampled_vertices)),
        "color_target_vertices": int(len(target.vertices)),
        "color_mean_distance_mm": float(np.mean(distances)),
        "color_max_distance_mm": float(np.max(distances)),
    }


def export_texture_bake_source(
    source_geometries: list[trimesh.Trimesh],
    transform: np.ndarray,
    output_path: Path,
) -> dict:
    """Export aligned raw textured meshes with PNG-compatible embedded images."""
    source_scene = trimesh.Scene()
    texture_count = 0
    for index, source in enumerate(source_geometries):
        mesh = source.copy()
        mesh.apply_transform(transform)
        material = getattr(mesh.visual, "material", None)
        texture = getattr(material, "baseColorTexture", None)
        if texture is not None:
            # Pixal3D uses EXT_texture_webp, unsupported by Blender 3.6's
            # importer. An image with no original format is emitted as PNG by
            # Trimesh's standard glTF exporter.
            material.baseColorTexture = texture.convert("RGBA")
            texture_count += 1
        source_scene.add_geometry(
            mesh,
            node_name=f"texture_source_{index}",
            geom_name=f"texture_source_{index}",
            transform=np.eye(4),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_scene.export(str(output_path), file_type="glb")
    return {
        "path": output_path.name,
        "geometry_count": len(source_geometries),
        "embedded_png_textures": texture_count,
        "bytes": output_path.stat().st_size,
    }


def _apply_to_both(
    geometry: trimesh.Trimesh, appearance: trimesh.Trimesh, transform: np.ndarray
) -> None:
    geometry.apply_transform(transform)
    appearance.apply_transform(transform)


def postprocess_one(
    raw_path: Path, output_path: Path, target_mm: float, args: argparse.Namespace
) -> dict:
    if not math.isfinite(target_mm) or target_mm <= 0:
        raise PipelineError(f"无效目标尺寸: {target_mm}")

    # ① flatten scene; ② merge to one mesh while retaining a color reference.
    geometries = load_and_flatten(raw_path)
    appearance = merge_geometries(geometries, preserve_colors=not args.no_color_transfer)
    merged = merge_geometries(geometries, preserve_colors=False)
    source_geometry_count = len(geometries)
    raw_to_model_transform = np.eye(4, dtype=np.float64)

    def record_transform(transform: np.ndarray) -> None:
        nonlocal raw_to_model_transform
        raw_to_model_transform = (
            np.asarray(transform, dtype=np.float64) @ raw_to_model_transform
        )

    # ③ obvious fragment cleanup.
    geometry, cleanup_stats = remove_small_fragments(
        merged, args.min_component_faces
    )

    # ④ six possible vertical axes.
    vertical_transform, selected_up, orientation_candidates = choose_vertical_axis(
        geometry, args.orientation_mode
    )
    _apply_to_both(geometry, appearance, vertical_transform)
    record_transform(vertical_transform)

    # ⑤ small bottom-plane leveling correction.
    plane_transform, plane_stats = fit_bottom_plane_correction(
        geometry, args.bottom_plane_max_tilt_deg
    )
    _apply_to_both(geometry, appearance, plane_transform)
    record_transform(plane_transform)

    # ⑥ Preserve the reference camera's front direction in semantic mode.
    # Principal-axis yaw is useful for bed packing, but it destroys the intended
    # front/side relationship and can rotate a correct product view by ~90°.
    yaw = 0.0 if args.orientation_mode == "semantic" else principal_xy_yaw(geometry)
    yaw_transform = trimesh.transformations.rotation_matrix(yaw, [0.0, 0.0, 1.0])
    _apply_to_both(geometry, appearance, yaw_transform)
    record_transform(yaw_transform)

    # ⑦ physical millimetre scaling.
    longest = float(geometry.extents.max())
    if not math.isfinite(longest) or longest <= 0:
        raise PipelineError(f"无效模型尺寸: {geometry.extents.tolist()}")
    initial_scale = float(target_mm) / longest
    initial_scale_transform = np.diag([initial_scale, initial_scale, initial_scale, 1.0])
    _apply_to_both(geometry, appearance, initial_scale_transform)
    record_transform(initial_scale_transform)
    translation = _ground_and_center(geometry)
    appearance.apply_translation(translation)
    record_transform(trimesh.transformations.translation_matrix(translation))

    # ⑧ robust plane clipping. The intentionally open cap is closed by SDF.
    geometry, effective_cut = robust_base_cut(geometry, args.base_cut_mm)
    if effective_cut > 0:
        cut_translation = np.array([0.0, 0.0, -effective_cut])
        appearance.apply_translation(cut_translation)
        record_transform(
            trimesh.transformations.translation_matrix(cut_translation)
        )

    # ⑨ voxel/SDF solidification; ⑩ component cleanup; ⑪ one-body enforcement.
    geometry, sdf_stats = sdf_solidify(
        geometry,
        voxel_mm=args.voxel_mm,
        sdf_offset_mm=args.sdf_offset_mm,
        max_voxel_axis=args.max_voxel_axis,
        min_component_faces=args.min_component_faces,
    )
    geometry, component_stats = keep_largest_body(geometry)
    geometry, sdf_topology_repair_stats = repair_sdf_topology(
        geometry,
        voxel_mm=float(sdf_stats["actual_voxel_mm"]),
        max_voxel_axis=args.max_voxel_axis,
        min_component_faces=args.min_component_faces,
    )

    # SDF expands the bounds. First return to millimetres, then restore the
    # exact planar floor which the iso-surface necessarily rounded.
    sdf_bound_scale = float(target_mm) / float(geometry.extents.max())
    geometry.apply_scale(sdf_bound_scale)
    appearance.apply_scale(sdf_bound_scale)
    record_transform(
        np.diag([sdf_bound_scale, sdf_bound_scale, sdf_bound_scale, 1.0])
    )
    sdf_ground_translation = _ground_and_center(geometry)
    appearance.apply_translation(sdf_ground_translation)
    record_transform(
        trimesh.transformations.translation_matrix(sdf_ground_translation)
    )

    # ⑫ exact final floor, one-body selection, sizing, centering and grounding.
    final_floor_cut = min(
        float(args.base_cut_mm),
        0.75 * float(sdf_stats["actual_voxel_mm"]) * sdf_bound_scale,
    )
    geometry, final_floor_stats = restore_flat_floor_after_sdf(
        geometry, final_floor_cut
    )
    if final_floor_stats["applied"]:
        floor_translation = np.array(
            [0.0, 0.0, -final_floor_stats["cut_mm"]]
        )
        appearance.apply_translation(floor_translation)
        record_transform(
            trimesh.transformations.translation_matrix(floor_translation)
        )
    final_dimension_scale = float(target_mm) / float(geometry.extents.max())
    geometry.apply_scale(final_dimension_scale)
    appearance.apply_scale(final_dimension_scale)
    record_transform(
        np.diag(
            [
                final_dimension_scale,
                final_dimension_scale,
                final_dimension_scale,
                1.0,
            ]
        )
    )
    final_translation = _ground_and_center(geometry)
    appearance.apply_translation(final_translation)
    record_transform(trimesh.transformations.translation_matrix(final_translation))

    geometry, final_geometry_weld = weld_for_topology(geometry)
    topology = mesh_topology_stats(geometry)
    strict_topology = strict_surface_stats(geometry)
    if not topology["watertight"]:
        raise PipelineError(f"Watertight 检验失败: {topology}")
    if topology["nonmanifold_edges"] != 0:
        raise PipelineError(f"Non-manifold 检验失败: {topology}")
    if topology["degenerate_faces"] != 0:
        raise PipelineError(f"退化面检验失败: {topology}")
    if topology["component_count"] != 1:
        raise PipelineError(f"Component 数量检验失败: {topology}")
    if not topology["winding_consistent"]:
        raise PipelineError(f"面绕序检验失败: {topology}")
    if not strict_surface_is_valid(strict_topology):
        raise PipelineError(f"严格流形/自相交检验失败: {strict_topology}")

    if args.no_color_transfer:
        color_stats = {"appearance": "uniform_default_material"}
    else:
        color_stats = transfer_vertex_colors(appearance, geometry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_scene = trimesh.Scene()
    output_scene.add_geometry(
        geometry, node_name="printable_body", geom_name="printable_body", transform=np.eye(4)
    )
    output_scene.export(str(output_path), file_type="glb")

    # ⑬-⑮ reload the actual file and repeat all submission-critical checks.
    checked_parts = load_and_flatten(output_path)
    checked = merge_geometries(checked_parts, preserve_colors=False)
    checked, export_weld = weld_for_topology(checked)
    checked_topology = mesh_topology_stats(checked)
    checked_strict_topology = strict_surface_stats(checked)
    final_bounds = np.asarray(checked.bounds, dtype=np.float64)
    final_extents = final_bounds[1] - final_bounds[0]
    if abs(float(final_bounds[0, 2])) > 1e-4:
        raise PipelineError(f"导出后最低点不是 Z=0: {final_bounds[0, 2]}")
    if not np.isclose(float(final_extents.max()), target_mm, rtol=1e-5, atol=1e-3):
        raise PipelineError(
            f"导出后最长边 {final_extents.max()} mm 与目标 {target_mm} mm 不一致"
        )
    if not checked_topology["watertight"]:
        raise PipelineError(f"导出后 Watertight 检验失败: {checked_topology}")
    if checked_topology["nonmanifold_edges"] != 0:
        raise PipelineError(f"导出后 Non-manifold 检验失败: {checked_topology}")
    if checked_topology["degenerate_faces"] != 0:
        raise PipelineError(f"导出后退化面检验失败: {checked_topology}")
    if checked_topology["component_count"] != 1:
        raise PipelineError(f"导出后 Component 数量检验失败: {checked_topology}")
    if not strict_surface_is_valid(checked_strict_topology):
        raise PipelineError(
            f"导出后严格流形/自相交检验失败: {checked_strict_topology}"
        )

    texture_source_path = output_path.with_name(".texture_source.glb")
    texture_source_stats = export_texture_bake_source(
        geometries, raw_to_model_transform, texture_source_path
    )

    return {
        "postprocess_version": "manufacturing_sdf_v4_semantic_orientation",
        "source_geometry_count": source_geometry_count,
        "merged_to_single_trimesh": True,
        "cleanup": cleanup_stats,
        "selected_up_axis": selected_up,
        "orientation_mode": args.orientation_mode,
        "orientation_candidates": orientation_candidates,
        "bottom_plane_correction": plane_stats,
        "yaw_mode": (
            "reference_semantic_front"
            if args.orientation_mode == "semantic"
            else "horizontal_principal_axis"
        ),
        "yaw_degrees": round(math.degrees(yaw), 6),
        "initial_uniform_scale": initial_scale,
        "target_longest_mm": target_mm,
        "base_cut_mm": float(effective_cut),
        "sdf": sdf_stats,
        "components": component_stats,
        "sdf_topology_repair": sdf_topology_repair_stats,
        "sdf_bound_correction_scale": sdf_bound_scale,
        "final_floor": final_floor_stats,
        "final_geometry_weld": final_geometry_weld,
        "export_weld": export_weld,
        "strict_topology": checked_strict_topology,
        "final_dimension_correction_scale": final_dimension_scale,
        "appearance": color_stats,
        "raw_to_model_transform": raw_to_model_transform.tolist(),
        "texture_bake_source": texture_source_stats,
        "vertices": int(len(checked.vertices)),
        "faces": int(len(checked.faces)),
        "bounds_mm": final_bounds.tolist(),
        "extents_mm": final_extents.tolist(),
        **checked_topology,
        "bytes": output_path.stat().st_size,
    }


def main() -> int:
    args = parse_args()
    tasks = discover_tasks(args.data_root, args.expected_count)
    if args.prompt_id:
        requested = set(args.prompt_id)
        tasks = [task for task in tasks if task["prompt_id"] in requested]
        found = {task["prompt_id"] for task in tasks}
        missing = sorted(requested - found)
        if missing:
            raise PipelineError(f"不存在的 prompt_id: {', '.join(missing)}")
    failures: list[str] = []
    processed = 0
    for index, task in enumerate(tasks, start=1):
        prompt_id = task["prompt_id"]
        output_dir = task_dir(args.work_root, prompt_id)
        raw_path = output_dir / "raw.glb"
        plan_path = output_dir / "plan.json"
        output_path = output_dir / "model.glb"
        if not raw_path.exists():
            failures.append(prompt_id)
            update_status(
                args.work_root,
                prompt_id,
                "postprocess",
                ok=False,
                seconds=0.0,
                error=f"缺少 {raw_path}",
            )
            print(f"[{index}/{len(tasks)}] 失败: 缺少 {raw_path}", file=sys.stderr)
            continue
        if not plan_path.exists():
            failures.append(prompt_id)
            update_status(
                args.work_root,
                prompt_id,
                "postprocess",
                ok=False,
                seconds=0.0,
                error=f"缺少 {plan_path}",
            )
            print(f"[{index}/{len(tasks)}] 失败: 缺少 {plan_path}", file=sys.stderr)
            continue
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(tasks)}] 跳过已有 {prompt_id}/model.glb")
            continue

        plan = load_json(plan_path)
        target_mm = float(plan["target_longest_mm"])
        started = time.monotonic()
        print(f"[{index}/{len(tasks)}] 制造后处理 {prompt_id}/raw.glb", flush=True)
        try:
            stats = postprocess_one(raw_path, output_path, target_mm, args)
            update_status(
                args.work_root,
                prompt_id,
                "postprocess",
                ok=True,
                seconds=round(time.monotonic() - started, 3),
                **stats,
            )
            processed += 1
            print(
                f"  完成: up={stats['selected_up_axis']}，"
                f"{stats['vertices']} 顶点/{stats['faces']} 面，"
                f"watertight={stats['watertight']}，components={stats['component_count']}",
                flush=True,
            )
        except Exception as exc:
            failures.append(prompt_id)
            update_status(
                args.work_root,
                prompt_id,
                "postprocess",
                ok=False,
                seconds=round(time.monotonic() - started, 3),
                error=str(exc),
            )
            print(f"  失败: {exc}", file=sys.stderr, flush=True)

    if failures:
        print(f"后处理失败 {len(failures)} 题: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"制造后处理完成：新处理 {processed} 题，总计 {len(tasks)} 题。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
