#!/usr/bin/env python3
"""Blender background script: render four standard views for every model.glb."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--texture-resolution",
        type=int,
        default=2048,
        help="烘焙到最终 model.glb 的方形底色纹理分辨率，默认2048",
    )
    parser.add_argument(
        "--skip-texture-bake",
        action="store_true",
        help="只渲染，不从 .texture_source.glb 烘焙最终模型纹理",
    )
    parser.add_argument(
        "--embed-uv-texture",
        action="store_true",
        help=(
            "把UV纹理写回model.glb；默认只给临时渲染副本烘焙，"
            "以避免UV接缝顶点使Windows 3D Builder误报开放边界"
        ),
    )
    parser.add_argument(
        "--prompt-id",
        action="append",
        default=[],
        help="只渲染指定 prompt_id；可重复传入，默认渲染全部任务",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not 256 <= args.texture_resolution <= 8192:
        parser.error("--texture-resolution 必须在 [256, 8192] 内")
    return args


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.curves, bpy.data.cameras, bpy.data.lights):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def world_bounds(mesh_objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for obj in mesh_objects for corner in obj.bound_box]
    if not corners:
        raise RuntimeError("导入后没有可渲染网格")
    minimum = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    maximum = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return minimum, maximum


def point_camera(camera: bpy.types.Object, target: Vector) -> None:
    direction = target - camera.location
    if direction.length == 0:
        raise RuntimeError("相机位置与目标重合")
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_sun(name: str, rotation: tuple[float, float, float], energy: float) -> None:
    data = bpy.data.lights.new(name=name, type="SUN")
    data.energy = energy
    data.angle = math.radians(25.0)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.rotation_euler = rotation


def configure_scene(resolution: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3.0
    scene.eevee.gtao_factor = 1.2
    scene.eevee.taa_render_samples = 16
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.filepath = ""
    scene.world.color = (0.8, 0.8, 0.8)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def update_status(
    task_dir: Path,
    ok: bool,
    seconds: float,
    error: str | None = None,
    **details,
) -> None:
    path = task_dir / "status.json"
    status = {}
    if path.exists():
        status = json.loads(path.read_text(encoding="utf-8"))
    status.setdefault("stages", {})["render"] = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": ok,
        "seconds": round(seconds, 3),
        **details,
        **({"error": error} if error else {}),
    }
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _imported_meshes(import_path: Path) -> list[bpy.types.Object]:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(import_path))
    return [
        obj
        for obj in bpy.context.scene.objects
        if obj not in before and obj.type == "MESH"
    ]


def _make_single_target(objects: list[bpy.types.Object]) -> bpy.types.Object:
    if not objects:
        raise RuntimeError("最终 model.glb 导入后没有网格")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def _model_has_base_texture(model_path: Path) -> bool:
    clear_scene()
    objects = _imported_meshes(model_path)
    for obj in objects:
        for slot in obj.material_slots:
            material = slot.material
            if not material or not material.use_nodes or not material.node_tree:
                continue
            if any(
                node.type == "TEX_IMAGE" and node.image is not None
                for node in material.node_tree.nodes
            ):
                return True
    return False


def bake_texture_to_model(
    task_dir: Path, texture_resolution: int, export_path: Path | None = None
) -> dict:
    """Bake aligned Pixal material onto a UV atlas for rendering or export."""
    model_path = task_dir / "model.glb"
    export_path = export_path or model_path
    source_path = task_dir / ".texture_source.glb"
    texture_path = task_dir / "model_basecolor.png"
    if not source_path.exists():
        if _model_has_base_texture(model_path):
            return {
                "applied": False,
                "reason": "model_already_has_base_texture",
                "texture": texture_path.name if texture_path.exists() else "embedded",
            }
        raise RuntimeError(
            f"缺少 {source_path}；请先用新版第04阶段重新生成 model.glb"
        )

    clear_scene()
    source_objects = _imported_meshes(source_path)
    if not source_objects:
        raise RuntimeError(f"{source_path} 导入后没有纹理源网格")
    target_objects = _imported_meshes(model_path)
    target = _make_single_target(target_objects)

    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        # 66 degrees avoids highly distorted islands around curved base edges.
        # Per-corner normals are cleared after baking, so this moderate island
        # count still exports compactly instead of splitting every triangle.
        angle_limit=math.radians(66.0),
        island_margin=0.003,
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=True,
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    if not target.data.uv_layers:
        raise RuntimeError("Smart UV Project 没有生成 UV")
    target.data.uv_layers.active.name = "UVMap"

    image = bpy.data.images.new(
        "model_basecolor",
        width=texture_resolution,
        height=texture_resolution,
        alpha=False,
        float_buffer=False,
    )
    image.generated_color = (0.5, 0.5, 0.5, 1.0)
    material = bpy.data.materials.new("BakedPixalMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output_node = nodes.new("ShaderNodeOutputMaterial")
    shader_node = nodes.new("ShaderNodeBsdfPrincipled")
    texture_node = nodes.new("ShaderNodeTexImage")
    texture_node.image = image
    texture_node.interpolation = "Linear"
    nodes.active = texture_node
    texture_node.select = True
    material.node_tree.links.new(texture_node.outputs["Color"], shader_node.inputs["Base Color"])
    material.node_tree.links.new(shader_node.outputs["BSDF"], output_node.inputs["Surface"])
    shader_node.inputs["Roughness"].default_value = 0.75
    target.data.materials.clear()
    target.data.materials.append(material)

    bpy.ops.object.select_all(action="DESELECT")
    for source in source_objects:
        source.select_set(True)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.render.bake.margin = max(8, texture_resolution // 256)
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.use_clear = True
    scene.render.bake.cage_extrusion = 2.0
    scene.render.bake.max_ray_distance = 6.0
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})

    image.filepath_raw = str(texture_path)
    image.file_format = "PNG"
    image.save()
    image.pack()

    # Keep the original per-corner normals while baking so rays at sharp base
    # edges point toward the source instead of diagonally away from it. Clear
    # them only after baking: retaining flat corner normals would force glTF to
    # duplicate nearly every triangle vertex independently of UV seams.
    if target.data.has_custom_normals:
        target.data.free_normals_split()
    for polygon in target.data.polygons:
        polygon.use_smooth = True
    target.data.update()

    # COLOR_0 would multiply the baked base texture in glTF. Remove the old
    # nearest-vertex fallback now that the higher-resolution texture exists.
    for attribute in list(target.data.color_attributes):
        target.data.color_attributes.remove(attribute)
    bpy.ops.object.select_all(action="DESELECT")
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.export_scene.gltf(
        filepath=str(export_path),
        export_format="GLB",
        use_selection=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_colors=False,
        export_image_format="AUTO",
        export_yup=True,
    )
    source_bytes = source_path.stat().st_size
    return {
        "applied": True,
        "method": "smart_uv_selected_to_active_diffuse_bake",
        "texture": texture_path.name,
        "texture_resolution": texture_resolution,
        "source_objects": len(source_objects),
        "target_vertices": len(target.data.vertices),
        "target_faces": len(target.data.polygons),
        "model_bytes": export_path.stat().st_size,
        "export_path": export_path.name,
        "temporary_source_removed": False,
        "temporary_source_bytes": source_bytes,
    }


def render_task(
    task_dir: Path,
    resolution: int,
    texture_resolution: int,
    skip_texture_bake: bool,
    embed_uv_texture: bool,
) -> dict:
    model_path = task_dir / "model.glb"
    if not model_path.exists():
        raise RuntimeError(f"缺少 {model_path}")
    render_model_path = model_path
    temporary_render_model: Path | None = None
    source_path = task_dir / ".texture_source.glb"
    if skip_texture_bake:
        bake_stats = {"applied": False, "reason": "disabled"}
    elif not source_path.exists() and not _model_has_base_texture(model_path):
        # The printable GLB still contains nearest-surface vertex colors. This
        # fallback permits render-only reruns after the large texture source was
        # removed by an earlier successful render.
        bake_stats = {
            "applied": False,
            "reason": "texture_source_missing_vertex_color_fallback",
        }
    else:
        if source_path.exists():
            if embed_uv_texture:
                render_model_path = model_path
            else:
                temporary_render_model = task_dir / ".render_textured.glb"
                render_model_path = temporary_render_model
            bake_stats = bake_texture_to_model(
                task_dir, texture_resolution, export_path=render_model_path
            )
            bake_stats["render_only"] = not embed_uv_texture
            bake_stats["final_model_appearance"] = (
                "uv_texture" if embed_uv_texture else "welded_vertex_color"
            )
        else:
            render_model_path = model_path
            bake_stats = {"applied": False, "reason": "model_already_has_base_texture"}

    try:
        clear_scene()
        configure_scene(resolution)
        bpy.ops.import_scene.gltf(filepath=str(render_model_path))
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        minimum, maximum = world_bounds(mesh_objects)
        center = (minimum + maximum) * 0.5
        extents = maximum - minimum
        largest = max(extents.x, extents.y, extents.z)
        if not math.isfinite(largest) or largest <= 0:
            raise RuntimeError(f"无效模型尺寸: {tuple(extents)}")

        camera_data = bpy.data.cameras.new("StandardCamera")
        camera = bpy.data.objects.new("StandardCamera", camera_data)
        bpy.context.collection.objects.link(camera)
        bpy.context.scene.camera = camera
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = largest * 1.35
        camera.data.clip_start = max(largest / 10_000.0, 0.001)
        camera.data.clip_end = largest * 20.0

        add_sun("Key", (math.radians(35), 0.0, math.radians(-35)), 3.0)
        add_sun("Fill", (math.radians(70), 0.0, math.radians(145)), 1.5)
        add_sun("Rim", (math.radians(25), 0.0, math.radians(210)), 1.0)

        distance = largest * 3.0
        views = {
            "front": center + Vector((0.0, -distance, 0.0)),
            "side": center + Vector((distance, 0.0, 0.0)),
            "top": center + Vector((0.0, 0.0, distance)),
            "perspective": center + Vector((distance, -distance, distance * 0.75)),
        }
        render_dir = task_dir / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        for name, location in views.items():
            camera.location = location
            point_camera(camera, center)
            bpy.context.scene.render.filepath = str(render_dir / f"{name}.png")
            bpy.ops.render.render(write_still=True)
    finally:
        if temporary_render_model is not None:
            temporary_render_model.unlink(missing_ok=True)

    # Only discard the large aligned texture source after all four views were
    # rendered successfully. In the default path model.glb was never replaced.
    if source_path.exists():
        source_path.unlink()
        bake_stats["temporary_source_removed"] = True
    return bake_stats


def main() -> int:
    args = parse_args()
    task_dirs = sorted(path for path in args.work_root.iterdir() if (path / "plan.json").exists())
    if args.prompt_id:
        requested = set(args.prompt_id)
        task_dirs = [path for path in task_dirs if path.name in requested]
        found = {path.name for path in task_dirs}
        missing = sorted(requested - found)
        if missing:
            raise RuntimeError(f"不存在的 prompt_id: {', '.join(missing)}")
    if not task_dirs:
        raise RuntimeError(f"在 {args.work_root} 没有发现任务目录")
    failures: list[str] = []
    required = ("front.png", "side.png", "top.png", "perspective.png")
    for index, task_path in enumerate(task_dirs, start=1):
        render_dir = task_path / "renders"
        pending_texture_source = (
            (task_path / ".texture_source.glb").exists()
            and not args.skip_texture_bake
        )
        if (
            not args.overwrite
            and not pending_texture_source
            and all((render_dir / name).exists() for name in required)
        ):
            print(f"[{index}/{len(task_dirs)}] 跳过已有 {task_path.name}/renders", flush=True)
            continue
        started = time.monotonic()
        print(f"[{index}/{len(task_dirs)}] 渲染 {task_path.name}", flush=True)
        try:
            bake_stats = render_task(
                task_path,
                args.resolution,
                args.texture_resolution,
                args.skip_texture_bake,
                args.embed_uv_texture,
            )
            update_status(
                task_path,
                True,
                time.monotonic() - started,
                texture_bake=bake_stats,
            )
        except Exception as exc:
            failures.append(task_path.name)
            update_status(task_path, False, time.monotonic() - started, str(exc))
            print(f"  失败: {exc}", file=sys.stderr, flush=True)
    if failures:
        print(f"渲染失败 {len(failures)} 题: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"四视图渲染完成，共 {len(task_dirs)} 题。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
