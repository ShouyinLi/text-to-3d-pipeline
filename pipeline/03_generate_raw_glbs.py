#!/usr/bin/env python3
"""Load Pixal3D once and generate raw.glb sequentially for all references.

An optional print-oriented distance-field pass runs before UV unwrapping and
texture baking.  Its millimetre parameters are interpreted using each task's
``plan.json`` target_longest_mm; the raw GLB itself remains in Pixal3D units and
stage 04 still performs the final physical scaling.
"""

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
    discover_tasks,
    load_json,
    task_dir,
    update_status,
)


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIXAL_ROOT = Path(os.environ.get("PIXAL_ROOT", str(BUNDLE_ROOT / "Pixal3D-master")))
DEFAULT_MODEL = Path(os.environ.get("PIXAL_MODEL", str(BUNDLE_ROOT / "weights" / "Pixal3D")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--expected-count", type=int, default=40)
    parser.add_argument("--pixal-root", type=Path, default=DEFAULT_PIXAL_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--resolution", type=int, choices=(1024, 1536), default=1536)
    parser.add_argument(
        "--gpus",
        default="0,1",
        help="两张物理 GPU；第一张用于 Pixal3D 推理，第二张用于 CuMesh/GLB 导出",
    )
    parser.add_argument("--fov", type=float, default=-1.0, help="手动水平FOV，弧度；负数使用MoGe估计")
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument(
        "--printable",
        action="store_true",
        help="纹理烘焙前执行距离场重建、薄壁加厚和非流形清理",
    )
    parser.add_argument(
        "--min-wall-mm",
        type=float,
        default=1.2,
        help="可打印模式的目标最小壁厚，默认1.2mm",
    )
    parser.add_argument(
        "--print-voxel-mm",
        type=float,
        default=0.6,
        help="可打印模式距离场体素尺寸，默认0.6mm",
    )
    parser.add_argument(
        "--min-component-faces",
        type=int,
        default=50,
        help="删除重建后小于此面数的组件；0表示保留全部，默认50",
    )
    parser.add_argument(
        "--prompt-id",
        action="append",
        default=[],
        help="只生成指定 prompt_id 的 raw.glb；可重复传入，默认处理全部任务",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def export_glb_on_gpu(
    mesh,
    pipeline,
    actual_resolution: int,
    export_device_index: int,
    torch_module,
    o_voxel_module,
    printable: bool,
):
    """Copy export inputs to a dedicated GPU and run the full CuMesh stage there."""
    export_device = torch_module.device(f"cuda:{export_device_index}")
    with torch_module.cuda.device(export_device_index):
        vertices = mesh.vertices.to(export_device)
        faces = mesh.faces.to(export_device)
        attrs = mesh.attrs.to(export_device)
        coords = mesh.coords.to(export_device)
        torch_module.cuda.synchronize(export_device)
        return o_voxel_module.postprocess.to_glb(
            vertices=vertices,
            faces=faces,
            attr_volume=attrs,
            coords=coords,
            attr_layout=pipeline.pbr_attr_layout,
            grid_size=actual_resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=1_000_000,
            texture_size=4096,
            # Printable mode has already been distance-field remeshed. Use the
            # standard branch for duplicate/non-manifold cleanup instead of
            # rebuilding the surface a second time.
            remesh=not printable,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=True,
        )


def main() -> int:
    args = parse_args()
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise PipelineError("--gpus 必须指定两张不同的 GPU，例如 0,1")
    if not args.pixal_root.is_dir():
        raise PipelineError(f"Pixal3D 代码目录不存在: {args.pixal_root}")
    if not args.model.is_dir():
        raise PipelineError(f"Pixal3D 模型目录不存在: {args.model}")

    tasks = discover_tasks(args.data_root, args.expected_count)
    if args.prompt_id:
        requested = set(args.prompt_id)
        tasks = [task for task in tasks if task["prompt_id"] in requested]
        found = {task["prompt_id"] for task in tasks}
        missing = sorted(requested - found)
        if missing:
            raise PipelineError(f"不存在的 prompt_id: {', '.join(missing)}")

    pending: list[tuple[dict, dict, Path]] = []
    for task in tasks:
        output_dir = task_dir(args.work_root, task["prompt_id"])
        plan_path = output_dir / "plan.json"
        reference_path = output_dir / "reference.png"
        if not plan_path.exists():
            raise PipelineError(f"缺少 {plan_path}")
        if not reference_path.exists():
            raise PipelineError(f"缺少 {reference_path}，请先运行 reference 阶段")
        if args.overwrite or not (output_dir / "raw.glb").exists():
            pending.append((task, load_json(plan_path), reference_path))

    if not pending:
        print(f"全部 {len(tasks)} 个 raw.glb 已存在，无需加载模型。")
        return 0

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("ATTN_BACKEND", "flash_attn")

    sys.path.insert(0, str(args.pixal_root.resolve()))

    import numpy as np
    import torch
    from PIL import Image

    import inference as pixal_inference
    import o_voxel

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise PipelineError(f"Pixal3D 双卡模式需要恰好两张可见 GPU，当前为 {torch.cuda.device_count()} 张")

    inference_device_index = 0
    export_device_index = 1
    inference_device = torch.device(f"cuda:{inference_device_index}")
    torch.cuda.set_device(inference_device_index)

    print(
        f"加载 Pixal3D {args.model}，待处理 {len(pending)}/{len(tasks)} 题\n"
        f"物理 GPU {gpu_ids[0]} -> cuda:0：Pixal3D 推理；"
        f"物理 GPU {gpu_ids[1]} -> cuda:1：CuMesh/GLB 导出"
    )
    pipeline = pixal_inference.init_pipeline(
        str(args.model), device=str(inference_device), low_vram=args.low_vram
    )
    moge_model = None
    if args.fov <= 0:
        print("在 CPU 加载一次 MoGe-2；每题估计相机时临时移入 GPU。")
        moge_model = pixal_inference.load_moge_model(device="cpu")

    failures: list[str] = []
    for index, (task, plan, reference_path) in enumerate(pending, start=1):
        prompt_id = task["prompt_id"]
        output_dir = task_dir(args.work_root, prompt_id)
        output_path = output_dir / "raw.glb"
        temp_path = output_dir / "_pixal_camera_input.png"
        started = time.monotonic()
        image_preprocessed = None
        mesh_list = None
        latent_outputs = None
        mesh = None
        glb = None
        print(f"[{index}/{len(pending)}] 生成 {prompt_id}/raw.glb", flush=True)
        try:
            torch.cuda.set_device(inference_device_index)
            with Image.open(reference_path) as source_image:
                image_preprocessed = pipeline.preprocess_image(source_image.convert("RGB"))
            image_preprocessed.save(temp_path)

            if args.fov > 0:
                grid_point = torch.tensor([-1.0, 0.0, 0.0])
                distance = pixal_inference.distance_from_fov(
                    args.fov,
                    grid_point,
                    torch.tensor([0, 511]),
                    1.0,
                    512,
                )["distance_from_x"]
                camera_params = {
                    "camera_angle_x": float(args.fov),
                    "distance": distance,
                    "mesh_scale": 1.0,
                }
            else:
                assert moge_model is not None
                moge_model = moge_model.to(inference_device)
                camera_params = pixal_inference.get_camera_params_wild_moge(
                    str(temp_path),
                    moge_model,
                    device=str(inference_device),
                    mesh_scale=1.0,
                    extend_pixel=0,
                    image_resolution=512,
                )
                moge_model = moge_model.cpu()
                torch.cuda.empty_cache()

            seed = int(plan["seed"])
            torch.manual_seed(seed)
            mesh_list, latent_outputs = pipeline.run(
                image_preprocessed,
                camera_params=camera_params,
                seed=seed,
                sparse_structure_sampler_params={
                    "steps": 12,
                    "guidance_strength": 7.5,
                    "guidance_rescale": 0.7,
                    "rescale_t": 5.0,
                },
                shape_slat_sampler_params={
                    "steps": 12,
                    "guidance_strength": 7.5,
                    "guidance_rescale": 0.5,
                    "rescale_t": 3.0,
                },
                tex_slat_sampler_params={
                    "steps": 12,
                    "guidance_strength": 1.0,
                    "guidance_rescale": 0.0,
                    "rescale_t": 3.0,
                },
                preprocess_image=False,
                return_latent=True,
                pipeline_type=f"{args.resolution}_cascade",
                max_num_tokens=49152,
            )
            actual_resolution = latent_outputs[2]
            mesh = mesh_list[0]
            target_mm = float(plan["target_longest_mm"])
            if args.printable:
                print(
                    "  执行可打印重建："
                    f"最长边={target_mm:g}mm，最小壁厚={args.min_wall_mm:g}mm，"
                    f"体素={args.print_voxel_mm:g}mm",
                    flush=True,
                )
                mesh.vertices, mesh.faces = pixal_inference.make_printable_mesh(
                    mesh.vertices,
                    mesh.faces,
                    target_size_mm=target_mm,
                    min_wall_mm=args.min_wall_mm,
                    voxel_mm=args.print_voxel_mm,
                    min_component_faces=args.min_component_faces,
                )
            print(
                f"  将 CuMesh/GLB 导出切换到物理 GPU {gpu_ids[1]} (cuda:1)",
                flush=True,
            )
            glb = export_glb_on_gpu(
                mesh=mesh,
                pipeline=pipeline,
                actual_resolution=actual_resolution,
                export_device_index=export_device_index,
                torch_module=torch,
                o_voxel_module=o_voxel,
                printable=args.printable,
            )
            pixal_rotation = np.array(
                [
                    [-1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, -1, 0, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float64,
            )
            glb.apply_transform(pixal_rotation)
            output_dir.mkdir(parents=True, exist_ok=True)
            glb.export(output_path, extension_webp=True)
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise PipelineError("Pixal3D 未写出有效 raw.glb")
            update_status(
                args.work_root,
                prompt_id,
                "raw_glb",
                ok=True,
                seconds=round(time.monotonic() - started, 3),
                model=str(args.model),
                resolution=args.resolution,
                actual_resolution=int(actual_resolution),
                seed=seed,
                inference_gpu=gpu_ids[0],
                export_gpu=gpu_ids[1],
                printable=args.printable,
                target_longest_mm=target_mm,
                min_wall_mm=args.min_wall_mm if args.printable else None,
                print_voxel_mm=args.print_voxel_mm if args.printable else None,
                min_component_faces=(
                    args.min_component_faces if args.printable else None
                ),
                bytes=output_path.stat().st_size,
            )
        except Exception as exc:
            failures.append(prompt_id)
            update_status(
                args.work_root,
                prompt_id,
                "raw_glb",
                ok=False,
                seconds=round(time.monotonic() - started, 3),
                error=str(exc),
            )
            print(f"  失败: {exc}", file=sys.stderr, flush=True)
        finally:
            # Drop references before empty_cache; especially important after a
            # failed CuMesh call because the next task otherwise inherits its peak.
            glb = None
            mesh = None
            mesh_list = None
            latent_outputs = None
            image_preprocessed = None
            for device_index in (export_device_index, inference_device_index):
                try:
                    with torch.cuda.device(device_index):
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            torch.cuda.set_device(inference_device_index)
            temp_path.unlink(missing_ok=True)

    if failures:
        print(f"raw GLB 阶段失败 {len(failures)} 题: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"raw GLB 阶段完成，共 {len(tasks)} 题。进程退出后释放两张 GPU。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
