# Reproduction

## 1. 硬件和系统

建议配置：

- Linux x86_64
- 两张可见的 NVIDIA CUDA GPU；当前流程要求恰好两张 GPU
- 每张 GPU 建议至少 48 GiB 显存
- Python 3.10
- Blender 3.6
- 至少约 180 GB 可用磁盘空间，用于模型权重、缓存和中间文件

Pixal3D、NATTEN、CuMesh 等 CUDA 扩展必须与本机 CUDA、PyTorch 和 GPU 架构匹配。

## 2. 获取第三方源码

在仓库根目录执行：

```bash
git clone --depth 1 https://github.com/TencentARC/Pixal3D.git Pixal3D-master
git clone --depth 1 https://github.com/valeoai/NAF.git NAF
git clone --recursive --depth 1 https://github.com/microsoft/TRELLIS.2.git TRELLIS.2
```

第三方项目不纳入本仓库，由各自许可证约束。

## 3. Python 环境

环境版本记录在 `env/`。推荐使用三个隔离环境：

```bash
conda create -n text3d-plan python=3.10 pip -y
conda create -n text3d-image python=3.10 pip -y
conda create -n text3d-pixal python=3.10 pip -y

conda run -n text3d-plan \
  python -m pip install -r env/planner-freeze.txt

conda run -n text3d-image \
  python -m pip install -r env/qwen-image-freeze.txt
```

Pixal3D 环境需要先按照 `Pixal3D-master/README.md` 安装 TRELLIS.2、NATTEN、CuMesh、O-Voxel 等 CUDA 扩展，再将 `env/pixal3d-freeze.txt` 作为版本参考。

## 4. 下载模型

```bash
python -m pip install -U "huggingface_hub[cli]"
mkdir -p weights/torch_home/hub/checkpoints

hf download Qwen/Qwen3.8-27B \
  --local-dir weights/Qwen3.8-27B

hf download Qwen/Qwen-Image-2512 \
  --local-dir weights/Qwen-Image-2512

hf download TencentARC/Pixal3D \
  --local-dir weights/Pixal3D

hf download camenduru/dinov3-vitl16-pretrain-lvd1689m \
  --local-dir weights/dinov3-vitl16

hf download briaai/RMBG-2.0 \
  --local-dir weights/RMBG-2.0

hf download Ruicheng/moge-2-vitl \
  --local-dir weights/moge-2-vitl

curl -fL \
  https://github.com/valeoai/NAF/releases/download/model/naf_release.pth \
  -o weights/torch_home/hub/checkpoints/naf_release.pth
```

下载及使用模型前请分别确认其许可证，特别是非商业使用限制。

## 5. 准备输入

```bash
mkdir -p input output
cp examples/prompts.example.jsonl input/prompts.jsonl
```

每行必须是一个 JSON 对象，至少包含：

- `prompt_id`：唯一任务 ID
- `prompt`：文本描述

## 6. 完整执行命令

将 Blender 路径替换为本机实际路径：

```bash
REPO_ROOT="$(pwd)"
CONDA_BASE="$(conda info --base)"

CUDA_VISIBLE_DEVICES=0,1 \
CUBE3D_PYTHON="$CONDA_BASE/envs/text3d-plan/bin/python" \
QWEN_IMAGE_PYTHON="$CONDA_BASE/envs/text3d-image/bin/python" \
PIXAL_PYTHON="$CONDA_BASE/envs/text3d-pixal/bin/python" \
PREPARE_PYTHON="$CONDA_BASE/envs/text3d-plan/bin/python" \
BLENDER_BIN="/path/to/blender-3.6/blender" \
bash pipeline/run_infer.sh \
  "$REPO_ROOT/input/prompts.jsonl" \
  "$REPO_ROOT/output"
```

输出目录包含 GLB 模型、元数据、渲染图、`manifest.json`、阶段日志和中间结果。

## 7. 分阶段执行

```bash
bash pipeline/run_pipeline.sh plans-new
bash pipeline/run_pipeline.sh references
bash pipeline/run_pipeline.sh glbs
bash pipeline/run_pipeline.sh postprocess
bash pipeline/run_pipeline.sh render
bash pipeline/run_pipeline.sh package
```

相关路径、Python 解释器、模型和处理参数均可通过 `run_pipeline.sh` 顶部列出的环境变量覆盖。
