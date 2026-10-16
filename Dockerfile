# UniForge3D — everything baked, nothing downloaded at runtime.
#
# Layout (mirrors Tost-Upscale-3/Dockerfile conventions):
#   * nvidia CUDA devel base (nvcc present: nvdiffrast JIT-compiles at runtime)
#   * pip deps = the exact working set proven on the RTX 3090 host
#     (torch 2.7.1/cu128, NOT setup.sh's torch 2.6.0/cu124)
#   * CUDA extensions compiled from pinned git commits (.deps/ is cloned
#     inside the build, so it stays out of the build context)
#   * all Hub weights baked via bake_models.py into $HF_HUB_CACHE
#   * runtime runs fully offline (HF_HUB_OFFLINE=1)
#
# Build (needs ~60 GB free; final image is ~55 GB):
#   DOCKER_BUILDKIT=1 docker build -t uniforge3d:1.0 .
#
# Run:
#   docker run --gpus all -p 8091:8091 uniforge3d:1.0
#   # Studio UI -> http://<host>:8091/   (upload views, Generate, view .glb)

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV USE_TORCH=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Bake + runtime share one cache; offline at runtime = fail fast, never download.
ENV HF_HOME=/opt/models
ENV HF_HUB_CACHE=/opt/models/hub
ENV HF_HUB_OFFLINE=1
# No GPU is visible at build time, so torch cannot auto-detect the arch list
# (empty list -> IndexError in _get_cuda_arch_flags). Pin it explicitly:
# 8.0 A100, 8.6 RTX 3090, 8.9 RTX 4090, 9.0 H100.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

RUN apt update -y && apt install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip git build-essential cmake ninja-build \
    libgl1 libglib2.0-0 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/uniforge3d

# --- Python deps: the proven working set (torch 2.7.1/cu128) ---
# setuptools/packaging pins match setup.sh; the numpy-2-compatible set
# (pandas, sklearn, pyarrow, soxr, ml_dtypes, wandb, shapely, accelerate,
# bitsandbytes) shadows nothing here — fresh container, installed for the
# versions the code was validated against.
RUN python3 -m pip install --no-cache-dir -U pip && \
    python3 -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128 && \
    python3 -m pip install --no-cache-dir \
        setuptools==80.9.0 wheel ninja cmake==3.31.6 packaging \
        transformers==4.57.3 peft==0.17.1 "huggingface-hub>=0.36,<1" hf_xet \
        "timm>=1.0,<2" "kornia>=0.8,<0.9" "numpy>=2.1,<2.3" \
        "Pillow>=10.4,<12" "safetensors>=0.7,<1" trimesh==4.6.3 \
        "opencv-python-headless>=4.10,<5" "easydict>=1.13,<2" \
        "tqdm>=4.66,<5" "scipy>=1.13,<2" \
        accelerate bitsandbytes pandas scikit-learn pyarrow soxr ml_dtypes \
        wandb shapely && \
    python3 -m pip install --no-cache-dir flash-attn==2.8.1 --no-build-isolation

# --- CUDA extensions, pinned commits from setup.sh ---
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2.git .deps/TRELLIS.2 && \
    git -C .deps/TRELLIS.2 checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd && \
    git -C .deps/TRELLIS.2 submodule update --init --recursive && \
    git clone --recursive https://github.com/JeffreyXiang/CuMesh.git .deps/CuMesh && \
    git -C .deps/CuMesh checkout 12289e1062f0603f2f0d0771b02e1395d247f26f && \
    git -C .deps/CuMesh submodule update --init --recursive && \
    git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git .deps/FlexGEMM && \
    git -C .deps/FlexGEMM checkout 6dd94a859c26ee8246888502eada3dd8ad85532e && \
    git -C .deps/FlexGEMM submodule update --init --recursive && \
    git clone --branch v0.4.0 https://github.com/NVlabs/nvdiffrast.git .deps/nvdiffrast && \
    python3 -m pip install --no-cache-dir ./.deps/nvdiffrast --no-build-isolation && \
    python3 -m pip install --no-cache-dir ./.deps/CuMesh --no-build-isolation && \
    python3 -m pip install --no-cache-dir ./.deps/FlexGEMM --no-build-isolation && \
    python3 -m pip install --no-cache-dir ./.deps/TRELLIS.2/o-voxel --no-build-isolation && \
    python3 -c "import importlib.util; mods=['o_voxel','cumesh','flex_gemm','nvdiffrast','flash_attn']; assert all(importlib.util.find_spec(m) for m in mods), mods; print('EXT_OK')" && \
    rm -rf .deps

# --- gltfpack (meshoptimizer) for game-ready mesh optimization ---
RUN git clone --depth 1 https://github.com/zeux/meshoptimizer.git /tmp/meshoptimizer && \
    cmake -S /tmp/meshoptimizer -B /tmp/meshoptimizer/build -DCMAKE_BUILD_TYPE=Release \
        -DMESHOPT_BUILD_GLTFPACK=ON && \
    cmake --build /tmp/meshoptimizer/build --target gltfpack -j"$(nproc)" && \
    cp /tmp/meshoptimizer/build/gltfpack /usr/local/bin/gltfpack && \
    rm -rf /tmp/meshoptimizer && gltfpack 2>&1 | head -1

# --- Model weights first: bake_models.py is standalone, so baking BEFORE the
# --- app code means later code edits never invalidate this heavy layer.
COPY bake_models.py ./
# Pre-seeded Hub cache for gated repos (DINOv3): plain COPY, no token needed.
COPY docker-hf-cache/. $HF_HUB_CACHE/

# --- Bake all Hub weights into the image (no runtime downloads) ---
# Includes the public camenduru/RMBG-2.0 mirror, so opaque inputs work too.
RUN HF_HUB_OFFLINE=0 python3 bake_models.py && \
    du -sh $HF_HUB_CACHE

# --- App code last: edits here only rebuild this small layer ---
# NOTE: example assets/ are intentionally NOT baked in.
COPY infer.py ui_run.py ui_server.py ./
COPY uniforge3d ./uniforge3d

EXPOSE 8091
CMD ["python3", "ui_server.py", "--port", "8091"]
