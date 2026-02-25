# PDI-Eval: Perspective Distortion Index for AI Video World Models

**PDI-Eval** 是一个专门用于量化 AI 视频生成模型（如 Sora, Kling, Luma）**空间比例与透视一致性**的自动化评测框架。本项目通过集成 **SAM2**、**Co-Tracker** 和 **Mega-SAM**，构建了一个从 2D 像素追踪到 3D 几何还原的物理审计流水线。

---

## 🛠 1. 环境配置 (Environment Setup)

本项目对 CUDA 版本极其敏感。为了确保 **Mega-SAM** 的底层 C++/CUDA 算子能够成功编译，请务必严格遵守以下版本配比：
*   **Python**: 3.10
*   **CUDA (Toolkit & Runtime)**: 11.8
*   **PyTorch**: 2.1.0

### 1.1 创建 Conda 环境
```bash
# 创建并激活环境
conda create -n pdi_eval python=3.10 -y
conda activate pdi_eval

# 安装基础构建工具
conda install -c conda-forge gxx_linux-64=11 gcc_linux-64=11 cmake -y

# 安装匹配 CUDA 11.8 的 PyTorch 栈 (非常重要，严禁直接 pip install torch)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 安装环境内的 CUDA 编译器 (确保编译时 nvcc 版本对齐)
conda install -c nvidia cuda-toolkit=11.8 -y
```

### 1.2 设置环境变量
为了确保编译脚本能找到正确的 CUDA 路径，请执行：
```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

---

## 📂 2. 克隆项目与子模块 (Submodules)

本项目引用了多个外部仓库，请使用递归克隆：
```bash
git clone --recursive https://github.com/your_username/PDI-Eval.git
cd PDI-Eval

# 如果已经克隆了主仓库，请初始化子模块
git submodule update --init --recursive
```

---

## 🏗 3. 编译与安装 (Installation)

### 3.1 安装 SAM2 & Co-Tracker
```bash
# 安装 SAM2
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# 安装 Co-Tracker
pip install git+https://github.com/facebookresearch/co-tracker.git

# 安装其他基础依赖
pip install -r requirements.txt
```

### 3.2 编译 Mega-SAM 底层算子 (手术式拆分安装)
由于 Mega-SAM 的 `base/setup.py` 包含双重调用 Bug，必须手动拆分安装。我们已在 `third_party/mega_sam/base/` 预置了拆分脚本：

```bash
cd third_party/mega_sam/base

# 备份原文件
mv setup.py setup_org.py

# 1. 安装 droid_backends
cp setup_droid.py setup.py
pip install -e . --no-build-isolation

# 2. 安装 lietorch
cp setup_lie.py setup.py
pip install -e . --no-build-isolation

# 还原
mv setup_org.py setup.py
cd ../../../

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

---

## 🧠 4. 模型权重 (Checkpoints)

请下载以下权重并放入 `checkpoints/` 对应目录（详细链接见项目 Wiki）：
*   `sam2_hiera_large.pt` -> `checkpoints/sam2/`
*   `cotracker3.pth` -> `checkpoints/tracker/`
(
    # 1. 进入目标文件夹
mkdir -p checkpoints/tracker
cd checkpoints/tracker

# 2. 下载离线版模型 (最适合 PDI 审计)
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth

cd ../..
)
<!-- *   `dust3r_vitl14_target_space.pth` -> `checkpoints/geometry/` -->
*   `depth_anything_vitl14.pth` -> `third_party/mega_sam/Depth-Anything/checkpoints/` （wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth）
*   `raft-things.pth` -> `third_party/mega_sam/cvd_opt/` 
（pip install gdown

cd third_party/mega_sam/cvd_opt/

gdown 1R8m_jMvCun-N45XkMvHlG0P38kXy-h6I）
---