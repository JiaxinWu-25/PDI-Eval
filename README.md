# PDI-Eval: Perspective Distortion Index for AI Video World Models

**PDI-Eval** 是一个专门用于量化 AI 视频生成模型（如 Sora、Kling、Luma）**空间比例与透视一致性**的自动化评测框架。本项目通过集成 **SAM2**、**Co-Tracker** 和 **Mega-SAM**，构建了一个从 2D 像素追踪到 3D 几何还原的物理审计流水线。

---

## 核心架构

- **perception/**: 提取场景深度、轨迹和分割掩码。
- **geometry/**: 处理齐次坐标投影与相机内外参变换。
- **evaluator/**: 核心审计逻辑，包括尺度校验、轨迹校验与 3D 体积稳定性校验。

---

## 1. 环境要求

本项目对 CUDA 版本极其敏感。为确保 Mega-SAM 底层 C++/CUDA 算子能够成功编译，**必须严格遵守以下版本配比**：

- **Python**: 3.10
- **CUDA Toolkit**: 11.8
- **PyTorch**: 2.1.0

---

## 2. 环境配置

### 2.1 创建 Conda 环境

```bash
conda create -n pdi_eval python=3.10 -y
conda activate pdi_eval

# 安装基础编译工具
conda install -c conda-forge gxx_linux-64=11 gcc_linux-64=11 cmake -y

# 安装匹配 CUDA 11.8 的 PyTorch（严禁直接 pip install torch，必须指定 index-url）
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 安装 CUDA 编译器，确保编译时 nvcc 版本对齐
conda install -c nvidia cuda-toolkit=11.8 -y
```

### 2.2 设置环境变量

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

> 建议将上述三行写入 `~/.bashrc` 或 `~/.zshrc` 以永久生效。

---

## 3. 克隆项目与子模块

```bash
git clone --recursive https://github.com/your_username/PDI-Eval.git
cd PDI-Eval

# 若已克隆主仓库，请初始化子模块
git submodule update --init --recursive
```

---

## 4. 安装依赖

### 4.1 安装 Python 基础依赖

```bash
pip install -r requirements.txt
```

### 4.2 安装 SAM2 与 Co-Tracker

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
pip install git+https://github.com/facebookresearch/co-tracker.git
```

### 4.3 安装 torch-scatter（必须强制指定 pt21 版本）

> **重要**：直接 `pip install torch-scatter` 可能会安装 pt20 旧版本，导致运行时出现 `undefined symbol` 错误。必须使用 `--force-reinstall` 确保安装与 PyTorch 2.1.0 匹配的版本。

```bash
pip install torch-scatter --force-reinstall -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

验证安装：
```bash
python -c "from torch_scatter import scatter_sum; print('torch_scatter OK')"
```

### 4.4 编译 Mega-SAM 底层算子

由于 `base/setup.py` 包含双重调用问题，必须手动拆分安装：

```bash
cd third_party/mega_sam/base

# 1. 编译并安装 droid_backends
cp setup_droid.py setup.py
pip install -e . --no-build-isolation

# 2. 将编译好的 droid_backends.so 复制到 site-packages（关键步骤，不可跳过）
#    编译完成后 .so 文件在当前目录，但 Python 运行时会从 site-packages 加载
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
cp droid_backends*.so "$SITE_PKG/"

# 3. 编译并安装 lietorch
cp setup_lie.py setup.py
pip install -e . --no-build-isolation

# 4. 还原 setup.py
cp setup_org.py setup.py

cd ../../../
```

验证安装：
```bash
python -c "import droid_backends; print('droid_backends OK')"
```

> **说明**：编译过程中出现大量 `-Wdeprecated-declarations`、`-Wreorder` 等警告是正常现象，不影响使用。只有出现 `error:` 才需要处理。

---

## 5. 下载模型权重

请将以下权重文件下载并放入对应目录：

### SAM2
```bash
mkdir -p checkpoints/sam2
# 下载 sam2_hiera_large.pt 和 sam2_hiera_l.yaml
# 官方地址：https://github.com/facebookresearch/segment-anything-2
wget -P checkpoints/sam2 https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

> `sam2_hiera_l.yaml` 配置文件随 SAM2 安装包附带，默认路径为 `checkpoints/sam2/sam2_hiera_l.yaml`。

### Co-Tracker (CoTracker3 Offline)
```bash
mkdir -p checkpoints/tracker
wget -P checkpoints/tracker https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
```

### Mega-SAM: Depth-Anything
```bash
mkdir -p third_party/mega_sam/Depth-Anything/checkpoints
wget -P third_party/mega_sam/Depth-Anything/checkpoints \
  https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth
```

### Mega-SAM: megasam_final.pth
```bash
mkdir -p third_party/mega_sam/checkpoints
# 从 Mega-SAM 官方仓库获取：https://github.com/mega-sam/mega-sam
```

### Mega-SAM: RAFT (用于 CVD 优化)
```bash
pip install gdown
cd third_party/mega_sam/cvd_opt/
gdown 1R8m_jMvCun-N45XkMvHlG0P38kXy-h6I
cd ../../../
```

权重配置文件位于 `configs/default.yaml`，可按需修改路径。

---

## 6. 快速开始

### 使用文字指定目标（推荐，全自动）

```bash
python main.py --input data/your_video.mp4 --text "train"
```

### 使用手动坐标指定目标

```bash
python main.py --input data/your_video.mp4 --points "[[500,500]]"
```

### 完整参数说明

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--input` | 必填 | 输入视频路径 |
| `--text` | None | 目标物体文字描述，使用 Florence-2 自动定位 |
| `--points` | None | 手动点击坐标，格式 `[[x, y]]`，与 `--text` 二选一 |
| `--config` | `configs/default.yaml` | 配置文件路径 |
| `--output_dir` | `results` | 输出目录 |

---

## 7. 调试工具

项目提供两个独立的 Debug 脚本，可单独验证各模块：

```bash
# 验证 SAM2 分割效果
python sam2_check.py --input data/your_video.mp4 --text "train" --output mask_check.mp4

# 验证 Co-Tracker 追踪效果
python cotracker_check.py --input data/your_video.mp4 --text "train" --output cotracker_check.mp4
```

---

## 8. 输出说明

运行完成后，结果保存在 `results/<视频名>/` 目录下：

- `*_pdi_report.txt` — 文字报告，包含 PDI 分数和各维度明细
- `*_scale_traj_errors.png` — 尺度与轨迹残差曲线图
- `*_volume_stability.png` — 3D 体积稳定性折线图
- `*_annotated.mp4` — 叠加了透视线和追踪点的标注视频

---

## 9. 项目结构

```
PDI-Eval/
├── checkpoints/              # 权重文件 (.pt, .pth)
├── configs/                  # 配置文件
│   └── default.yaml
├── data/                     # 测试视频
├── third_party/              # Git 子模块
│   └── mega_sam/             # Mega-SAM 仓库
├── src/
│   └── pdi_eval/
│       ├── pipeline.py       # 总控：管理模型加载与数据流转
│       ├── perception/       # 感知层：SAM2 / CoTracker / MegaSAM 封装
│       ├── geometry/         # 相机投影与坐标变换
│       ├── evaluator/        # 审计逻辑：尺度、轨迹、体积
│       ├── metrics/          # PDI 指标合成
│       ├── data/             # 缓存管理
│       └── utils/            # 日志与可视化
├── main.py                   # 命令行入口
├── sam2_check.py             # SAM2 调试脚本
├── cotracker_check.py        # Co-Tracker 调试脚本
├── requirements.txt
└── README.md
```

---

## 10. 评分标准

| PDI 分数 | 等级 | 含义 |
| :--- | :--- | :--- |
| < 0.1 | A | 物理逻辑严丝合缝 |
| 0.1 – 0.3 | B | 存在轻微几何抖动 |
| 0.3 – 0.6 | C | 明显透视幻觉/滑步 |
| > 0.6 | F | 物理逻辑彻底崩溃 |

---

## 11. 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@article{yourname2025pdi,
  title={PDI-Eval: Quantitative Auditing of Perspective Consistency in Video World Models},
  author={Your Name and Collaborators},
  journal={arXiv preprint},
  year={2025}
}
```

---

**License**: MIT (Core Code) / Apache-2.0 (Mega-SAM)

### 三、 如果 Mega-SAM 重建坏了怎么办？（核心风险控管）

这是一个非常现实的问题：**如果“尺子”本身就是弯的，怎么量出视
频的对错？** 

重建质量差（$Z$ 值乱跳）确实是目前 3D 视觉的痛点。为了保证你
的实验效果，你需要引入 **“鲁棒性审计” (Robustness Audit)
**：

#### 1. 引入置信度图 (Confidence Map)
*   Mega-SAM 会输出一个**置信度评分**。
*   **策略**：在计算 PDI 指标时，如果某几帧的重建置信度低于
阈值，**直接剔除这些帧**，不计入评分。这能保证你的 PDI 分数
是由“可靠的 3D 数据”算出来的。

#### 2. 时域平滑过滤 (Temporal Smoothing)
*   **策略**：物理世界的深度 $Z$ 不可能在 1/30 秒内突变 5 
米。如果算出来的 $Z$ 序列波动率（一阶差分）超过了物理极限，我
们就判定为**重建失败**而非视频幻觉。


## 有关运行一次的时间，有无要求



## 代码要做到什么程度？


## 要测试多少个视频模型？用开源的吗？闭源的模型是直接接入
api，输入prompt让它生成，再进行检测
seekdance

测试几类视频，每类视频有几个video,(一共二三十个min)
test as much as you can

## 有关timeline，3月5号截止，什么时候将初稿写好比较合适呢？
overleaf


## real和fake的对比（明天完成）
目前可以先测三四个。

要理解 SAM2 自动识别物体的逻辑，我们需要区分两个层面：
**SAM2 自身的“分出所有东西”能力**，以及我们通过插件（如 
Florence-2）实现的**“理解名字并识别”能力**。

简单来说，SAM2 本质上是一把**“万能剪刀”**，它知道哪里是物体
的边界，但它默认并不知道这个物体叫什么。
### 1. 语义识别逻辑：Florence-2 / Grounding DINO + SAM2
这是你目前代码中使用的逻辑（**最聪明、最常用的自动化方式
**）。
*   **逻辑链条**：**看到文字 -> 找到框 -> 传给 SAM2**。
*   **具体步骤**：
    1.  **文本理解**：你输入 "train"（火车）。
    2.  **目标检测（检测器）**：Florence-2 或 Grounding 
    DINO 这种“大脑”模型先扫描图片。它们经过了海量图文对训
    练，知道“火车”长什么样。
    3.  **坐标锁定**：检测器在图上画一个矩形框（Bounding 
    Box），确定火车的范围。
    4.  **提示点生成**：我们取这个矩形框的**中心点**，作为
    “提示点（Prompt Point）”传给 SAM2。
    5.  **精准分割**：SAM2 接收到点后，利用其强大的边缘感知
    能力，把火车的精确轮廓抠出来。
### 2. 几何发现逻辑：SAM2 的 "Everything" 模式
这是 SAM2 官方自带的自动化逻辑。

*   **逻辑链条**：**地毯式搜索 -> 过滤 -> 产出实例**。
*   **具体步骤**：
    1.  **全图撒点**：模型在整张图片上均匀地铺设一个点阵（例
    如 32x32 个点）。
    2.  **并行分割**：对每一个点，SAM2 都会尝试预测它所属的
    物体掩码（Mask）。
    3.  **去重与过滤 (NMS)**：因为一个物体上会有多个点，会
    产生重复的 Mask。模型通过“交并比 (IoU)”和“稳定性分数”剔
    除重叠的和质量差的掩码。
    4.  **结果**：它会把图里所有的石头、草地、火车、云朵全部
    标出来。
*   **缺点**：它分出了 100 个物体，但它不知道哪个是你要审计
的“火车”。

---
### 3. 时空连续逻辑：SAM2 的视频记忆机制
这是 SAM2 为什么能“自动”在视频里一直盯着某个物体的核心。
### 4. 总结：在你的 PDI-Eval 项目中，逻辑是怎样的？

你现在的 **全自动版本**（使用 `--text "train"`）逻辑如下：

1.  **Florence-2 (大脑)**：在视频第 1 帧搜索符合 "train" 
语义的物体。
2.  **转化为 Prompt**：将找到的物体中心坐标 `[x, y]` 给 
SAM2。
3.  **SAM2 (剪刀+记忆)**：
    *   在第 1 帧抠出火车轮廓。
    *   通过**记忆银行**自动在接下来的 400 多帧里“粘”住这辆
    车。
    *   **自动提取**：代码从每一帧生成的 Mask 中自动计算高
    度 $h$ 和质心 $x$。

### 💡 为什么这种“组合逻辑”对你的研究最有利？
因为你要做 **Benchmark (基准测试)**。
*   如果靠人手动点，每次点的位置稍有偏差，PDI 分数就会变，实
验就不可复现。
*   **全自动语义逻辑** 保证了：只要输入 "train"，系统每次都
会以同样的方式锁定物体中心，你的 PDI 评估结果才是**客观、可
复现、具备学术公信力**的。