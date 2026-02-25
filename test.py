import numpy as np

# 加载缓存文件
data = np.load("output/cache/demo_mega_sam.npz", allow_pickle=True)

# 查看里面存的键
print("Cache Keys:", data.files)

# 查看深度 Z 的变化
z_values = data['depth_z']
print("Depth Z sequence:", z_values)

# 判定：如果 Z 序列是一串非常平滑、死板的等差数列，说明是 Mock 数据。
# 如果 Z 序列带有细微波动、或者是从 3D 优化出来的非线性数值，说明模型在工作。