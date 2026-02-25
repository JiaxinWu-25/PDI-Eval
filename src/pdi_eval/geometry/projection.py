import numpy as np
from typing import List, Tuple, Optional
from sklearn.linear_model import RANSACRegressor

class ProjectionJudge:
    """透视一致性判别式 (PDI Rule)
    
    核心功能：
    1. 计算 H-X 齐次性残差 (灵魂指标 ε)
    2. RANSAC 鲁棒估算消失点 (Vanishing Point)
    3. 深度-高度演化审计 (1/Z^2 规律校验)
    """
    def __init__(self, cx: float, cy: float):
        """
        cx, cy: 画面主点 (Principal Point)，通常为 (W/2, H/2)
        """
        self.cx = cx
        self.cy = cy

    def calculate_hx_residue(self, h_seq: np.ndarray, x_seq: np.ndarray) -> np.ndarray:
        """实现公式: ε = |(h1/hi) - (x1-cx)/(xi-cx)|
        
        Args:
            h_seq: 像素高度序列 (T,)
            x_seq: 物体质心横坐标序列 (T,)
            
        Returns:
            residues: 全序列的一致性残差序列 (T-1,)
        """
        if len(h_seq) < 2:
            return np.array([0.0])
            
        h1 = max(h_seq[0], 1.0) # 最小高度保护
        x1_relative = x_seq[0] - self.cx
        
        # 结果数组
        residues = []
        
        # 遍历后续帧进行一致性审计
        for i in range(1, len(h_seq)):
            hi = max(h_seq[i], 1.0)
            xi_relative = x_seq[i] - self.cx
            
            # 计算尺度缩放比与位移收敛比
            # 引入 epsilon=1e-3 避免除零，并处理物体穿过画面中心线的情况
            ratio_h = h1 / hi
            
            # 只有当物体不在中心点附近时才计算轨迹收敛比 (x_rel = 0 意味着位移收敛到消失点)
            if np.abs(x1_relative) > 1.0 and np.abs(xi_relative) > 1.0:
                ratio_x = x1_relative / xi_relative
                epsilon = np.abs(ratio_h - ratio_x)
            else:
                # 处于画面中心或初始位置就在中心的物体，主要考察 h(t) 的节奏
                epsilon = 0.0 
                
            residues.append(epsilon)
            
        return np.array(residues)

    def estimate_vanishing_point(self, tracks: np.ndarray) -> Tuple[float, float]:
        """利用多组亚像素轨迹的交点 RANSAC 求解消失点
        
        Args:
            tracks: (N_points, T_frames, 2) 像素轨迹 (来自 Co-Tracker)
            
        Returns:
            (vp_x, vp_y): 估算出的消失点像素坐标
        """
        lines = []
        # 对每个追踪点拟合运动直线方程 y = ax + b
        for i in range(tracks.shape[0]):
            pts = tracks[i] # (T, 2)
            if len(pts) < 5: continue
            
            # 线性回归拟合轨迹 (x 为自变量)
            x = pts[:, 0].reshape(-1, 1)
            y = pts[:, 1]
            
            try:
                # 使用 RANSAC 排除噪声轨迹 (如挥手等非刚体运动点)
                ransac = RANSACRegressor(min_samples=2, residual_threshold=1.0)
                ransac.fit(x, y)
                a = ransac.estimator_.coef_[0]
                b = ransac.estimator_.intercept_
                lines.append((a, b)) # y = ax + b
            except:
                continue
                
        # 求解多条直线的公共交点
        # 此处使用简化的交点计算逻辑：通过直线方程的联立求解
        if len(lines) < 2:
            return self.cx, self.cy # 默认返回主点
            
        # 构造超定方程: a_i * x - 1 * y = -b_i
        A = np.zeros((len(lines), 2))
        B = np.zeros(len(lines))
        for i, (a, b) in enumerate(lines):
            A[i, 0] = a
            A[i, 1] = -1
            B[i] = -b
            
        # 最小二乘求解 VP
        vp, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        return float(vp[0]), float(vp[1])

    def check_scale_evolution(self, h_seq: np.ndarray, z_seq: np.ndarray) -> np.ndarray:
        """验证高度缩减与深度的演化是否符合 1/Z^2 规律
        
        公式: h2 / h1 = z1 / z2 (在针孔模型下)
        """
        # 计算 h * z 的乘积，理论上应为常数 (取决于焦距 f 和物理高度 H)
        # 若乘积波动过大，说明出现了“体积呼吸感”
        h_z_product = h_seq * z_seq
        normalized_product = h_z_product / h_z_product[0] # 首帧归一化
        
        # 返回偏离常数的偏差
        return np.abs(normalized_product - 1.0)
