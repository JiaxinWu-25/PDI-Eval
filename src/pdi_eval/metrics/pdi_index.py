import numpy as np
from typing import Dict, List, Any, Union

class PDIIndexCalculator:
    """PDI 总分合成模块 (The Judge)
    
    核心功能：
    1. 指标聚合：接收尺度、轨迹和 3D 稳定性的残差。
    2. 鲁棒性处理：兼容输入为序列(Array)或预计算好的 RMSE(Scalar)。
    3. 等级判定：提供直观的物理真实度等级。
    """
    def __init__(self, w_scale: float = 0.4, w_traj: float = 0.4, w_vol: float = 0.2):
        self.weights = {
            "scale": w_scale,
            "trajectory": w_traj,
            "volume": w_vol
        }

    def _ensure_scalar(self, val: Union[float, np.ndarray]) -> float:
        """辅助函数：确保输入被转化为标量误差值"""
        if isinstance(val, (np.ndarray, list)):
            if len(val) == 0: return 0.0
            # 如果传入的是序列，则计算其 RMSE；如果已经是单个值的数组，则取该值
            if len(val) > 1:
                return float(np.sqrt(np.mean(np.square(val))))
            return float(np.array(val).flatten()[0])
        return float(val)

    def compute_pdi(self, scale_errors: Union[float, np.ndarray], 
                    trajectory_errors: Union[float, np.ndarray], 
                    volume_cv: float) -> Dict[str, Any]:
        """合成最终 PDI 分数与明细
        
        Args:
            scale_errors: 尺度残差序列或其 RMSE 值
            trajectory_errors: 轨迹残差序列或其 RMSE 值
            volume_cv: 3D 空间的变异系数
        """
        # 1. 统一转化为 RMSE 标量，增强协同鲁棒性
        rmse_scale = self._ensure_scalar(scale_errors)
        rmse_traj = self._ensure_scalar(trajectory_errors)
        
        # 2. 加权合成最终 PDI 分数
        pdi_score = (self.weights["scale"] * rmse_scale + 
                     self.weights["trajectory"] * rmse_traj + 
                     self.weights["volume"] * volume_cv)
        
        # 3. 判定等级
        grade = self.assign_grade(pdi_score)
        
        return {
            "pdi_score": round(pdi_score, 4),
            "grade": grade,
            "breakdown": {
                "scale_component": round(rmse_scale, 4),
                "traj_component": round(rmse_traj, 4),
                "volume_component": round(volume_cv, 4)
            }
        }

    def assign_grade(self, score: float) -> str:
        """物理真实度等级判定标准"""
        if score < 0.1: return "A (Physical Realism) - 物理逻辑严丝合缝"
        if score < 0.3: return "B (Minor Jitter) - 存在轻微几何抖动"
        if score < 0.6: return "C (Obvious Distortion) - 明显透视幻觉/滑步"
        return "F (Geometric Failure) - 物理逻辑彻底崩溃"
