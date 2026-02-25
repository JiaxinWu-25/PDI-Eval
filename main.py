import argparse
import yaml
import sys
import os
from pathlib import Path

# 确保能搜到 src/ 目录下的 pdi_eval 包
sys.path.append(os.path.join(os.getcwd(), "src"))

from pdi_eval.pipeline import PDIEvaluationPipeline
from pdi_eval.utils.logger import pdi_logger
from pdi_eval.utils.visualizer import EvidenceVisualizer

def main():
    parser = argparse.ArgumentParser(description="PDI-Eval CLI Runner")
    parser.add_argument("--input", type=str, required=True, help="视频文件路径")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--points", type=str, default="[[500,500]]", help="SAM2 初始点击点坐标")
    args = parser.parse_args()

    # 1. 验证配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        pdi_logger.error(f"Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. 初始化流水线
    pipeline = PDIEvaluationPipeline(config=config)
    
    # 3. 运行审计
    pdi_logger.info(f"🚀 开始审计视频: {args.input}")
    try:
        # 安全解析坐标点
        click_points = eval(args.points)
    except Exception as e:
        pdi_logger.error(f"Invalid points format: {args.points}. Error: {e}")
        return

    # 获取执行结果
    report = pipeline.run(video_path=args.input, click_points=click_points)

    # 4. 生成可视化物证
    video_stem = Path(args.input).stem
    output_path = Path(args.output_dir) / video_stem
    output_path.mkdir(parents=True, exist_ok=True)
    
    viz = EvidenceVisualizer(output_dir=str(output_path))
    
    # A. 绘制尺度与轨迹残差图
    viz.draw_error_curves(
        report['breakdown']['scale_history'],
        report['breakdown']['traj_history'],
        video_stem
    )
    
    # B. 绘制 3D 体积变化折线图
    viz.draw_volume_stability(
        report['breakdown']['volume_history'],
        video_stem
    )
    

    # B. 保存标注视频
    annotated_path = pipeline.get_annotated_video()

    # --- 新增：保存文本报告 ---
    report_txt_path = output_path / f"{video_stem}_pdi_report.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write("        PDI-Eval Final Audit Report\n")
        f.write("="*50 + "\n")
        f.write(f"Video Source:  {args.input}\n")
        f.write(f"Target Points: {args.points}\n")
        f.write("-" * 50 + "\n")
        f.write(f"FINAL PDI SCORE: {report['pdi_score']:.4f}\n")
        f.write(f"OVERALL GRADE:   {report['grade']}\n")
        f.write("-" * 50 + "\n")
        f.write("INDICATOR BREAKDOWN:\n")
        f.write(f" - Scale Component (1/Z Law):    {report['breakdown'].get('scale_component', 0):.4f}\n")
        f.write(f" - Trajectory Component (H-X):   {report['breakdown'].get('traj_component', 0):.4f}\n")
        f.write(f" - Volume Component (Stability): {report['breakdown'].get('volume_component', 0):.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Results generated at: {output_path}\n")
        f.write("="*50 + "\n")

    pdi_logger.info(f"✅ 审计完成！最终 PDI 分数: {report['pdi_score']:.4f} [{report['grade']}]")
    pdi_logger.info(f"📄 文本报告已生成: {report_txt_path}")
    pdi_logger.info(f"📁 详细结果与物证已存至: {output_path}")

if __name__ == "__main__":
    main()