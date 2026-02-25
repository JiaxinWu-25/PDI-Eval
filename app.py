import gradio as gr
import yaml
import numpy as np
import sys
import os
from pathlib import Path

# 确保能搜到 src/ 目录下的 pdi_eval 包
sys.path.append(os.path.join(os.getcwd(), "src"))

from pdi_eval.pipeline import PDIEvaluationPipeline
from pdi_eval.utils.logger import pdi_logger

# 预加载配置
config_path = "configs/default.yaml"
if not os.path.exists(config_path):
    # 极简默认配置
    config = {
        'sam_ckpt': "checkpoints/sam2/sam2_hiera_large.pt",
        'sam_cfg': "sam2_hiera_l.yaml",
        'engine_3d': 'mega_sam',
        'weights': {'w_scale': 0.4, 'w_trajectory': 0.4, 'w_volume': 0.2}
    }
else:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

# 初始化总控逻辑
pipeline = PDIEvaluationPipeline(config=config)

def predict_on_click(video_input, evt: gr.SelectData):
    if video_input is None:
        return None, None, 0.0, "Please upload a video first."
        
    # 1. 获取用户点击坐标
    # Gradio 的 SelectData 包含了点击的位置
    click_point = [[evt.index[0], evt.index[1]]]
    pdi_logger.info(f"User clicked at: {click_point}")
    
    # 2. 运行后端 Pipeline
    # 由于 app.py 主要是展示，建议这里调用 pipeline 的 cache 机制
    try:
        report = pipeline.run(video_path=video_input, click_points=click_point)
        
        # 3. 提取可视化组件
        # 包含叠加了消失点和透视线的视频
        result_video = pipeline.get_annotated_video() 
        # 生成误差曲线的 Plot 对象 (Gradio 支持路径)
        error_plot = pipeline.get_error_plot() 
        
        return (
            result_video,
            error_plot,
            report['pdi_score'],
            report['grade']
        )
    except Exception as e:
        pdi_logger.error(f"Pipeline failed: {e}")
        return None, None, 0.0, f"Error: {e}"

# 构建 UI 界面
with gr.Blocks(title="PDI-Eval Space") as demo:
    gr.Markdown("# 🕵️ PDI-Eval: World Model Perspective Auditor")
    gr.Markdown("评估 AI 视频生成模型（如 Sora, Kling, Luma）的**物理一致性与透视稳定性**。")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Step 1: Upload Video")
            gr.Markdown("**Step 2: Click the object in the preview to audit**")
        
        with gr.Column():
            video_output = gr.Video(label="Audit Visualization (Perspective Lines)")
    
    with gr.Row():
        plot_output = gr.Image(label="Residual Analysis (Geometric Deviations)") # 改为 Image 显示保存的 png
        with gr.Column():
            pdi_score = gr.Number(label="Final PDI Index")
            verdict = gr.Textbox(label="Verdict / Grade")

    # 逻辑绑定：点击视频组件触发 predict_on_click
    video_input.select(
        predict_on_click, 
        inputs=[video_input], 
        outputs=[video_output, plot_output, pdi_score, verdict]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
