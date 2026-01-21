import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import List, Tuple
import warnings

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def calculate_preprocessing_metrics(original: np.ndarray, processed: np.ndarray) -> dict:
    """
    计算预处理前后的评估指标
    """
    # 避免循环引用，在函数内部导入
    from evaluation_visualization import calculate_snr
    
    correlation = np.corrcoef(original.flatten(), processed.flatten())[0, 1] if np.std(original) != 0 and np.std(processed) != 0 else 0.0
    rmse = np.sqrt(np.mean((original - processed) ** 2))
    
    orig_snr = calculate_snr(original)
    proc_snr = calculate_snr(processed)
    
    dot_product = np.sum(original * processed)
    norm_orig = np.linalg.norm(original)
    norm_proc = np.linalg.norm(processed)
    if norm_orig == 0 or norm_proc == 0:
        sam = 0.0
    else:
        cos_theta = np.clip(dot_product / (norm_orig * norm_proc), -1.0, 1.0)
        sam = np.arccos(cos_theta)
    
    return {
        'correlation': correlation,
        'rmse': rmse,
        'snr_original': orig_snr,
        'snr_processed': proc_snr,
        'snr_improvement': proc_snr - orig_snr,
        'sam': sam
    }

def visualize_preprocessing_step(
    original_spectrum: np.ndarray,
    processed_spectrum: np.ndarray,
    wavelengths: np.ndarray,
    step_name: str,
    sample_idx: int = 0,
    output_dir: str = "preprocessing_analysis"
) -> None:
    """
    可视化单个预处理步骤的结果：绘制原始图、处理后图和残差图
    """
    metrics = calculate_preprocessing_metrics(original_spectrum, processed_spectrum)
    
    min_len = min(len(wavelengths), len(original_spectrum), len(processed_spectrum))
    wavelengths = wavelengths[:min_len]
    original_spectrum = original_spectrum[:min_len]
    processed_spectrum = processed_spectrum[:min_len]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 上图：光谱对比
    ax1.plot(wavelengths, original_spectrum, label='原始光谱', alpha=0.8, color='blue', linewidth=1.2)
    ax1.plot(wavelengths, processed_spectrum, label=f'{step_name}后', alpha=0.8, color='red', linewidth=1.2)
    ax1.set_title(f'{step_name} - 样品 {sample_idx}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 下图：差值
    diff_spectrum = processed_spectrum - original_spectrum
    ax2.plot(wavelengths, diff_spectrum, label='差值 (处理后 - 原始)', color='green', alpha=0.8, linewidth=1.2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title(f'{step_name} - 光谱变化', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 文本框显示指标
    metrics_text = (
        f'相关性: {metrics["correlation"]:.4f}\n'
        f'RMSE: {metrics["rmse"]:.6f}\n'
        f'SNR改善: {metrics["snr_improvement"]:+.4f}\n'
        f'SAM: {metrics["sam"]:.4f}'
    )
    
    fig.text(0.85, 0.5, metrics_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{step_name}_sample_{sample_idx}_{timestamp}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"已保存预处理可视化图: {filename}")

def visualize_preprocessing_steps(
    original_spectrum: np.ndarray,
    processed_spectra: List[np.ndarray],
    wavelengths: np.ndarray,
    step_names: List[str],
    sample_idx: int = 0,
    output_dir: str = "preprocessing_analysis"
) -> None:
    """
    可视化多个预处理步骤的结果：绘制原始图、处理后图和残差图
    """
    n_steps = len(processed_spectra)
    if n_steps == 0:
        warnings.warn("没有提供任何预处理步骤的结果。")
        return
    
    fig, axes = plt.subplots(n_steps, 1, figsize=(14, 10 * n_steps))
    if n_steps == 1:
        axes = [axes]
    
    wl_len = len(wavelengths)
    for i, (spec, label) in enumerate(zip(processed_spectra, step_names)):
        ax = axes[i]
        
        # 裁剪或补齐数据以匹配波长轴
        curr_spec = spec
        if len(curr_spec) > wl_len:
            curr_spec = curr_spec[:wl_len]
        elif len(curr_spec) < wl_len:
            # 简单补0，防止绘图崩溃
            curr_spec = np.pad(curr_spec, (0, wl_len - len(curr_spec)))
            
        color = plt.get_cmap('viridis')(i / n_steps)
        ax.plot(wavelengths, curr_spec, label=label, color=color, linewidth=1.2)
        ax.set_ylabel("Intensity")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # 绘制原始光谱
    ax = axes[0]
    ax.plot(wavelengths, original_spectrum, label='原始光谱', alpha=0.8, color='blue', linewidth=1.2)
    ax.set_title(f'预处理步骤 - 样品 {sample_idx}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"preprocessing_steps_sample_{sample_idx}_{timestamp}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"已保存预处理可视化图: {filename}")

def visualize_complete_preprocessing_pipeline(
    viz_list: List[Tuple[np.ndarray, str]],
    wavelengths: np.ndarray,
    sample_idx: int = 0,
    output_dir: str = "overall_preprocessing"
) -> None:
    """
    可视化完整的预处理流水线：从原始数据到最终处理结果
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    wl_len = len(wavelengths)
    
    # 绘制每个预处理步骤
    for i, (spectrum, label) in enumerate(viz_list):
        # 确保数据长度与波长轴一致
        curr_spec = spectrum
        if len(curr_spec) > wl_len:
            curr_spec = curr_spec[:wl_len]
        elif len(curr_spec) < wl_len:
            # 补零以保持一致性
            curr_spec = np.pad(curr_spec, (0, wl_len - len(curr_spec)))
        
        # 使用不同颜色绘制每一步
        color = plt.get_cmap('tab10')(i / len(viz_list))
        ax.plot(wavelengths, curr_spec, label=label, color=color, linewidth=1.2)
    
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.set_title(f'Complete Preprocessing Pipeline - Sample {sample_idx}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"complete_pipeline_sample_{sample_idx}_{timestamp}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"已保存完整预处理流水线图: {filename}")