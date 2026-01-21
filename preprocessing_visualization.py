import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import List, Tuple
import warnings

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 优化全局绘图设置：高清、细线、网格
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

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
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=300)
    
    # 上图：光谱对比
    ax1.plot(wavelengths, original_spectrum, label='原始光谱', alpha=0.7, color='blue', linewidth=1.0)
    ax1.plot(wavelengths, processed_spectrum, label=f'{step_name}后', alpha=0.8, color='red', linewidth=1.0)
    ax1.set_title(f'{step_name} - 样品 {sample_idx}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 下图：差值
    diff_spectrum = processed_spectrum - original_spectrum
    ax2.plot(wavelengths, diff_spectrum, label='差值 (处理后 - 原始)', color='green', alpha=0.8, linewidth=1.0)
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
    
    # 调整文本框位置，避免遮挡，放在子图2的右下角   
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.98, 0.05, metrics_text, transform=ax2.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{step_name}_sample_{sample_idx}_{timestamp}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
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
    
    fig, axes = plt.subplots(n_steps, 1, figsize=(14, 6 * n_steps), dpi=300)
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
        ax.plot(wavelengths, curr_spec, label=label, color=color, linewidth=1.0)
        ax.set_ylabel("Intensity")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # 绘制原始光谱
    ax = axes[0]
    ax.plot(wavelengths, original_spectrum, label='原始光谱', alpha=0.6, color='gray', linewidth=0.8, linestyle='--')
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
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    
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
        ax.plot(wavelengths, curr_spec, label=label, color=color, linewidth=1.0, alpha=0.8)
    
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.set_title(f'Complete Preprocessing Pipeline - Sample {sample_idx}', fontsize=14, fontweight='bold')
    # 将图例移到图外，防止遮挡密集的光谱数据
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"complete_pipeline_sample_{sample_idx}_{timestamp}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"已保存完整预处理流水线图: {filename}")

def visualize_resampling_quality(
    orig_wl: np.ndarray, orig_spec: np.ndarray,
    resamp_wl: np.ndarray, resamp_spec: np.ndarray,
    metrics: dict,
    sample_idx: int = 0,
    output_dir: str = "resampling_analysis"
) -> None:
    """
    可视化重采样效果。
    注意：由于 orig_wl 和 resamp_wl 长度不同，这里只做叠加展示，不画残差曲线。
    """
    fig = plt.figure(figsize=(14, 10), dpi=300)
    gs = fig.add_gridspec(2, 2)
    
    # 主图：全谱对比 (Overlay)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(orig_wl, orig_spec, c='gray', s=5, alpha=0.5, label='原始离散点 (Raw LQ)')
    ax1.plot(resamp_wl, resamp_spec, 'r-', linewidth=0.8, alpha=0.8, label=f'重采样曲线 ({metrics["method"]})')
    ax1.set_title(f"重采样效果评估 - Sample {sample_idx}", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Intensity")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 寻找一个显著的峰进行放大展示
    peak_idx = np.argmax(orig_spec)
    peak_wl = orig_wl[peak_idx]
    span = (orig_wl.max() - orig_wl.min()) * 0.05 # 展示 5% 的范围
    
    xlim_min = max(orig_wl.min(), peak_wl - span/2)
    xlim_max = min(orig_wl.max(), peak_wl + span/2)
    
    # 子图2：局部放大 (Zoom-in)
    ax2 = fig.add_subplot(gs[1, 0])
    mask_orig = (orig_wl >= xlim_min) & (orig_wl <= xlim_max)
    mask_resamp = (resamp_wl >= xlim_min) & (resamp_wl <= xlim_max)
    
    ax2.plot(orig_wl[mask_orig], orig_spec[mask_orig], 'ko', markersize=3, label='原始点', alpha=0.6)
    ax2.plot(resamp_wl[mask_resamp], resamp_spec[mask_resamp], 'r.-', linewidth=1.0, markersize=2, label='重采样')
    ax2.set_title(f"局部细节放大 ({xlim_min:.1f}-{xlim_max:.1f} nm)")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Intensity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3：CV 残差分析 (最有说服力的部分)
    ax3 = fig.add_subplot(gs[1, 1])
    
    if 'cv_residuals' in metrics:
        cv_wl = metrics['sorted_wavelengths']
        cv_res = metrics['cv_residuals']
        # 绘制残差散点
        ax3.scatter(cv_wl, cv_res, c='blue', s=3, alpha=0.4, label='CV Residuals')
        ax3.axhline(0, color='black', linestyle='--', linewidth=1)
        ax3.set_title(f"交叉验证残差分布 (K={metrics['n_folds']}) - 算法稳定性检测")
        ax3.set_xlabel("Wavelength (nm)")
        ax3.set_ylabel("Residual (Orig - Pred)")
        ax3.grid(True, alpha=0.3)
        
        # 计算残差范围，用于设置Y轴，避免个别离群点压缩整体视图
        std_res = float(np.std(cv_res))
        if std_res == 0: std_res = 1e-6 # 防止除零或无效范围
        ax3.set_ylim(-5*std_res, 5*std_res)

    # 将指标文本以浮动框形式放入图中
    text_str = (
        f"Method: {metrics['method']}\n"
        f"CV-RMSE: {metrics['cv_rmse']:.4f}\n"
        f"CV-MAPE: {metrics['cv_mape']:.2f}%"
    )
    ax3.text(0.95, 0.95, text_str, transform=ax3.transAxes, 
             fontsize=10, va='top', ha='right', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"))
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"resampling_eval_sample_{sample_idx}_{timestamp}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"   [Plot] 重采样评估图已保存: {filename}")

def visualize_data_alignment(
    lq_wl_raw: np.ndarray, lq_spec_raw: np.ndarray,
    lq_wl_trim: np.ndarray, lq_spec_trim: np.ndarray,
    resamp_wl: np.ndarray, resamp_spec: np.ndarray,
    sample_idx: int = 0,
    output_dir: str = "data_alignment"
) -> None:
    """
    可视化数据加载、裁剪和重采样的过程
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=300)
    
    # Subplot 1: Raw Spectrum with Trimmed Region Highlighted
    ax1.plot(lq_wl_raw, lq_spec_raw, 'k-', alpha=0.6, linewidth=0.8, label='原始光谱 (Raw LQ)')
    
    # Highlight the trimmed region
    trim_min, trim_max = np.min(lq_wl_trim), np.max(lq_wl_trim)
    ax1.axvspan(trim_min, trim_max, color='yellow', alpha=0.2, label='裁剪区域 (ROI)')
    
    ax1.set_title(f"步骤1: 原始光谱与感兴趣区域 (Sample {sample_idx})", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Intensity")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Trimmed vs Resampled
    ax2.scatter(lq_wl_trim, lq_spec_trim, c='gray', s=5, alpha=0.6, label='裁剪后离散点 (Trimmed LQ)')
    ax2.plot(resamp_wl, resamp_spec, 'r-', linewidth=1.0, alpha=0.8, label='重采样后 (Resampled HQ Grid)')
    
    ax2.set_title(f"步骤2: 裁剪与重采样效果对比", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Intensity")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_alignment_sample_{sample_idx}_{timestamp}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()
    print(f"   [Plot] 数据对齐可视化图已保存: {filename}")