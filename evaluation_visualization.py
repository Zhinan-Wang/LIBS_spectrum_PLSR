import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import os
import warnings
from datetime import datetime
from scipy.signal import find_peaks

# 设置支持中文和上标字符的字体
def _set_plot_style():
    """统一设置绘图样式"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

_set_plot_style()

def create_timestamp_directory(base_dir: str = "results") -> str:
    """创建以当前时间戳命名的目录结构"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(os.path.join(full_dir, "model_analysis"), exist_ok=True) 
    os.makedirs(os.path.join(full_dir, "element_prediction"), exist_ok=True)
    os.makedirs(os.path.join(full_dir, "run"), exist_ok=True)
    
    return full_dir

def calculate_snr(spectrum: np.ndarray) -> float:
    """计算光谱的信噪比 (Signal-to-Noise Ratio)"""
    if spectrum.size == 0: return 0.0
    signal = np.max(spectrum) - np.min(spectrum)
    noise = np.std(spectrum)    
    return signal / noise if noise != 0 else 0

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """计算评估指标: RMSE, Correlation, SAM"""
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
    
    if y_true.size == 0: return 0.0, 0.0, 0.0
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr_matrix = np.corrcoef(y_true, y_pred)
        corr = corr_matrix[0, 1]
    else:
        corr = 0.0
    
    # SAM
    dot_product = np.dot(y_true, y_pred)
    norm_true = np.linalg.norm(y_true)
    norm_pred = np.linalg.norm(y_pred)
    sam = 0.0
    if norm_true > 0 and norm_pred > 0:
        cos_theta = np.clip(dot_product / (norm_true * norm_pred), -1.0, 1.0)
        sam = np.arccos(cos_theta)
    
    return rmse, corr, sam

def plot_true_vs_pred_scatter(y_true, y_pred, element_name, filepath):
    """绘制真实值与预测值散点图"""
    _set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Line', alpha=0.8)
    
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{element_name}: True vs Predicted Values\n(R$^2$ = {r2:.4f}, RMSE = {rmse:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def plot_results(val_lq: np.ndarray, val_hq: np.ndarray, val_pred: np.ndarray, 
                wavelengths: np.ndarray, sample_idx: int = 0, 
                title: str = "Spectral Calibration Result", timestamp_dir: str = "."):
    """
    绘制光谱校准对比图 (功能 E)
    包含: LQ输入, HQ真实值, HQ预测值
    """
    _set_plot_style()
    
    # 确保索引不越界
    if sample_idx >= len(val_lq): sample_idx = 0
    
    lq = val_lq[sample_idx]
    hq = val_hq[sample_idx]
    pred = val_pred[sample_idx]
    
    # 确保维度匹配
    min_len = min(len(wavelengths), len(lq), len(hq), len(pred))
    wl = wavelengths[:min_len]
    lq = lq[:min_len]
    hq = hq[:min_len]
    pred = pred[:min_len]
    
    # 计算指标
    rmse_calib = np.sqrt(mean_squared_error(hq, pred))
    corr_calib = np.corrcoef(hq, pred)[0, 1]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, dpi=300)
    
    # 子图1: 光谱对比
    ax1.plot(wl, lq, 'gray', alpha=0.5, label='Input LQ (Resampled)', linewidth=0.8)
    ax1.plot(wl, hq, 'b-', label='Target HQ', linewidth=1.0, alpha=0.8)
    ax1.plot(wl, pred, 'r--', label='Predicted HQ', linewidth=1.0)
    
    ax1.set_ylabel('Intensity (a.u.)')
    ax1.set_title(f"{title} - Sample {sample_idx}\nRMS={rmse_calib:.2f}, Corr={corr_calib:.4f}")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 残差
    error = hq - pred
    ax2.plot(wl, error, 'k-', linewidth=0.8)
    ax2.axhline(0, color='r', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Residual (Target - Pred)')
    ax2.set_title('Prediction Error')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    save_dir = os.path.join(timestamp_dir, "model_analysis")
    os.makedirs(save_dir, exist_ok=True)
    filename = f"spectral_calibration_sample_{sample_idx}_{datetime.now().strftime('%H%M%S')}.png"    
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()
    print(f"   [Plot] 校准效果图已保存: {filename}")

def plot_performance_comparison(results_lq: Dict, results_calib: Dict, results_hq: Dict, timestamp_dir: str, results_calib_self: Optional[Dict] = None):
    """绘制三种模式的 R2, RMSE, MRE 性能对比柱状图"""
    _set_plot_style()
    
    # 找出共同存在的元素
    keys_sets = [set(results_lq.keys()), set(results_calib.keys()), set(results_hq.keys())]
    if results_calib_self:
        keys_sets.append(set(results_calib_self.keys()))
    
    elements = sorted(list(set.intersection(*keys_sets)))
    if not elements: return

    save_dir = os.path.join(timestamp_dir, "model_analysis")
    os.makedirs(save_dir, exist_ok=True)

    # 定义要绘制的指标配置: (字典键名, 显示名称, Y轴标签, 文件名)
    metrics_config = [
        ('r2', r'R$^2$', r'R$^2$ Score (验证集)', "performance_comparison_R2.png"),
        ('rmse', 'RMSE', 'RMSE (均方根误差)', "performance_comparison_RMSE.png"),
        ('mre', 'MRE', 'MRE (平均相对误差 %)', "performance_comparison_MRE.png")
    ]

    x = np.arange(len(elements))
    
    if results_calib_self:
        width = 0.2
        offsets = [-1.5, -0.5, 0.5, 1.5]
    else:
        width = 0.25
        offsets = [-1, 0, 1]

    for key, name, ylabel, filename in metrics_config:
        vals_lq = [results_lq[e][key] for e in elements]
        vals_calib = [results_calib[e][key] for e in elements]
        vals_hq = [results_hq[e][key] for e in elements]
        vals_calib_self = []
        if results_calib_self:
            vals_calib_self = [results_calib_self[e][key] for e in elements]

        fig, ax = plt.subplots(figsize=(14 if results_calib_self else 12, 6), dpi=300)
        
        ax.bar(x + offsets[0]*width, vals_lq, width, label='LQ-only (原始)', alpha=0.8, color='gray')
        ax.bar(x + offsets[1]*width, vals_calib, width, label='Calib-Spec (HQ-Train)', alpha=0.9, color='dodgerblue')
        if results_calib_self:
            ax.bar(x + offsets[2]*width, vals_calib_self, width, label='Calib-Self (Calib-Train)', alpha=0.9, color='purple')
        ax.bar(x + offsets[-1]*width, vals_hq, width, label='HQ-only (理想上限)', alpha=0.8, color='green')

        ax.set_ylabel(ylabel)
        ax.set_title(f'不同模式下的元素预测性能对比 ({name})')
        ax.set_xticks(x)
        ax.set_xticklabels(elements)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # 自动调整Y轴下限，如果数据都是正数则从0开始
        all_vals = vals_lq + vals_calib + vals_hq
        if results_calib_self: all_vals += vals_calib_self
        min_val = min(all_vals)
        if min_val >= 0:
            ax.set_ylim(bottom=0)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()
        print(f"   [Plot] 性能对比柱状图已保存: {filename}")

def plot_prediction_scatter_comparison(results_lq: Dict, results_calib: Dict, results_hq: Dict, element: str, timestamp_dir: str, results_calib_self: Optional[Dict] = None):
    """为指定元素绘制三联对比散点图 (LQ vs Calib vs HQ)"""
    _set_plot_style()
    
    if element not in results_lq or element not in results_calib or element not in results_hq:
        return
    if results_calib_self and element not in results_calib_self:
        return

    modes = [('LQ-only', results_lq, 'gray'), ('Calib-Spec (HQ-Train)', results_calib, 'dodgerblue')]
    if results_calib_self:
        modes.append(('Calib-Self (Calib-Train)', results_calib_self, 'purple'))
    modes.append(('HQ-only', results_hq, 'green'))
    
    # 计算全局最大最小值以统一坐标轴
    all_vals = []
    for _, res, _ in modes:
        all_vals.extend(res[element]['y_true'])
        all_vals.extend(res[element]['y_pred'])
    
    vmin, vmax = min(all_vals), max(all_vals)
    margin = (vmax - vmin) * 0.05
    vmin -= margin
    vmax += margin

    n_cols = len(modes)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), dpi=300, sharey=True)
    
    for ax, (name, res, color) in zip(axes, modes):
        data = res[element]
        ax.scatter(data['y_true'], data['y_pred'], alpha=0.6, c=color, edgecolors='k', linewidth=0.5)
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', alpha=0.6, label='Ideal')
        
        ax.set_title(f"{name}\nR$^2$={data['r2']:.3f}  RMSE={data['rmse']:.3f}")
        ax.set_xlabel("True Value")
        if ax == axes[0]: ax.set_ylabel("Predicted Value")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)

    plt.suptitle(f"Element Prediction Comparison: {element}", fontsize=14, y=1.05)
    plt.tight_layout()
    
    save_dir = os.path.join(timestamp_dir, "element_prediction", "comparison")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"compare_{element}.png"), dpi=300, bbox_inches='tight')
    plt.close()