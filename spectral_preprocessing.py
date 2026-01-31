import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d, PchipInterpolator, Akima1DInterpolator
from scipy.signal import savgol_filter, wiener, find_peaks, correlate
from scipy.ndimage import shift
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Tuple, List, Optional, Dict
import warnings
import time
from joblib import Parallel, delayed, cpu_count

def handle_shape(func):
    """装饰器：统一处理1D/2D数组输入输出，自动适配形状"""
    def wrapper(self, spectra, *args, **kwargs):
        spectra = np.asarray(spectra)
        original_ndim = spectra.ndim
        
        # 统一转为 2D (n_samples, n_features)
        if original_ndim == 1:
            spectra = spectra.reshape(1, -1)
        
        result = func(self, spectra, *args, **kwargs)
        
        # 修复：如果是元组（如返回光谱和波长），直接返回，不进行形状还原
        if isinstance(result, tuple):
            return result

        # 如果输入是 1D 且输出是数组，则还原为 1D
        if original_ndim == 1 and isinstance(result, np.ndarray):
            return result.flatten()
            
        return result
    return wrapper

def _resample_worker(spectrum: np.ndarray, lq_wavelengths: np.ndarray, 
                    ref_wavelengths: np.ndarray, method: str) -> np.ndarray:
    """单条光谱重采样的独立工作函数"""
    # 简单的波长裁剪与排序逻辑
    if len(lq_wavelengths) != len(spectrum):
        min_len = min(len(lq_wavelengths), len(spectrum))
        lq_wavelengths = lq_wavelengths[:min_len]
        spectrum = spectrum[:min_len]
    
    # 确保波长单调递增且无重复
    if not np.all(np.diff(lq_wavelengths) > 0):
        sorted_idx = np.argsort(lq_wavelengths)
        lq_wavelengths = lq_wavelengths[sorted_idx]
        spectrum = spectrum[sorted_idx]
        
        # 去重 (取第一个出现的值)
        unique_wl, unique_idx = np.unique(lq_wavelengths, return_index=True)
        if len(unique_wl) < len(lq_wavelengths):
            lq_wavelengths = lq_wavelengths[unique_idx]
            spectrum = spectrum[unique_idx]
    
    try:
        if method == 'cubic_spline':
            cs = CubicSpline(lq_wavelengths, spectrum, extrapolate=True)
            return cs(ref_wavelengths)
        elif method == 'pchip':
            pchip = PchipInterpolator(lq_wavelengths, spectrum, extrapolate=True)
            return pchip(ref_wavelengths)
        elif method == 'akima':
            akima = Akima1DInterpolator(lq_wavelengths, spectrum)
            return akima(ref_wavelengths, extrapolate=True)
        elif method == 'linear':
            f = interp1d(lq_wavelengths, spectrum, kind='linear', bounds_error=False, fill_value="extrapolate") # type: ignore
            return f(ref_wavelengths)
        elif method == 'nearest':
            f = interp1d(lq_wavelengths, spectrum, kind='nearest', bounds_error=False, fill_value="extrapolate") # type: ignore
            return f(ref_wavelengths)
        else:
            # 默认回退
            f = interp1d(lq_wavelengths, spectrum, kind='linear', bounds_error=False, fill_value="extrapolate") # type: ignore
            return f(ref_wavelengths)
    except Exception:
        # 出错返回全0，避免进程崩溃
        return np.zeros_like(ref_wavelengths)

def _asls_worker(y: np.ndarray, H: sparse.spmatrix, p: float, niter: int) -> np.ndarray:
    """AsLS 单个样本处理函数"""
    n_points = len(y)
    w = np.ones(n_points)
    z = np.zeros_like(y)
    for _ in range(niter):
        W = sparse.diags(w)
        Z = W + H
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return y - z

class SpectralPreprocessor:
    """
    光谱预处理器 (矩阵加速版 - 已修复数值稳定性)
    """
    def __init__(self, target_wavelengths: Optional[np.ndarray] = None):
        self.target_wavelengths = target_wavelengths

    @handle_shape
    def normalize_max(self, spectra: np.ndarray) -> np.ndarray:
        max_vals = np.max(spectra, axis=1, keepdims=True)
        max_vals = np.where(max_vals == 0, 1, max_vals)
        return spectra / max_vals

    @handle_shape
    def area_normalization(self, spectra: np.ndarray) -> np.ndarray:
        areas = np.trapz(np.abs(spectra), axis=1, dx=1)
        areas = np.where(areas == 0, 1, areas)
        return spectra / areas[:, np.newaxis]

    @handle_shape
    def min_max_normalization(self, spectra: np.ndarray) -> np.ndarray:
        """Min-Max 归一化: 将光谱缩放到 [0, 1] 范围"""
        min_vals = np.min(spectra, axis=1, keepdims=True)
        max_vals = np.max(spectra, axis=1, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1, range_vals)
        return (spectra - min_vals) / range_vals

    @handle_shape
    def l2_normalization(self, spectra: np.ndarray) -> np.ndarray:
        """L2 范数归一化 (向量归一化)"""
        norms = np.linalg.norm(spectra, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return spectra / norms

    @handle_shape
    def asls_baseline_correction(self, spectra: np.ndarray, lam: float = 1e5, p: float = 0.001, niter: int = 10) -> np.ndarray:
        """
        AsLS (Asymmetric Least Squares) 基线校正
        适用于复杂背景的去除，比多项式拟合更鲁棒。
        :param lam: 平滑参数 (lambda)，通常 1e4 ~ 1e9。值越大基线越平滑。
        :param p: 不对称参数，通常 0.001 ~ 0.1。
        :param niter: 迭代次数。
        """
        n_samples, n_points = spectra.shape
        corrected = np.zeros_like(spectra)
        
        # 构造二阶差分矩阵 D 的 H = lam * D.T * D
        # 利用稀疏矩阵加速计算
        diagonals = [np.ones(n_points-2), -2*np.ones(n_points-2), np.ones(n_points-2)]
        D = sparse.diags(diagonals, offsets=[0, 1, 2], shape=(n_points-2, n_points)) # type: ignore
        H = lam * (D.T @ D)
        
        # 并行处理 (使用 threading 后端因为 spsolve 释放 GIL 且 H 矩阵共享)
        if n_samples > 5:
            results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(_asls_worker)(spectra[i], H, p, niter) for i in range(n_samples)
            )
            return np.array(results)
        else:
            return np.array([_asls_worker(spectra[i], H, p, niter) for i in range(n_samples)])

    @handle_shape
    def snv_normalization(self, spectra: np.ndarray) -> np.ndarray:
        means = np.mean(spectra, axis=1, keepdims=True)
        stds = np.std(spectra, axis=1, keepdims=True)
        stds = np.where(stds == 0, 1, stds)
        return (spectra - means) / stds

    @handle_shape
    def baseline_correction(self, spectra: np.ndarray, degree: int = 3) -> np.ndarray:
        """
        基线校正 (矩阵加速版 - 数值稳定)
        """
        n_samples, n_points = spectra.shape
        
        # 修复：将 x 归一化到 [-1, 1] 区间，防止高阶多项式数值溢出
        x = np.linspace(-1, 1, n_points)
        
        # 1. 构建范德蒙德矩阵
        X_mat = np.vander(x, degree + 1)
        
        try:
            # 2. 计算伪逆矩阵 (利用SVD，数值更稳定)
            pinv = np.linalg.pinv(X_mat)
            
            # 3. 计算系数: Beta = (X^T X)^-1 X^T * Y
            coeffs = np.dot(pinv, spectra.T)
            
            # 4. 重建基线
            baseline = np.dot(X_mat, coeffs).T
            
            return spectra - baseline
        except Exception as e:
            print(f"矩阵基线校正失败，回退到逐行模式: {e}")
            # 回退方案：使用原始坐标的 polyfit (numpy内部处理了平移缩放)
            corrected = np.zeros_like(spectra)
            orig_x = np.arange(n_points)
            for i in range(n_samples):
                p = np.polyfit(orig_x, spectra[i], degree)
                corrected[i] = spectra[i] - np.polyval(p, orig_x)
            return corrected

    @handle_shape
    def smoothing(self, spectra: np.ndarray, window_length: int = 5, polyorder: int = 3, method: str = 'savgol') -> np.ndarray:
        if method == 'savgol':
            wl = window_length
            if wl >= spectra.shape[1]: wl = spectra.shape[1] - 1
            if wl % 2 == 0: wl -= 1
            if wl < 3: return spectra
            
            poly = min(polyorder, wl - 1)
            # savgol_filter 支持 axis=-1 并行处理
            return savgol_filter(spectra, window_length=wl, polyorder=poly, axis=-1)
            
        elif method == 'wiener':
            output = np.zeros_like(spectra)
            for i in range(spectra.shape[0]):
                output[i] = wiener(spectra[i])
            return output
        return spectra

    @handle_shape
    def spectral_alignment(self, spectra: np.ndarray, max_shift: int = 10, reference_method: str = 'mean') -> np.ndarray:
        """
        光谱对齐 (Spectral Alignment / Shift Correction)
        基于 FFT 互相关检测并修正光谱的微小位移。
        
        :param max_shift: 允许的最大位移像素数。
        :param reference_method: 'mean' (使用平均光谱作为参考) 或 'first' (使用第一条光谱)。
        """
        n_samples, n_points = spectra.shape
        aligned_spectra = np.zeros_like(spectra)
        
        # 1. 确定参考光谱
        if reference_method == 'first':
            ref_spec = spectra[0]
        else:
            ref_spec = np.mean(spectra, axis=0)
            
        # 去除直流分量以便更好地进行互相关
        ref_centered = ref_spec - np.mean(ref_spec)
        
        for i in range(n_samples):
            curr_spec = spectra[i]
            curr_centered = curr_spec - np.mean(curr_spec)
            
            # 2. 计算互相关 (mode='same' 保持长度一致，中心点在 len/2)
            cc = correlate(curr_centered, ref_centered, mode='same')
            
            # 3. 寻找峰值位移
            mid_point = len(cc) // 2
            # 只在 max_shift 范围内搜索，防止匹配到错误的远处峰
            search_start = max(0, mid_point - max_shift)
            search_end = min(len(cc), mid_point + max_shift + 1)
            
            roi = cc[search_start:search_end]
            if len(roi) == 0:
                shift_val = 0
            else:
                # lag > 0 表示 curr 向右偏移了，需要向左移回来
                shift_val = (search_start + np.argmax(roi)) - mid_point
            
            # 4. 应用反向位移 (mode='nearest' 填充边界)
            # shift 函数支持亚像素平移，这里我们使用计算出的整数位移
            aligned_spectra[i] = shift(curr_spec, -shift_val, mode='nearest')
            
        return aligned_spectra

    def resample_spectrum(self, wavelengths: np.ndarray, spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 简单的单条处理封装
        if self.target_wavelengths is None:
            return wavelengths, spectrum
        
        resampled = _resample_worker(spectrum, wavelengths, self.target_wavelengths, 'cubic_spline')
        return self.target_wavelengths, resampled
    
    # 兼容性方法
    @handle_shape
    def msc_normalization(self, spectra: np.ndarray, reference_spectrum: Optional[np.ndarray] = None) -> np.ndarray:
        if reference_spectrum is None:
            ref_spec = np.mean(spectra, axis=0)
        else:
            ref_spec = reference_spectrum
            
        msc_spectra = np.zeros_like(spectra)
        for i in range(spectra.shape[0]):
            fit = np.polyfit(ref_spec, spectra[i], 1)
            msc_spectra[i] = (spectra[i] - fit[1]) / fit[0]
        return msc_spectra
    
    @handle_shape
    def derivative_spectrum(self, spectra: np.ndarray, order: int = 1) -> np.ndarray:
        return np.diff(spectra, n=order, axis=-1, prepend=spectra[:, :order])


# ================= 辅助函数 =================

def find_overlap_wavelength_range(lq_wavelengths: np.ndarray, hq_wavelengths: np.ndarray) -> np.ndarray:
    overlap_min = max(np.min(lq_wavelengths), np.min(hq_wavelengths))
    overlap_max = min(np.max(lq_wavelengths), np.max(hq_wavelengths))
    
    if overlap_min >= overlap_max:
        # 如果没有重叠，给出一个友好的错误提示而不是崩溃
        raise ValueError(f"波长范围无重叠: LQ({np.min(lq_wavelengths):.1f}-{np.max(lq_wavelengths):.1f}) vs HQ({np.min(hq_wavelengths):.1f}-{np.max(hq_wavelengths):.1f})")
    
    mask = (hq_wavelengths >= overlap_min) & (hq_wavelengths <= overlap_max)
    return hq_wavelengths[mask]

def load_spectral_data_from_csv(lq_dir: str, hq_dir: str, sample_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    import os
    all_lq, all_hq = [], []
    valid_ids = []
    
    # 读取第一个样本确定波长
    if not sample_ids:
        raise ValueError("样品列表为空")

    first_id = sample_ids[0]
    try:
        hq_path = os.path.join(hq_dir, f"{first_id}.csv")
        lq_path = os.path.join(lq_dir, f"{first_id}.csv")
        
        if not os.path.exists(hq_path) or not os.path.exists(lq_path):
             raise FileNotFoundError(f"找不到首个样品文件: {first_id}")

        hq_wl = np.array(pd.read_csv(hq_path, header=None).iloc[:, 0].values)
        lq_wl = np.array(pd.read_csv(lq_path, header=None).iloc[:, 0].values)
    except Exception as e:
        raise ValueError(f"读取首个样本失败: {e}")

    overlap_wl = find_overlap_wavelength_range(lq_wl, hq_wl)
    overlap_min, overlap_max = np.min(overlap_wl), np.max(overlap_wl)

    for sid in sample_ids:
        # 加载 LQ
        lq_p = os.path.join(lq_dir, f"{sid}.csv")
        hq_p = os.path.join(hq_dir, f"{sid}.csv")
        
        # 必须同时存在才加载，保证数据对齐
        if os.path.exists(lq_p) and os.path.exists(hq_p):
            # 假设LQ只有一列强度
            # 优化: 使用 usecols 只读取第2列(索引1)，减少内存占用和IO时间
            try:
                # usecols=[1] 返回 DataFrame, values 为 2D array, 需要 flatten 转为 1D
                spec = pd.read_csv(lq_p, header=None, usecols=[1]).values.flatten()
            except Exception:
                # 如果读取失败(例如文件结构不同)，回退到读取所有列
                spec = pd.read_csv(lq_p, header=None).iloc[:, 1].values
            
            # 加载 HQ
            df = pd.read_csv(hq_p, header=None)
            wl_full = df.iloc[:, 0].values
            # 截取重叠区
            mask = (wl_full >= overlap_min) & (wl_full <= overlap_max)
            
            # 读取所有激发的强度列 (通常从第1列开始)
            intensity_data = df.iloc[:, 1:].values
            
            # 截取波长范围
            spectra_trimmed = intensity_data[mask, :] # (n_points, n_replicates)
            
            # 转置为 (n_replicates, n_points) 以便后续处理
            all_hq.append(spectra_trimmed.T)
            valid_ids.append(sid)
        else:
            print(f"   [Warning] 样品 {sid} 数据不完整 (LQ: {os.path.exists(lq_p)}, HQ: {os.path.exists(hq_p)}) - 已跳过")

    return np.array(all_lq), np.array(all_hq), lq_wl, overlap_wl, valid_ids

def resample_to_reference(lq_spectra: np.ndarray, lq_wavelengths: np.ndarray, 
                         ref_wavelengths: np.ndarray, method: str = 'cubic_spline') -> Tuple[np.ndarray, dict]:
    """并行重采样函数"""
    # 确保输入是2D
    if lq_spectra.ndim == 1: 
        lq_spectra = lq_spectra.reshape(1, -1)
    
    print(f"   [Resample] 启动并行重采样 (Method: {method}, CPUs: {cpu_count()})...")
    start_time = time.time()
    
    results = Parallel(n_jobs=-1)(
        delayed(_resample_worker)(lq_spectra[i], lq_wavelengths, ref_wavelengths, method) 
        for i in range(len(lq_spectra))
    )
    
    resampled = np.array(results)
    print(f"   [Resample] 完成，耗时: {time.time()-start_time:.2f}s")
    
    return resampled, {}

def evaluate_resampling_reliability(wavelengths: np.ndarray, spectrum: np.ndarray, method: str = 'cubic_spline', n_folds: int = 5) -> dict:
    """
    科学评估重采样可靠性：使用 K-Fold 交叉验证评估插值误差。
    注意：评估是在原始波长网格上进行的，避免了重采样前后波长点数不一致的问题。
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    
    # 确保输入是排序的
    sort_idx = np.argsort(wavelengths)
    x = wavelengths[sort_idx]
    y = spectrum[sort_idx]
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rmses = []
    mapes = []
    y_cv_pred = np.zeros_like(y) # 用于存储交叉验证的预测值
    
    for train_index, test_index in kf.split(x):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        
        # 必须对训练集重新排序，因为插值函数要求 x 单调
        sub_sort = np.argsort(x_train)
        x_train = x_train[sub_sort]
        y_train = y_train[sub_sort]
        
        # 使用现有的 worker 进行预测 (预测被挖掉的点 x_test)
        # 这里我们预测的是原始数据中被掩盖的点，所以维度是匹配的
        y_pred = _resample_worker(y_train, x_train, x_test, method)
        y_cv_pred[test_index] = y_pred # 收集预测结果
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmses.append(rmse)
        
        # MAPE (忽略接近0的点以防爆炸)
        mask = np.abs(y_test) > (np.max(np.abs(y)) * 0.01)
        if np.any(mask):
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
            mapes.append(mape)
            
    return {
        "cv_rmse": np.mean(rmses),
        "cv_mape": np.mean(mapes) if mapes else 0.0,
        "method": method,
        "n_folds": n_folds,
        # 返回详细数据用于绘图
        "cv_residuals": y - y_cv_pred,
        "sorted_wavelengths": x,
        "sorted_original": y
    }