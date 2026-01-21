import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d, PchipInterpolator, Akima1DInterpolator
from scipy.signal import savgol_filter, wiener, resample as fourier_resample, find_peaks
from scipy.fft import fft, ifft
from typing import Tuple, List, Optional, Dict, Callable, Any
import warnings
from joblib import Parallel, delayed, cpu_count


def handle_shape(func):
    """装饰器：处理1D/2D数组输入输出，自动适配形状"""
    def wrapper(self, spectra, *args, **kwargs):
        spectra = np.asarray(spectra)
        original_shape = spectra.shape
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        
        result = func(self, spectra, *args, **kwargs)
        
        if len(original_shape) == 1:
            result = result.flatten()
        return result
    return wrapper

def _resample_worker(spectrum: np.ndarray, lq_wavelengths: np.ndarray, 
                    ref_wavelengths: np.ndarray, method: str) -> np.ndarray:
    """
    单条光谱重采样的独立工作函数，用于并行处理
    """
    # 处理波长和光谱长度不一致的情况
    if len(lq_wavelengths) != len(spectrum):
        min_len = min(len(lq_wavelengths), len(spectrum))
        lq_wavelengths_trimmed = lq_wavelengths[:min_len]
        spectrum_trimmed = spectrum[:min_len]
    else:
        lq_wavelengths_trimmed = lq_wavelengths
        spectrum_trimmed = spectrum
    
    # 确保波长单调递增
    if not np.all(lq_wavelengths_trimmed[1:] > lq_wavelengths_trimmed[:-1]):
        unique_wavelengths, indices = np.unique(lq_wavelengths_trimmed, return_index=True)
        spectrum_trimmed = spectrum_trimmed[indices]
        lq_wavelengths_trimmed = unique_wavelengths
    
    try:
        if method == 'cubic_spline':
            cs = CubicSpline(lq_wavelengths_trimmed, spectrum_trimmed, extrapolate=True)
            return cs(ref_wavelengths)
        elif method == 'linear':
            f = interp1d(lq_wavelengths_trimmed, spectrum_trimmed, kind='linear', 
                        bounds_error=False, fill_value=(spectrum_trimmed[0] + spectrum_trimmed[-1]) / 2)
            return f(ref_wavelengths)
        elif method == 'nearest':
            fill_val = (spectrum_trimmed[0], spectrum_trimmed[-1])
            f = interp1d(lq_wavelengths_trimmed, spectrum_trimmed, kind='nearest', 
                        bounds_error=False, fill_value=fill_val) # type: ignore
            return f(ref_wavelengths)
        elif method == 'pchip':
            pchip = PchipInterpolator(lq_wavelengths_trimmed, spectrum_trimmed, extrapolate=True)
            return pchip(ref_wavelengths)
        elif method == 'akima':
            akima = Akima1DInterpolator(lq_wavelengths_trimmed, spectrum_trimmed)
            return akima(ref_wavelengths, extrapolate=True)
        elif method == 'quadratic':
            f = interp1d(lq_wavelengths_trimmed, spectrum_trimmed, kind='quadratic',
                            bounds_error=False, fill_value=(spectrum_trimmed[0] + spectrum_trimmed[-1]) / 2)
            return f(ref_wavelengths)
        elif method == 'cubic':
            f = interp1d(lq_wavelengths_trimmed, spectrum_trimmed, kind='cubic',
                            bounds_error=False, fill_value=(spectrum_trimmed[0] + spectrum_trimmed[-1]) / 2)
            return f(ref_wavelengths)
        elif method == 'sinc':
            return _sinc_resample(spectrum_trimmed, lq_wavelengths_trimmed, ref_wavelengths)
        elif method == 'fourier':
            # Fourier resampling logic (simplified for worker)
            # Note: Complex fourier logic might be better kept in main or duplicated if simple
            raise NotImplementedError("Fourier method not fully supported in parallel worker yet")
        else:
            raise ValueError(f"Unknown resampling method: {method}")
    except Exception as e:
        raise ValueError(f"Error during {method} interpolation: {str(e)}")

class SpectralPreprocessor:
    """
    光谱预处理器，用于执行波长轴校准、面积归一化和基线校正
    """
    
    def __init__(self, target_wavelengths: Optional[np.ndarray] = None):
        self.target_wavelengths = target_wavelengths
        self.is_fitted = False
    
    def fit_resample_params(self, wavelengths: np.ndarray) -> 'SpectralPreprocessor':
        if self.target_wavelengths is None:
            self.target_wavelengths = wavelengths
        self.is_fitted = True
        return self
    
    def resample_spectrum(self, wavelengths: np.ndarray, spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.target_wavelengths is None:
            return wavelengths, spectrum
        
        if not np.all(np.diff(wavelengths) > 0):
            sorted_idx = np.argsort(wavelengths)
            wavelengths = wavelengths[sorted_idx]
            spectrum = spectrum[sorted_idx]
        
        min_wl = max(np.min(wavelengths), np.min(self.target_wavelengths))
        max_wl = min(np.max(wavelengths), np.max(self.target_wavelengths))
        
        mask = (self.target_wavelengths >= min_wl) & (self.target_wavelengths <= max_wl)
        target_wl_filtered = self.target_wavelengths[mask]
        
        if len(target_wl_filtered) > 0:
            cs = CubicSpline(wavelengths, spectrum, extrapolate=False)
            interpolated_spectrum = cs(target_wl_filtered)
            
            extended_spectrum = np.full_like(self.target_wavelengths, np.nan)
            extended_spectrum[mask] = interpolated_spectrum
            
            # 修复 1: 使用 np.asarray 确保返回类型严格匹配 ndarray
            # 优化: 使用 numpy 原生方法替代 pandas 操作以提高性能
            mask_nan = np.isnan(extended_spectrum)
            if np.any(mask_nan):
                # 简单的边缘填充逻辑
                extended_spectrum = pd.Series(extended_spectrum).ffill().bfill().to_numpy()
            return self.target_wavelengths, extended_spectrum
        else:
            # 修复 2: 添加 type: ignore 忽略 Pylance 对 tuple 类型的误报
            fill_val = (spectrum[0], spectrum[-1])
            f = interp1d(wavelengths, spectrum, kind='nearest', bounds_error=False, fill_value=fill_val) # type: ignore
            interpolated_spectrum = f(self.target_wavelengths)
            return self.target_wavelengths, interpolated_spectrum

    def normalize_max(self, spectra: np.ndarray) -> np.ndarray:
        spectra = np.asarray(spectra)
        original_shape = spectra.shape
        
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        
        max_vals = np.max(spectra, axis=1, keepdims=True)
        max_vals = np.where(max_vals == 0, 1, max_vals)
        normalized_spectra = spectra / max_vals
        
        if len(original_shape) == 1:
            normalized_spectra = normalized_spectra.flatten()
        
        return normalized_spectra

    def area_normalization(self, spectra: np.ndarray) -> np.ndarray:
        spectra = np.asarray(spectra)
        original_shape = spectra.shape
        
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        
        areas = np.trapz(np.abs(spectra), axis=1, dx=1)
        areas = np.where(areas == 0, 1, areas)
        area_normalized_spectra = spectra / areas[:, np.newaxis]
        
        if len(original_shape) == 1:
            area_normalized_spectra = area_normalized_spectra.flatten()
        
        return area_normalized_spectra

    def snv_normalization(self, spectra: np.ndarray) -> np.ndarray:
        spectra = np.asarray(spectra)
        original_shape = spectra.shape
        
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        
        means = np.mean(spectra, axis=1, keepdims=True)
        stds = np.std(spectra, axis=1, keepdims=True)
        stds = np.where(stds == 0, 1, stds)
        
        snv_normalized_spectra = (spectra - means) / stds
        
        if len(original_shape) == 1:
            snv_normalized_spectra = snv_normalized_spectra.flatten()
        
        return snv_normalized_spectra

    def msc_normalization(self, spectra: np.ndarray, reference_spectrum: Optional[np.ndarray] = None) -> np.ndarray:
        spectra = np.asarray(spectra)
        original_shape = spectra.shape
        
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        
        if reference_spectrum is None:
            ref_spec = np.mean(spectra, axis=0)
        else:
            ref_spec = reference_spectrum
            
        assert ref_spec is not None
        
        msc_corrected_spectra = np.zeros_like(spectra)
        for i in range(spectra.shape[0]):
            regression_result = np.polyfit(ref_spec, spectra[i], deg=1)
            a1, a0 = regression_result[0], regression_result[1]
            msc_corrected_spectra[i] = (spectra[i] - a0) / a1
        
        if len(original_shape) == 1:
            msc_corrected_spectra = msc_corrected_spectra.flatten()
        
        return msc_corrected_spectra

    def vector_normalization(self, spectra: np.ndarray) -> np.ndarray:
        spectra = np.asarray(spectra)
        original_shape = spectra.shape
        
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        
        norms = np.linalg.norm(spectra, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vector_normalized_spectra = spectra / norms
        
        if len(original_shape) == 1:
            vector_normalized_spectra = vector_normalized_spectra.flatten()
        
        return vector_normalized_spectra

    def baseline_correction(self, spectra: np.ndarray, degree: int = 3) -> np.ndarray:
        spectra = np.asarray(spectra)
        original_shape = spectra.shape
        
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        
        corrected_spectra = np.zeros_like(spectra)
        
        for i in range(spectra.shape[0]):
            coeffs = np.polyfit(np.arange(len(spectra[i])), spectra[i], deg=degree)
            baseline = np.polyval(coeffs, np.arange(len(spectra[i])))
            corrected_spectra[i] = spectra[i] - baseline
        
        if len(original_shape) == 1:
            corrected_spectra = corrected_spectra.flatten()
        
        return corrected_spectra

    def smoothing(self, spectra: np.ndarray, window_length: int = 5, polyorder: int = 3, method: str = 'savgol') -> np.ndarray:
        spectra = np.asarray(spectra)
        original_shape = spectra.shape
        
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        
        smoothed_spectra = np.zeros_like(spectra)
        
        for i in range(spectra.shape[0]):
            if method == 'savgol':
                wl = min(window_length, spectra[i].shape[0])
                if wl % 2 == 0: wl -= 1
                if wl < 1: wl = 1
                if wl < polyorder: polyorder = max(0, wl - 1)
                
                if wl > polyorder:
                    smoothed_spectra[i] = savgol_filter(spectra[i], window_length=wl, polyorder=polyorder)
                else:
                    smoothed_spectra[i] = spectra[i]
            elif method == 'wiener':
                smoothed_spectra[i] = wiener(spectra[i])
            else:
                raise ValueError(f"未知的平滑方法: {method}")
        
        if len(original_shape) == 1:
            smoothed_spectra = smoothed_spectra.flatten()
        
        return smoothed_spectra

    def derivative_spectrum(self, spectra: np.ndarray, order: int = 1, method: str = 'diff') -> np.ndarray:
        spectra = np.asarray(spectra)
        original_shape = spectra.shape
        
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        
        derivative_spectra = np.zeros_like(spectra)
        
        for i in range(spectra.shape[0]):
            if method == 'diff':
                derivative_spectra[i] = np.diff(spectra[i], n=order, axis=-1)
                for _ in range(order):
                    derivative_spectra[i] = np.append(derivative_spectra[i], derivative_spectra[i][-1])
            elif method == 'savgol':
                if order == 1:
                    derivative_spectra[i] = savgol_filter(spectra[i], window_length=5, polyorder=3, deriv=1)
                elif order == 2:
                    derivative_spectra[i] = savgol_filter(spectra[i], window_length=5, polyorder=3, deriv=2)
                else:
                    window_length = max(5, 2 * order + 1)
                    if window_length % 2 == 0: window_length += 1
                    derivative_spectra[i] = savgol_filter(spectra[i], window_length=window_length, polyorder=3, deriv=order)
            else:
                raise ValueError(f"未知的导数计算方法: {method}")
        return derivative_spectra

    def apply_preprocessing_pipeline(self, spectra: np.ndarray, pipeline: List[Dict]) -> np.ndarray:
        processed_spectra = spectra.copy()
        for step in pipeline:
            method = step['method']
            params = step.get('params', {})
            if hasattr(self, method):
                processed_spectra = getattr(self, method)(processed_spectra, **params)
            else:
                raise ValueError(f"未知的预处理方法: {method}")
        return processed_spectra


def find_overlap_wavelength_range(lq_wavelengths: np.ndarray, hq_wavelengths: np.ndarray) -> np.ndarray:
    lq_min, lq_max = np.min(lq_wavelengths), np.max(lq_wavelengths)
    hq_min, hq_max = np.min(hq_wavelengths), np.max(hq_wavelengths)
    
    overlap_min = max(lq_min, hq_min)
    overlap_max = min(lq_max, hq_max)
    
    if overlap_min >= overlap_max:
        raise ValueError(f"No overlap between LQ ({lq_min:.2f}-{lq_max:.2f}) and HQ ({hq_min:.2f}-{hq_max:.2f}) wavelength ranges")
    
    overlap_mask = (hq_wavelengths >= overlap_min) & (hq_wavelengths <= overlap_max)
    overlap_hq_wavelengths = hq_wavelengths[overlap_mask]
    return overlap_hq_wavelengths


def load_spectral_data_from_csv(lq_dir: str, hq_dir: str, 
                               sample_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import os
    
    all_lq_spectra = []
    all_hq_spectra = []
    
    sample_file = os.path.join(hq_dir, f"{sample_ids[0]}.csv")
    if os.path.exists(sample_file):
        df = pd.read_csv(sample_file, header=None)
        hq_wavelengths = np.array(df.iloc[:, 0].values)
    else:
        raise ValueError(f"Cannot find HQ spectrum file for sample {sample_ids[0]}")
    
    lq_sample_file = os.path.join(lq_dir, f"{sample_ids[0]}.csv")
    if os.path.exists(lq_sample_file):
        df = pd.read_csv(lq_sample_file, header=None)
        lq_wavelengths = np.array(df.iloc[:, 0].values)
    else:
        raise ValueError(f"Cannot find LQ spectrum file for sample {sample_ids[0]}")
    
    overlap_wavelengths = find_overlap_wavelength_range(lq_wavelengths, hq_wavelengths)
    
    for sample_id in sample_ids:
        lq_file = os.path.join(lq_dir, f"{sample_id}.csv")
        if os.path.exists(lq_file):
            df = pd.read_csv(lq_file, header=None)
            lq_spectrum_full = df.iloc[:, 1].values
            all_lq_spectra.append(lq_spectrum_full)
        else:
            warnings.warn(f"Cannot find LQ spectrum file for sample {sample_id}, skipping this sample")
    
    for sample_id in sample_ids:
        hq_file = os.path.join(hq_dir, f"{sample_id}.csv")
        if os.path.exists(hq_file):
            df = pd.read_csv(hq_file, header=None)
            hq_spectra = []
            hq_wavelengths_full = df.iloc[:, 0].values
            
            intensity_cols = [col for col in range(1, df.shape[1])]
            for col in intensity_cols[:10]:
                if col < df.shape[1]:
                    spectrum_full = df.iloc[:, col].values
                    hq_overlap_mask = (hq_wavelengths_full >= np.min(overlap_wavelengths)) & (hq_wavelengths_full <= np.max(overlap_wavelengths))
                    spectrum = spectrum_full[hq_overlap_mask]
                    hq_spectra.append(spectrum)
            
            all_hq_spectra.append(np.array(hq_spectra))
        else:
            warnings.warn(f"Cannot find HQ spectrum file for sample {sample_id}, skipping this sample")
    
    if len(all_lq_spectra) == 0 or len(all_hq_spectra) == 0:
        raise ValueError("No valid spectral files found")
    
    return np.array(all_lq_spectra), np.array(all_hq_spectra), np.array(overlap_wavelengths)


def resample_to_reference(lq_spectra: np.ndarray, lq_wavelengths: np.ndarray, 
                         ref_wavelengths: np.ndarray, method: str = 'cubic_spline') -> Tuple[np.ndarray, dict]:
    from scipy.signal import resample as signal_resample
    import time
    
    if len(lq_spectra.shape) == 1:
        lq_spectra = lq_spectra.reshape(1, -1)
    
    # 尝试向量化处理
    if method == 'cubic_spline':
        try:
            if not np.all(np.diff(lq_wavelengths) > 0):
                sorted_idx = np.argsort(lq_wavelengths)
                lq_wavelengths = lq_wavelengths[sorted_idx]
                lq_spectra = lq_spectra[:, sorted_idx]
            
            cs = CubicSpline(lq_wavelengths, lq_spectra, axis=1, extrapolate=True)
            resampled_spectra = cs(ref_wavelengths)
            eval_params = _calculate_resampling_eval_params_for_dataset(lq_spectra, resampled_spectra, lq_wavelengths, ref_wavelengths)
            return resampled_spectra, eval_params
        except Exception as e:
            warnings.warn(f"向量化三次样条插值失败，回退到循环模式: {e}")

    # 并行处理回退方案
    print(f"   [Resample] 启动并行重采样 (Method: {method}, CPUs: {cpu_count()})...")
    start_time = time.time()
    
    # 使用 joblib 进行并行处理
    # 注意：lq_wavelengths 假设对所有光谱是通用的，如果不是，需要修改此处逻辑
    results = Parallel(n_jobs=-1)(
        delayed(_resample_worker)(
            lq_spectra[i], lq_wavelengths, ref_wavelengths, method
        ) for i in range(lq_spectra.shape[0])
    )
    
    resampled_spectra = np.array(results)
    elapsed = time.time() - start_time
    print(f"   [Resample] 完成，耗时: {elapsed:.2f}s")
    
    
    if len(resampled_spectra) > 0:
        eval_params = _calculate_resampling_eval_params_for_dataset(lq_spectra, resampled_spectra, 
                                                                   lq_wavelengths, ref_wavelengths)
    else:
        eval_params = {}
    
    return resampled_spectra, eval_params


def _sinc_resample(spectrum: np.ndarray, original_wavelengths: np.ndarray, 
                   target_wavelengths: np.ndarray) -> np.ndarray:
    f = interp1d(original_wavelengths, spectrum, kind='linear', 
                 bounds_error=False, fill_value=(spectrum[0] + spectrum[-1]) / 2)
    resampled_orig = f(target_wavelengths)
    return resampled_orig


def _calculate_resampling_eval_params(original_spectrum: np.ndarray, resampled_spectrum: np.ndarray,
                                     original_wavelengths: np.ndarray, target_wavelengths: np.ndarray) -> dict:
    min_len = min(len(original_spectrum), len(resampled_spectrum))
    orig_limited = original_spectrum[:min_len]
    resamp_limited = resampled_spectrum[:min_len]
    
    correlation = np.corrcoef(orig_limited, resamp_limited)[0, 1] if np.std(orig_limited) != 0 and np.std(resamp_limited) != 0 else 0.0
    rmse = np.sqrt(np.mean((orig_limited - resamp_limited) ** 2))
    
    mean_orig = np.mean(np.abs(orig_limited))
    relative_error = np.mean(np.abs(orig_limited - resamp_limited)) / mean_orig * 100 if mean_orig != 0 else 0.0
    
    dot_product = np.sum(orig_limited * resamp_limited)
    norm_orig = np.linalg.norm(orig_limited)
    norm_resamp = np.linalg.norm(resamp_limited)
    sam = np.arccos(np.clip(dot_product / (norm_orig * norm_resamp), -1.0, 1.0)) if norm_orig != 0 and norm_resamp != 0 else 0.0
    
    orig_area = np.trapz(np.abs(orig_limited))
    resamp_area = np.trapz(np.abs(resamp_limited))
    area_ratio = abs(resamp_area / orig_area) if orig_area != 0 else 0.0
    
    orig_peak_prominence = np.std(orig_limited) * 0.1
    resamp_peak_prominence = np.std(resamp_limited) * 0.1
    
    orig_peaks, _ = find_peaks(orig_limited, prominence=orig_peak_prominence)
    resamp_peaks, _ = find_peaks(resamp_limited, prominence=resamp_peak_prominence)
    
    if len(orig_peaks) == 0:
        peak_preservation = 1.0 if len(resamp_peaks) == 0 else 0.0
    elif len(resamp_peaks) == 0:
        peak_preservation = 0.0
    else:
        matched_peaks = 0
        tolerance = max(1, min(5, len(orig_limited) // 20))
        for orig_peak in orig_peaks:
            distances = np.abs(resamp_peaks - orig_peak)
            if np.min(distances) <= tolerance:
                matched_peaks += 1
        peak_preservation = matched_peaks / len(orig_peaks)
    
    return {
        'correlation': correlation,
        'rmse': rmse,
        'relative_error_percent': relative_error,
        'sam': sam,
        'area_ratio': area_ratio,
        'peak_preservation_ratio': peak_preservation,
        'original_points_count': len(original_spectrum),
        'target_points_count': len(resampled_spectrum),
        'resampling_factor': len(resampled_spectrum) / len(original_spectrum) if len(original_spectrum) > 0 else 0.0
    }


def _calculate_resampling_eval_params_for_dataset(original_spectra: np.ndarray, resampled_spectra: np.ndarray,
                                                 original_wavelengths: np.ndarray, target_wavelengths: np.ndarray) -> dict:
    resampled_original_spectra = []
    
    for i in range(original_spectra.shape[0]):
        orig_spectrum = original_spectra[i]
        
        if len(original_wavelengths) != len(orig_spectrum):
            min_len = min(len(original_wavelengths), len(orig_spectrum))
            orig_spectrum = orig_spectrum[:min_len]
            curr_wl = original_wavelengths[:min_len]
        else:
            curr_wl = original_wavelengths
            
        f = interp1d(curr_wl, orig_spectrum, kind='linear', bounds_error=False, fill_value=np.nan)
        resampled_orig = f(target_wavelengths)
        
        nan_mask = np.isnan(resampled_orig)
        if np.any(nan_mask):
            left_idx = np.where(~nan_mask)[0]
            if len(left_idx) > 0:
                first_valid = left_idx[0]
                last_valid = left_idx[-1]
                resampled_orig[:first_valid] = resampled_orig[first_valid]
                resampled_orig[last_valid+1:] = resampled_orig[last_valid]
            else:
                resampled_orig[:] = orig_spectrum[0] if len(orig_spectrum) > 0 else 0
        resampled_original_spectra.append(resampled_orig)
    
    resampled_original_spectra = np.array(resampled_original_spectra)
    
    correlations = []
    rmse_values = []
    relative_errors = []
    sam_values = []
    area_ratios = []
    peak_preservation_ratios = []
    
    for i in range(resampled_original_spectra.shape[0]):
        orig_spectrum = resampled_original_spectra[i]
        resamp_spectrum = resampled_spectra[i]
        
        # 相关性
        if np.std(orig_spectrum) != 0 and np.std(resamp_spectrum) != 0:
            correlation = np.corrcoef(orig_spectrum, resamp_spectrum)[0, 1]
            correlation = np.clip(correlation, -1.0, 1.0)
        else:
            correlation = 0.0
        correlations.append(correlation)
        
        # RMSE
        rmse = np.sqrt(np.mean((orig_spectrum - resamp_spectrum) ** 2))
        rmse_values.append(rmse)
        
        # 相对误差
        mean_orig = np.mean(np.abs(orig_spectrum))
        relative_error = np.mean(np.abs(orig_spectrum - resamp_spectrum)) / mean_orig * 100 if mean_orig != 0 else 0.0
        relative_errors.append(relative_error)
        
        # SAM
        dot_product = np.sum(orig_spectrum * resamp_spectrum)
        norm_orig = np.linalg.norm(orig_spectrum)
        norm_resamp = np.linalg.norm(resamp_spectrum)
        sam = np.arccos(np.clip(dot_product / (norm_orig * norm_resamp), -1.0, 1.0)) if norm_orig != 0 and norm_resamp != 0 else 0.0
        sam_values.append(sam)
        
        # 面积比
        orig_area = np.trapz(np.abs(orig_spectrum))
        resamp_area = np.trapz(np.abs(resamp_spectrum))
        area_ratio = abs(resamp_area / orig_area) if orig_area != 0 else 0.0
        area_ratios.append(area_ratio)
        
        # 峰值保留
        orig_peak_prominence = np.std(orig_spectrum) * 0.1
        resamp_peak_prominence = np.std(resamp_spectrum) * 0.1
        orig_peaks, _ = find_peaks(orig_spectrum, prominence=orig_peak_prominence)
        resamp_peaks, _ = find_peaks(resamp_spectrum, prominence=resamp_peak_prominence)
        
        if len(orig_peaks) == 0:
            peak_preservation = 1.0 if len(resamp_peaks) == 0 else 0.0
        elif len(resamp_peaks) == 0:
            peak_preservation = 0.0
        else:
            matched_peaks = 0
            tolerance = max(1, min(5, len(orig_spectrum) // 20))
            for orig_peak in orig_peaks:
                distances = np.abs(resamp_peaks - orig_peak)
                if np.min(distances) <= tolerance:
                    matched_peaks += 1
            peak_preservation = matched_peaks / len(orig_peaks)
        peak_preservation_ratios.append(peak_preservation)
    
    return {
        'correlation': np.mean(correlations),
        'correlation_std': np.std(correlations),
        'rmse': np.mean(rmse_values),
        'rmse_std': np.std(rmse_values),
        'relative_error_percent': np.mean(relative_errors),
        'relative_error_percent_std': np.std(relative_errors),
        'sam': np.mean(sam_values),
        'sam_std': np.std(sam_values),
        'area_ratio': np.mean(area_ratios),
        'area_ratio_std': np.std(area_ratios),
        'peak_preservation_ratio': np.mean(peak_preservation_ratios),
        'peak_preservation_ratio_std': np.std(peak_preservation_ratios),
        'original_points_count': len(original_wavelengths),
        'target_points_count': len(target_wavelengths),
        'resampling_factor': len(target_wavelengths) / len(original_wavelengths) if len(original_wavelengths) > 0 else 0.0,
        'n_spectra': resampled_original_spectra.shape[0]
    }


def compare_resampling_methods(original_spectra: np.ndarray, 
                              original_wavelengths: np.ndarray,
                              target_wavelengths: np.ndarray,
                              methods: Optional[List[str]] = None) -> Dict[str, dict]:
    if methods is None:
        methods = ['cubic_spline', 'linear', 'pchip', 'akima', 'nearest', 'quadratic', 'cubic']
    results = {}
    for method in methods:
        try:
            resampled_spectra, _ = resample_to_reference(original_spectra, original_wavelengths, target_wavelengths, method=method)
            eval_params = _calculate_resampling_eval_params_for_dataset(
                original_spectra, resampled_spectra, original_wavelengths, target_wavelengths
            )
            eval_params['method'] = method
            results[method] = eval_params
        except Exception as e:
            print(f"方法 {method} 出错: {str(e)}")
            continue
    return results


def select_best_resampling_method(results: Dict[str, dict], 
                                 primary_metric: str = 'correlation',
                                 secondary_metric: str = 'peak_preservation_ratio') -> str:
    if not results:
        return 'cubic_spline'
    scores = {}
    for method, params in results.items():
        primary_val = params[primary_metric]
        secondary_val = params[secondary_metric]
        
        if primary_metric in ['relative_error_percent', 'rmse']:
            primary_val = 1 / (1 + primary_val)
        if secondary_metric in ['relative_error_percent', 'rmse']:
            secondary_val = 1 / (1 + secondary_val)
            
        combined_score = 0.7 * primary_val + 0.3 * secondary_val
        scores[method] = combined_score
    
    best_method = max(scores, key=lambda method: scores[method])
    return best_method


def auto_select_best_resampling_method(original_spectra: np.ndarray, 
                                    original_wavelengths: np.ndarray,
                                    target_wavelengths: np.ndarray,
                                    methods: Optional[List[str]] = None,
                                    primary_metric: str = 'correlation',
                                    secondary_metric: str = 'peak_preservation_ratio') -> Tuple[str, Dict]:
    comparison_results = compare_resampling_methods(
        original_spectra, original_wavelengths, target_wavelengths, methods
    )
    best_method = select_best_resampling_method(comparison_results, primary_metric, secondary_metric)
    return best_method, comparison_results