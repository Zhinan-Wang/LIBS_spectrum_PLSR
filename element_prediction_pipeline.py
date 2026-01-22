"""
元素含量预测流水线模块 (优化修正版)
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import os
from datetime import datetime
import pickle

# 导入项目模块
from plsr_model import PLSRSpectralModel, find_optimal_components_for_element
from spectral_preprocessing import SpectralPreprocessor
from evaluation_visualization import create_timestamp_directory

def load_element_data(file_path: str) -> Optional[pd.DataFrame]:
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        else:
            print(f"不支持的文件格式: {file_path}")
            return None
    except Exception as e:
        print(f"加载元素含量数据时出错: {str(e)}")
        return None

def save_results_to_run_folder(results_dict: Dict, timestamp_dir: Optional[str], method_name: str):
    if not results_dict or not timestamp_dir: return
    run_dir = os.path.join(timestamp_dir, "run")
    os.makedirs(run_dir, exist_ok=True)
    
    data_list = []
    for elem, metrics in results_dict.items():
        if isinstance(metrics, dict):
            row = {'Element': elem, 'Model_Type': method_name}
            # 安全获取指标，默认为0
            row.update({k: metrics.get(k, 0) for k in ['r2', 'rmse', 'mre', 'n_components']})
            data_list.append(row)
            
    if data_list:
        pd.DataFrame(data_list).to_csv(
            os.path.join(run_dir, f"element_prediction_{method_name}.csv"), 
            index=False, encoding='utf-8-sig'
        )

class ElementPredictionPipeline:
    def __init__(self, spectral_model: Optional[PLSRSpectralModel] = None, parsimony_threshold: float = 0.01, scale: bool = False):
        self.spectral_model = spectral_model
        self.parsimony_threshold = parsimony_threshold
        self.scale = scale
        self.element_models = {}
        self.element_names = []
        
    def _train_generic(self, 
                       spectra: np.ndarray, 
                       element_data: pd.DataFrame, 
                       train_indices: List[int], 
                       val_indices: List[int],
                       mode_name: str,
                       timestamp_dir: Optional[str] = None,
                       val_spectra: Optional[np.ndarray] = None) -> Dict:
        """核心通用方法"""
        print(f"\n>>> 启动预测流程: [{mode_name}]")
        
        # 1. 自动识别元素列
        if not self.element_names:
            ignore = ['id', 'sample', 'sample_id', 'no', 'index', 'name', '序号', '元素']
            self.element_names = []
            for c in element_data.columns:
                if str(c).strip().lower() in ignore: continue
                # 尝试转换为数值，如果大部分行都是有效数值，则认为是元素列
                # 这能兼容包含少量 '<LOD' 或空值的列
                converted = pd.to_numeric(element_data[c], errors='coerce')
                valid_ratio = converted.notna().sum() / len(element_data)
                if valid_ratio > 0.5: # 超过50%是有效数值
                    self.element_names.append(c)
            print(f"    检测到元素: {self.element_names}")

        models_dir = os.path.join(timestamp_dir, "models", mode_name) if timestamp_dir else None
        if models_dir: os.makedirs(models_dir, exist_ok=True)

        self.element_models = {}
        train_results_dict = {}
        
        for element in self.element_names:
            # 数据转换与对齐
            y_full = pd.to_numeric(element_data[element], errors='coerce').values
            y_train = y_full[train_indices]
            
            # 过滤无效值
            mask = ~np.isnan(y_train)
            if not np.any(mask): 
                print(f"    警告: 元素 {element} 没有有效的训练数据")
                continue
            
            X_train_valid = spectra[train_indices][mask]
            y_train_valid = y_train[mask]
            
            # 自动寻优 (限制最大组件数)
            max_c = min(15, len(y_train_valid) - 1, X_train_valid.shape[1] - 1)
            if max_c < 1: max_c = 1
            
            opt_n = find_optimal_components_for_element(
                X_train_valid, y_train_valid, 
                max_components=max_c, 
                parsimony_threshold=self.parsimony_threshold, 
                scale=self.scale,
                timestamp_dir=timestamp_dir,
                element_name=f"{mode_name}_{element}"
            )
            
            # 训练模型
            pls = PLSRegression(n_components=opt_n, scale=self.scale)
            pls.fit(X_train_valid, y_train_valid)
            
            # 保存
            if models_dir:
                try:
                    with open(os.path.join(models_dir, f"{element}.pkl"), 'wb') as f:
                        pickle.dump(pls, f)
                except Exception as e:
                    print(f"    模型保存失败: {e}")
            
            # 记录训练集指标
            y_pred_train = pls.predict(X_train_valid).flatten()
            r2_train = r2_score(y_train_valid, y_pred_train)
            rmse_train = np.sqrt(mean_squared_error(y_train_valid, y_pred_train))
            
            # 记录训练结果以便后续合并绘图
            train_results_dict[element] = {
                'y_true': y_train_valid,
                'y_pred': y_pred_train,
                'r2': r2_train,
                'rmse': rmse_train
            }

            self.element_models[element] = {'model': pls, 'n_components': opt_n}
            print(f"    - {element:<5}: n={opt_n:<2}, Train R2={r2_train:.4f}")

        # 2. 验证集评估
        validation_source = val_spectra if val_spectra is not None else spectra
        return self.evaluate_element_prediction_on_validation_set(
            validation_source[val_indices], element_data.iloc[val_indices], 
            val_indices=list(range(len(val_indices))),
            title_prefix=mode_name, 
            timestamp_dir=timestamp_dir,
            train_results=train_results_dict
        )

    # 接口保持兼容
    def train_element_models_with_lq_only(self, lq_spectra, element_data, train_indices, val_indices, timestamp_dir=None):
        return self._train_generic(lq_spectra, element_data, train_indices, val_indices, "LQ-only", timestamp_dir)
    
    def train_element_models_with_hq_only(self, hq_spectra, element_data, train_indices, val_indices, timestamp_dir=None):
        return self._train_generic(hq_spectra, element_data, train_indices, val_indices, "HQ-only", timestamp_dir)

    def train_element_models_with_calibrated_spectra(self, calibrated_spectra, element_data, train_indices, val_indices, timestamp_dir=None):
        return self._train_generic(calibrated_spectra, element_data, train_indices, val_indices, "Calib-Spec", timestamp_dir)

    def train_element_models_hq_train_calib_test(self, hq_spectra, calibrated_spectra, element_data, train_indices, val_indices, timestamp_dir=None):
        return self._train_generic(hq_spectra, element_data, train_indices, val_indices, "Calib-Spec(HQ-Train)", timestamp_dir, val_spectra=calibrated_spectra)

    def evaluate_element_prediction_on_validation_set(self, calibrated_spectra, element_data, val_indices, title_prefix="", timestamp_dir=None, train_results=None):
        results = {}
        out_dir = os.path.join(timestamp_dir, "element_prediction") if timestamp_dir else None
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        
        print(f"    > 正在评估验证集 ({title_prefix})...")
        
        for element in self.element_names:
            if element not in self.element_models: continue
            
            model = self.element_models[element]['model']
            y_val = pd.to_numeric(element_data[element], errors='coerce').values
            mask = ~np.isnan(y_val)
            
            if not np.any(mask): continue
            
            X_val = calibrated_spectra[mask]
            y_true = y_val[mask]
            
            # 维度匹配检查
            if X_val.shape[0] != y_true.shape[0]:
                print(f"    错误: {element} 验证数据维度不匹配 X:{X_val.shape} Y:{y_true.shape}")
                continue

            y_pred = model.predict(X_val).flatten()
            
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mre = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-9)) * 100
            
            results[element] = {
                'r2': r2, 'rmse': rmse, 'mre': mre, 
                'n_components': self.element_models[element]['n_components'],
                'y_true': y_true,
                'y_pred': y_pred
            }
            
            if out_dir:
                train_res = train_results.get(element) if train_results else None
                
                if train_res:
                    # 合并绘图：左边训练集，右边验证集
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
                    
                    # Train Plot
                    t_true, t_pred = train_res['y_true'], train_res['y_pred']
                    ax1.scatter(t_true, t_pred, alpha=0.6, edgecolors='k', linewidth=0.5, label='Train')
                    vmin_t = min(t_true.min(), t_pred.min())
                    vmax_t = max(t_true.max(), t_pred.max())
                    ax1.plot([vmin_t, vmax_t], [vmin_t, vmax_t], 'r--', alpha=0.8)
                    ax1.set_title(f"{title_prefix} (Train) - {element}\nR2={train_res['r2']:.3f} RMSE={train_res['rmse']:.3f}")
                    ax1.set_xlabel("True"); ax1.set_ylabel("Pred")
                    ax1.grid(True, alpha=0.3)
                    
                    # Val Plot
                    ax2.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5, color='orange', label='Val')
                    vmin_v = min(y_true.min(), y_pred.min())
                    vmax_v = max(y_true.max(), y_pred.max())
                    ax2.plot([vmin_v, vmax_v], [vmin_v, vmax_v], 'r--', alpha=0.8)
                    ax2.set_title(f"{title_prefix} (Val) - {element}\nR2={r2:.3f} RMSE={rmse:.3f}")
                    ax2.set_xlabel("True"); ax2.set_ylabel("Pred")
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"pred_{title_prefix}_{element}.png"), dpi=300)
                    plt.close()
                else:
                    # 仅验证集绘图 (旧逻辑)
                    plt.figure(figsize=(6, 5), dpi=300)
                    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
                    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
                    plt.title(f"{title_prefix} {element}\nR2={r2:.3f} RMSE={rmse:.3f}")
                    plt.xlabel("True"); plt.ylabel("Pred")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"val_{title_prefix}_{element}.png"), dpi=300)
                    plt.close()
                
        save_results_to_run_folder(results, timestamp_dir, title_prefix)
        return results
    