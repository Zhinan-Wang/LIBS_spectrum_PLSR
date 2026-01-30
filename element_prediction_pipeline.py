"""
元素含量预测流水线模块 (优化修正版)
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union
import os
from datetime import datetime
import pickle

# 导入项目模块
from plsr_model import PLSRSpectralModel, GenericSpectralModel, train_element_model
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
            
            # 提取验证集指标
            row.update({k: metrics.get(k, None) for k in ['r2', 'rmse', 'mre', 'n_components']})
            
            # 提取训练集指标 (如果存在)
            if 'train_r2' in metrics:
                row.update({
                    'train_r2': metrics.get('train_r2'),
                    'train_rmse': metrics.get('train_rmse'),
                    'train_mre': metrics.get('train_mre')
                })
                
            data_list.append(row)
            
    if data_list:
        df = pd.DataFrame(data_list)
        # 调整列顺序：元素 -> 模型 -> 主成分 -> 训练集指标 -> 验证集指标
        cols = ['Element', 'Model_Type', 'n_components', 'train_r2', 'train_rmse', 'train_mre', 'r2', 'rmse', 'mre']
        # 只保留存在的列
        final_cols = [c for c in cols if c in df.columns]
        df = df[final_cols]
        
        df.to_csv(
            os.path.join(run_dir, f"element_prediction_{method_name}.csv"), 
            index=False, encoding='utf-8-sig'
        )

def _run_cars_selection(X, y, n_runs=50, n_folds=5, max_components=10, random_state=None):
    """
    CARS (Competitive Adaptive Reweighted Sampling) 特征选择算法实现
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples, n_features = X.shape
    
    # 至少保留2个特征
    if n_features <= 2:
        return np.arange(n_features)

    # 参数初始化: r_i = a * exp(-k * i)
    # r_0 = 1, r_{N-1} = 2/n_features
    k = np.log(n_features / 2) / (n_runs - 1)
    a = 1.0 
    
    RMSECV_list = []
    feature_subsets = []
    
    # 初始变量集合 (全集)
    current_vars = np.arange(n_features)
    
    from sklearn.model_selection import KFold, cross_val_predict
    from joblib import parallel_backend
    
    for i in range(n_runs):
        # 1. 蒙特卡洛采样 (MCS) - 随机抽取 80% 样本用于建模计算权重
        n_train = int(0.8 * n_samples)
        if n_train < 3: n_train = n_samples
        
        rand_idx = np.random.choice(n_samples, n_train, replace=True)
        X_sub = X[rand_idx][:, current_vars]
        y_sub = y[rand_idx]
        
        # 2. 建立 PLS 模型计算回归系数
        n_comp = min(max_components, n_train - 2, X_sub.shape[1])
        if n_comp < 1: n_comp = 1
        
        pls = PLSRegression(n_components=n_comp, scale=False)
        try:
            pls.fit(X_sub, y_sub)
        except:
            break 
            
        # 3. 计算权重 (基于回归系数绝对值)
        coefs = np.abs(pls.coef_).flatten()
        w = coefs / np.sum(coefs)
        
        # 4. 计算保留率 r_i (EDF算法) & 确定保留数量
        r_i = a * np.exp(-k * i)
        n_keep = int(np.round(n_features * r_i))
        if n_keep < 2: n_keep = 2
        if n_keep > len(current_vars): n_keep = len(current_vars)
        
        # 5. 自适应重加权采样 (ARS)
        w = w / w.sum()
        keep_indices_local = np.random.choice(len(current_vars), size=n_keep, replace=False, p=w)
        current_vars = current_vars[keep_indices_local]
        
        # 6. 交叉验证评估当前子集
        X_sel = X[:, current_vars]
        n_comp_cv = min(max_components, X_sel.shape[1], int(n_samples * 0.8) - 1)
        if n_comp_cv < 1: n_comp_cv = 1
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        pls_cv = PLSRegression(n_components=n_comp_cv, scale=False)
        
        try:
            with parallel_backend('threading'):
                y_cv = cross_val_predict(pls_cv, X_sel, y, cv=kf, n_jobs=-1)
            rmse = np.sqrt(mean_squared_error(y, y_cv))
        except Exception:
            rmse = np.inf
            
        RMSECV_list.append(rmse)
        feature_subsets.append(current_vars)
        
        if len(current_vars) <= 2: break
            
    if not RMSECV_list: return np.arange(n_features)
        
    best_idx = np.argmin(RMSECV_list)
    best_features = feature_subsets[best_idx]
    print(f"    [CARS] 迭代 {n_runs} 次, 最佳子集包含 {len(best_features)} 个变量 (RMSE={RMSECV_list[best_idx]:.4f})")
    return best_features

class ElementPredictionPipeline:
    def __init__(self, spectral_model: Optional[Union[PLSRSpectralModel, GenericSpectralModel]] = None, prediction_config: Optional[Dict] = None, feature_selection_config: Optional[Dict] = None, wavelengths: Optional[np.ndarray] = None):
        self.spectral_model = spectral_model
        self.prediction_config = prediction_config if prediction_config else {"method": "PLSR", "params": {}}
        if isinstance(feature_selection_config, dict):
            self.feature_selection_config = feature_selection_config
        else:
            self.feature_selection_config = {"enabled": False}
        self.wavelengths = wavelengths
        self.element_models = {}
        self.element_names = []
        
    def _train_generic(self, 
                       spectra: np.ndarray, 
                       element_data: pd.DataFrame, 
                       train_indices: List[int], 
                       val_indices: List[int],
                       mode_name: str,
                       timestamp_dir: Optional[str] = None,
                       val_spectra: Optional[np.ndarray] = None,
                       selection_method: Optional[str] = None) -> Dict:
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
            
            # --- 特征选择 (Feature Selection) ---
            selected_indices = None
            
            # Pylance 修复: 使用局部变量并强制类型检查，解决 'bool' object has no attribute 'get'
            fs_config = self.feature_selection_config
            if not isinstance(fs_config, dict):
                fs_config = {"enabled": False}

            if fs_config.get('enabled', False):
                # 支持直接参数或 params 字典
                method = fs_config.get('method', 'pearson')
                
                # Pylance 修复: 确保 fs_params 是字典，防止 'bool' object has no attribute 'get'
                _params = fs_config.get('params')
                if isinstance(_params, dict):
                    fs_params = _params
                else:
                    fs_params = fs_config
                    
                threshold = fs_params.get('threshold', 0.1)
                
                if method == 'pearson':
                    # 向量化计算皮尔逊相关系数
                    # corr = dot(x_c, y_c) / (norm(x_c) * norm(y_c))
                    X_c = X_train_valid - X_train_valid.mean(axis=0)
                    y_c = y_train_valid - y_train_valid.mean()
                    
                    numer = np.dot(y_c, X_c)
                    denom = np.linalg.norm(y_c) * np.linalg.norm(X_c, axis=0)
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        corrs = numer / denom
                    corrs = np.nan_to_num(corrs) # 处理除零导致的 NaN
                    
                    # 筛选绝对值大于阈值的特征
                    selected_indices = np.where(np.abs(corrs) >= threshold)[0]
                    
                    # 兜底机制：如果选出的特征太少 (<10)，则放弃筛选，防止模型无法训练
                    if len(selected_indices) < 10:
                        print(f"    [Feature Selection] {element} 筛选后特征过少 ({len(selected_indices)}), 跳过筛选。")
                        selected_indices = None
                    else:
                        X_train_valid = X_train_valid[:, selected_indices]
                        # print(f"    [Feature Selection] {element} 保留特征: {len(selected_indices)}/{spectra.shape[1]}")
                
                elif method == 'roi':
                    # 基于物理特征窗口 (ROI) 的筛选
                    if self.wavelengths is None:
                        print(f"    [Feature Selection] 错误: 选择了 ROI 方法但未提供波长数据。")
                    else:
                        roi_ranges_config = fs_config.get('roi_ranges')
                        if not isinstance(roi_ranges_config, dict):
                            roi_ranges_config = {}
                        
                        # 尝试匹配元素名 (去除空格)
                        ranges = roi_ranges_config.get(element) or roi_ranges_config.get(element.strip())
                        
                        # 智能匹配氧化物 (e.g. SiO2 -> Si)
                        if not ranges:
                            # 按长度降序排列键，防止前缀误判 (e.g. 避免 'C' 匹配 'Ca')
                            sorted_keys = sorted(roi_ranges_config.keys(), key=len, reverse=True)
                            for key in sorted_keys:
                                if element.strip().startswith(key):
                                    ranges = roi_ranges_config[key]
                                    print(f"    [Feature Selection] 自动映射: {element} -> {key}")
                                    break

                        if ranges:
                            mask = np.zeros(len(self.wavelengths), dtype=bool)
                            for start_wl, end_wl in ranges:
                                if abs(start_wl - end_wl) < 1e-6:
                                    # 单点 ROI: 寻找最近波长
                                    idx = (np.abs(self.wavelengths - start_wl)).argmin()
                                    mask[idx] = True
                                else:
                                    mask |= (self.wavelengths >= start_wl) & (self.wavelengths <= end_wl)
                            
                            selected_indices = np.where(mask)[0]
                            if len(selected_indices) == 0:
                                print(f"    [Feature Selection] {element} ROI 范围内无数据，跳过筛选。")
                                selected_indices = None
                            else:
                                X_train_valid = X_train_valid[:, selected_indices]
                                # print(f"    [Feature Selection] {element} ROI 保留特征: {len(selected_indices)} 点")
                
                elif method == 'rfe':
                    # 递归特征消除 (RFE)
                    n_features_to_select = fs_params.get('n_features_to_select', None) # None 默认为一半
                    step = fs_params.get('step', 0.1) # 每次移除 10%
                    
                    estimator = PLSRegression(n_components=2)
                    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=step)
                    selector = selector.fit(X_train_valid, y_train_valid)
                    selected_indices = np.where(selector.support_)[0]
                    print(f"    [Feature Selection] RFE 选中 {len(selected_indices)} 个特征")
                    X_train_valid = X_train_valid[:, selected_indices]

                elif method == 'cars':
                    # CARS 算法
                    selected_indices = _run_cars_selection(X_train_valid, y_train_valid, n_runs=fs_params.get('n_runs', 50), n_folds=fs_params.get('n_folds', 5))
                    X_train_valid = X_train_valid[:, selected_indices]

            # 动态调整最大组件数 (针对 PLSR)
            # 注意：这里只是为了防止报错，具体逻辑在 train_element_model 内部也会处理，
            # 但为了安全起见，我们在这里修改 config 副本
            current_config = self.prediction_config.copy()
            params = current_config.get('params', {}).copy()
            
            max_c = params.get('max_components', 15)
            max_c = min(max_c, len(y_train_valid) - 1, X_train_valid.shape[1] - 1)
            if max_c < 1: max_c = 1
            params['max_components'] = max_c
            
            # 如果传入了特定的 selection_method (覆盖 config)
            if selection_method:
                params['selection_method'] = selection_method
            
            current_config['params'] = params
            
            # 调用通用训练函数
            model, metrics = train_element_model(
                X_train_valid, y_train_valid,
                config=current_config,
                element_name=f"{mode_name}_{element}",
                timestamp_dir=timestamp_dir
            )
            
            opt_n = metrics.get('n_components', 0)
            cv_q2 = metrics.get('cv_score', 0)
            cv_history = metrics.get('cv_history', None)
            
            # 保存
            if models_dir:
                try:
                    with open(os.path.join(models_dir, f"{element}.pkl"), 'wb') as f:
                        pickle.dump(model, f)
                except Exception as e:
                    print(f"    模型保存失败: {e}")
            
            # 记录训练集指标
            y_pred_train_raw = model.predict(X_train_valid)
            # 修复 Pylance 类型检查误报 (PLSRegression 可能被推断为返回 tuple)
            if isinstance(y_pred_train_raw, tuple):
                y_pred_train = y_pred_train_raw[0]
            else:
                y_pred_train = y_pred_train_raw
            y_pred_train = np.asarray(y_pred_train).flatten()
            r2_train = r2_score(y_train_valid, y_pred_train)
            rmse_train = np.sqrt(mean_squared_error(y_train_valid, y_pred_train))
            mre_train = np.mean(np.abs(y_pred_train - y_train_valid) / (np.abs(y_train_valid) + 1e-9)) * 100
            
            # 记录训练结果以便后续合并绘图
            train_results_dict[element] = {
                'y_true': y_train_valid,
                'y_pred': y_pred_train,
                'r2': r2_train,
                'rmse': rmse_train,
                'mre': mre_train,
                'cv_history': cv_history
            }

            self.element_models[element] = {'model': model, 'n_components': opt_n, 'selected_features': selected_indices}
            print(f"    - {element:<5}: n={opt_n:<2}, Train R2={r2_train:.4f}, RMSE={rmse_train:.4f}, MRE={mre_train:.2f}%, CV Q2={cv_q2:.4f}")

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
    def train_element_models_with_lq_only(self, lq_spectra, element_data, train_indices, val_indices, timestamp_dir=None, selection_method=None):
        return self._train_generic(lq_spectra, element_data, train_indices, val_indices, "LQ-only", timestamp_dir, selection_method=selection_method)
    
    def train_element_models_with_hq_only(self, hq_spectra, element_data, train_indices, val_indices, timestamp_dir=None, selection_method=None):
        return self._train_generic(hq_spectra, element_data, train_indices, val_indices, "HQ-only", timestamp_dir, selection_method=selection_method)

    def train_element_models_with_calibrated_spectra(self, calibrated_spectra, element_data, train_indices, val_indices, timestamp_dir=None, selection_method=None):
        return self._train_generic(calibrated_spectra, element_data, train_indices, val_indices, "Calib-Spec", timestamp_dir, selection_method=selection_method)

    def train_element_models_hq_train_calib_test(self, hq_spectra, calibrated_spectra, element_data, train_indices, val_indices, timestamp_dir=None, selection_method=None):
        return self._train_generic(hq_spectra, element_data, train_indices, val_indices, "Calib-Spec(HQ-Train)", timestamp_dir, val_spectra=calibrated_spectra, selection_method=selection_method)

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
            
            # 应用特征选择掩码
            selected_indices = self.element_models[element].get('selected_features')
            if selected_indices is not None:
                X_val = X_val[:, selected_indices]
            
            # 维度匹配检查
            if X_val.shape[0] != y_true.shape[0]:
                print(f"    错误: {element} 验证数据维度不匹配 X:{X_val.shape} Y:{y_true.shape}")
                continue

            y_pred = model.predict(X_val).flatten()
            
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mre = np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-9)) * 100
            
            print(f"    - {element:<5}: Val R2={r2:.4f}, RMSE={rmse:.4f}, MRE={mre:.2f}%")
            
            results[element] = {
                'r2': r2, 'rmse': rmse, 'mre': mre, 
                'n_components': self.element_models[element]['n_components'],
                'y_true': y_true,
                'y_pred': y_pred
            }
            
            # 新增：将训练集指标合并到结果字典中，以便保存到CSV
            if train_results and element in train_results:
                t_res = train_results[element]
                results[element].update({
                    'train_r2': t_res.get('r2'),
                    'train_rmse': t_res.get('rmse'),
                    'train_mre': t_res.get('mre'),
                    'cv_history': t_res.get('cv_history')
                })
            
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
    