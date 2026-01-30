import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import f
import os
import pickle
from typing import Optional
from joblib import parallel_backend

# 新增算法库
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

class PLSRSpectralModel:
    """PLSR 光谱校准模型"""
    
    def __init__(self, n_components: int = 25, scale: bool = False):
        self.n_components = n_components
        self.scale = scale
        self.model = PLSRegression(n_components=n_components, scale=scale)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        if np.any(np.isnan(X_train)): X_train = np.nan_to_num(X_train)
        if np.any(np.isnan(y_train)): y_train = np.nan_to_num(y_train)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"   [Model] PLSR模型已拟合 (n={self.n_components})")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_fitted: raise ValueError("模型未拟合")
        if np.any(np.isnan(X_test)): X_test = np.nan_to_num(X_test)
        return self.model.predict(X_test)

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"   [Model] 保存至: {filepath}")

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        self.n_components = self.model.n_components

class GenericSpectralModel:
    """通用光谱模型包装器 (适配 SVR, ElasticNet, RF 等)"""
    def __init__(self, model):
        self.model = model
        self.is_fitted = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted: raise ValueError("模型未拟合")
        return self.model.predict(X)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

def find_optimal_components(X: np.ndarray, Y: np.ndarray, 
                           max_components: int = 15, 
                           task_type: str = 'calibration',
                           timestamp_dir: Optional[str] = None,
                           parsimony_threshold: float = 0.01,
                           scale: bool = False,
                           X_base: Optional[np.ndarray] = None,
                           element_name: Optional[str] = None,
                           selection_method: str = "1-se",
                           f_test_alpha: float = 0.05,
                           wold_r_threshold: float = 0.95) -> tuple:
    """
    自动参数寻优 (并行版 LOO-CV)
    """
    # 预处理：处理 NaN 值，防止交叉验证崩溃
    X = np.nan_to_num(X)
    Y = np.nan_to_num(Y)

    n_samples, n_features = X.shape
    limit = min(n_samples - 1, n_features, max_components)
    if limit < 1: limit = 1
    
    components_range = range(1, limit + 1)
    score_list = []
    mse_values = []
    se_values = []
    
    print(f"   [Auto-ML] 启动并行 LOO-CV 寻优 (Task: {task_type}, Max: {limit}, Scale: {scale}, Strategy: {selection_method})...")
    
    loo = LeaveOneOut()
    
    for n in components_range:
        model = PLSRegression(n_components=n, scale=scale)
        
        # 使用 cross_val_predict 获取所有样本的留一法预测值
        # 这比手动循环更高效且代码更整洁，同时支持并行
        try:
            # 针对 Windows 小样本优化：强制使用 threading 后端避免进程创建(spawn)开销
            with parallel_backend('threading'):
                y_cv = cross_val_predict(model, X, Y, cv=loo, n_jobs=-1)
        except Exception as e:
            print(f"      [Warning] CV failed for n={n}: {e}")
            score_list.append(np.inf if task_type == 'prediction' else -1.0)
            continue

        # --- 统一计算 MSE 和 SE (用于高级选择策略) ---
        mse = mean_squared_error(Y, y_cv)
        sq_errors = (Y.flatten() - y_cv.flatten()) ** 2
        se = np.std(sq_errors, ddof=1) / np.sqrt(len(sq_errors))
        mse_values.append(mse)
        se_values.append(se)

        if task_type == 'prediction':
            # 预测任务主要关注 RMSE
            rmse = np.sqrt(mse)
            score_list.append(rmse)
            
        else: 
            # 校准任务主要关注 Correlation
            # 处理差异学习的重构
            if X_base is not None:
                y_final_pred = y_cv + X_base
                y_final_true = Y + X_base
            else:
                y_final_pred = y_cv
                y_final_true = Y
            
            # 向量化计算行相关性 (比逐个循环快)
            # Center the data
            pred_mean = np.mean(y_final_pred, axis=1, keepdims=True)
            true_mean = np.mean(y_final_true, axis=1, keepdims=True)
            pred_c = y_final_pred - pred_mean
            true_c = y_final_true - true_mean
            
            # Calculate correlation
            num = np.sum(pred_c * true_c, axis=1)
            den = np.sqrt(np.sum(pred_c**2, axis=1) * np.sum(true_c**2, axis=1))
            
            # Handle division by zero
            valid = den > 1e-9
            corrs = np.zeros(len(Y))
            corrs[valid] = num[valid] / den[valid]
            
            score_list.append(np.mean(corrs))

    scores = np.array(score_list)

    # --- 优化主成分选择策略 ---
    # 如果选择了基于 MSE 的策略 (min_mse, 1-se, f_test, wold_r)，则使用通用 MSE 逻辑
    mse_strategies = ["min_mse", "1-se", "f_test", "wold_r"]
    
    if selection_method in mse_strategies:
        # 找到 MSE 最小的点
        best_idx = np.argmin(mse_values)
        min_mse = mse_values[best_idx]
        min_se = se_values[best_idx]
        best_score = scores[best_idx] # RMSE
        
        optimal_idx = best_idx

        if selection_method == "1-se":
            # 策略 A: 1-SE Rule (一倍标准误准则) - 防过拟合，适合噪声大数据
            # 目标阈值 = 最小MSE + 它的标准误
            target_threshold = min_mse + min_se
            for i in range(best_idx):
                if mse_values[i] <= target_threshold:
                    optimal_idx = i
                    break
        elif selection_method == "min_mse":
            # 策略 B: Min-MSE (最小误差准则) - 追求最高精度，适合欠拟合场景
            optimal_idx = best_idx
            
        elif selection_method == "f_test":
            # 策略 C: F-test (Haaland & Thomas) - 统计显著性检验
            # 寻找主成分数 h < h_min，使得 MSE(h) 与 MSE(h_min) 无显著差异
            f_crit = f.ppf(1 - f_test_alpha, n_samples, n_samples)
            
            # 默认选 min_mse，然后尝试向前寻找更简单的模型
            optimal_idx = best_idx
            min_mse = mse_values[best_idx]
            
            for i in range(best_idx):
                f_stat = mse_values[i] / min_mse
                # 如果 F统计量 < 临界值，说明该简化模型与最佳模型无显著差异，可以接受
                if f_stat <= f_crit:
                    optimal_idx = i
                    break
                    
        elif selection_method == "wold_r":
            # 策略 D: Wold's R 准则 (停止规则)
            # 当 PRESS(k)/PRESS(k-1) > threshold 时停止
            optimal_idx = 0 # 默认至少 1 个主成分
            for i in range(1, len(mse_values)):
                ratio = mse_values[i] / mse_values[i-1]
                if ratio < wold_r_threshold:
                    optimal_idx = i
                else:
                    break # 停止增加主成分
        else:
            # 默认回退
            optimal_idx = best_idx

    else:
        # 默认策略 (针对校准任务的 Correlation Parsimony)
        # 或者当 selection_method 设置为 'parsimony' 时
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        optimal_idx = best_idx
        
        # 阈值 (parsimony_threshold)
        threshold = parsimony_threshold
        
        for i in range(best_idx):
            current_score = scores[i]
            # scores 是 Correlation (例如 best = 0.999)
            # 修正逻辑：将 "1 - Correlation" 视为误差 (Dissimilarity)
            best_error = 1.0 - best_score
            current_error = 1.0 - current_score
            
            if current_error <= best_error * (1 + threshold):
                optimal_idx = i
                break
    
    optimal_n = components_range[optimal_idx]
    
    # 计算选定模型的验证指标 (Q2 或 Correlation)
    if task_type == 'prediction':
        # Q2 = 1 - (MSE_cv / Var_y)
        opt_mse = mse_values[optimal_idx]
        var_y = np.var(Y)
        if var_y == 0:
            val_score = 0.0
        else:
            val_score = 1.0 - (opt_mse / var_y)
    else:
        val_score = scores[optimal_idx]

    if optimal_idx != best_idx:
        if selection_method in mse_strategies:
            print(f"   [Auto-ML] {selection_method}准则生效: 选择 n={optimal_n} (MSE: {mse_values[optimal_idx]:.4f}) 替代 n={components_range[best_idx]} (Best MSE: {mse_values[best_idx]:.4f}, SE: {se_values[best_idx]:.4f})")
        else:
            print(f"   [Auto-ML] 简约策略生效: 选择 n={optimal_n} (Score: {scores[optimal_idx]:.4f}) 替代 n={components_range[best_idx]} (Best: {best_score:.4f})")
    
    if timestamp_dir:
        plt.figure(figsize=(8, 4))
        plt.plot(components_range, scores, 'o-', markersize=4)
        
        # 绘制最佳点和选定点
        plt.plot(components_range[best_idx], scores[best_idx], 'r*', markersize=10, label=f'Global Best: {components_range[best_idx]}')
        if optimal_idx != best_idx:
            plt.plot(optimal_n, scores[optimal_idx], 'go', markersize=8, label=f'Selected: {optimal_n}')
            
        plt.axvline(x=optimal_n, color='g', linestyle='--', alpha=0.5)
        
        title = f"Optimal Components (LOO-CV) - {task_type}"
        if element_name:
            title += f" [{element_name}]"
        plt.title(title)
        
        y_label = "RMSE" if task_type == 'prediction' else "Correlation"
        plt.xlabel("Components"); plt.ylabel(y_label)
        plt.legend(); plt.grid(True, alpha=0.3)
        
        save_dir = os.path.join(timestamp_dir, "model_analysis")
        if element_name:
            save_dir = os.path.join(save_dir, "cv_plots")
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"cv_optimization_{task_type}"
        if element_name:
            safe_name = str(element_name).replace(" ", "").replace("/", "_")
            filename += f"_{safe_name}"
        filename += ".png"
        
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

    # Return history for external plotting
    history = {
        'components': list(components_range),
        'scores': score_list,
        'metric': 'RMSE' if task_type == 'prediction' else 'Correlation'
    }
    return optimal_n, val_score, history

def find_optimal_components_for_element(X, y, max_components=15, parsimony_threshold=0.01, scale=False, timestamp_dir=None, element_name=None, selection_method="1-se", f_test_alpha=0.05, wold_r_threshold=0.95):
    return find_optimal_components(X, y, max_components, task_type='prediction', parsimony_threshold=parsimony_threshold, scale=scale, timestamp_dir=timestamp_dir, element_name=element_name, selection_method=selection_method, f_test_alpha=f_test_alpha, wold_r_threshold=wold_r_threshold)

def _clean_params(params, ignore_keys=None):
    """清理不属于当前模型的参数"""
    if ignore_keys is None:
        ignore_keys = ['max_components', 'parsimony_threshold', 'selection_method', 
                       'f_test_alpha', 'wold_r_threshold', 'learn_difference']
    
    # 复制并移除 PLSR 特有参数
    clean = params.copy()
    for k in ignore_keys:
        clean.pop(k, None)
    
    # 单独处理 scale，因为很多模型需要外部 StandardScaler
    scale = clean.pop('scale', False)
    return clean, scale

def train_calibration_model(X, Y, config, timestamp_dir=None, X_base=None):
    """
    根据配置训练光谱校准模型
    """
    method = config.get('method', 'PLSR')
    params = config.get('params', {})
    
    if method.upper() == 'PLSR':
        max_components = params.get('max_components', 15)
        scale = params.get('scale', False)
        parsimony_threshold = params.get('parsimony_threshold', 0.01)
        selection_method = params.get('selection_method', 'parsimony') # 默认为相关性简约策略
        f_test_alpha = params.get('f_test_alpha', 0.05)
        wold_r_threshold = params.get('wold_r_threshold', 0.95)
        
        # 自动寻优
        optimal_n, best_score, history = find_optimal_components(
            X, Y, 
            max_components=max_components, 
            task_type='calibration', 
            timestamp_dir=timestamp_dir, 
            parsimony_threshold=parsimony_threshold, 
            selection_method=selection_method,
            f_test_alpha=f_test_alpha,
            wold_r_threshold=wold_r_threshold,
            scale=scale, 
            X_base=X_base
        )
        
        model = PLSRSpectralModel(n_components=optimal_n, scale=scale)
        model.fit(X, Y)
        
        return model, {'n_components': optimal_n, 'score': best_score, 'history': history}
    
    elif method.upper() == 'SVR':
        # SVR 光谱校准 (多输出回归)
        params_clean, use_scale = _clean_params(params)
        # 默认参数优化
        if 'kernel' not in params_clean: params_clean['kernel'] = 'rbf'
        
        estimator = SVR(**params_clean)
        if use_scale: estimator = make_pipeline(StandardScaler(), estimator)
        
        # 使用 MultiOutputRegressor 包装以支持多波长输出
        model = MultiOutputRegressor(estimator, n_jobs=-1)
        model.fit(X, Y)
        return GenericSpectralModel(model), {'score': 0}

    elif method.upper() == 'ELASTICNET':
        params_clean, use_scale = _clean_params(params)
        estimator = ElasticNet(**params_clean)
        if use_scale: estimator = make_pipeline(StandardScaler(), estimator)
        
        model = MultiOutputRegressor(estimator, n_jobs=-1)
        model.fit(X, Y)
        return GenericSpectralModel(model), {'score': 0}

    elif method.upper() in ['RF', 'RANDOMFOREST']:
        params_clean, _ = _clean_params(params) # RF 通常不需要缩放
        if 'n_estimators' not in params_clean: params_clean['n_estimators'] = 100
        
        model = RandomForestRegressor(**params_clean, n_jobs=-1)
        model.fit(X, Y)
        return GenericSpectralModel(model), {'score': 0}

    else:
        raise ValueError(f"不支持的校准方法: {method} (目前仅支持 PLSR)")

def train_element_model(X, y, config, element_name, timestamp_dir=None):
    """
    根据配置训练元素预测模型
    """
    method = config.get('method', 'PLSR')
    params = config.get('params', {})
    
    if method.upper() == 'PLSR':
        max_components = params.get('max_components', 15)
        scale = params.get('scale', False)
        parsimony_threshold = params.get('parsimony_threshold', 0.01)
        selection_method = params.get('selection_method', '1-se')
        f_test_alpha = params.get('f_test_alpha', 0.05)
        wold_r_threshold = params.get('wold_r_threshold', 0.95)
        
        # 自动寻优
        optimal_n, val_score, history = find_optimal_components_for_element(
            X, y, 
            max_components=max_components, 
            parsimony_threshold=parsimony_threshold, 
            scale=scale, 
            timestamp_dir=timestamp_dir, 
            element_name=element_name, 
            selection_method=selection_method, 
            f_test_alpha=f_test_alpha, 
            wold_r_threshold=wold_r_threshold
        )
        
        model = PLSRegression(n_components=optimal_n, scale=scale)
        model.fit(X, y)
        
        return model, {'n_components': optimal_n, 'cv_score': val_score, 'cv_history': history}
        
    elif method.upper() == 'SVR':
        params_clean, use_scale = _clean_params(params)
        
        # 支持简单的网格搜索 (如果参数是列表)
        grid_params = {k: v for k, v in params_clean.items() if isinstance(v, list)}
        fixed_params = {k: v for k, v in params_clean.items() if not isinstance(v, list)}
        
        estimator = SVR(**fixed_params)
        if use_scale:
            estimator = make_pipeline(StandardScaler(), estimator)
            # 调整 grid 参数的 key
            grid_params = {f'svr__{k}': v for k, v in grid_params.items()} if grid_params else {}

        if grid_params:
            print(f"   [Auto-ML] SVR 网格搜索: {grid_params}")
            model = GridSearchCV(estimator, grid_params, cv=5, n_jobs=-1)
        else:
            model = estimator
            
        model.fit(X, y)
        if isinstance(model, GridSearchCV):
            best_params = model.best_params_
        else:
            best_params = fixed_params
        return model, {'n_components': 0, 'cv_score': 0, 'best_params': best_params}

    elif method.upper() == 'ELASTICNET':
        params_clean, use_scale = _clean_params(params)
        estimator = ElasticNet(**params_clean)
        if use_scale: estimator = make_pipeline(StandardScaler(), estimator)
        
        estimator.fit(X, y)
        return estimator, {'n_components': 0}

    elif method.upper() in ['RF', 'RANDOMFOREST']:
        params_clean, _ = _clean_params(params)
        if 'n_estimators' not in params_clean: params_clean['n_estimators'] = 100
        
        # 检查是否需要网格搜索
        grid_params = {k: v for k, v in params_clean.items() if isinstance(v, list)}
        fixed_params = {k: v for k, v in params_clean.items() if not isinstance(v, list)}
        
        model = RandomForestRegressor(**fixed_params, n_jobs=-1)
        if grid_params:
             model = GridSearchCV(RandomForestRegressor(n_jobs=-1), grid_params, cv=5, n_jobs=-1)
             
        model.fit(X, y)
        return model, {'n_components': 0}

    else:
        raise ValueError(f"不支持的预测方法: {method} (目前仅支持 PLSR)")