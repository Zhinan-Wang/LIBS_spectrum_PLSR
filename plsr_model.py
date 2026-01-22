import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import pickle
from typing import Optional

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

def find_optimal_components(X: np.ndarray, Y: np.ndarray, 
                           max_components: int = 15, 
                           task_type: str = 'calibration',
                           timestamp_dir: Optional[str] = None,
                           parsimony_threshold: float = 0.01,
                           scale: bool = False,
                           X_base: Optional[np.ndarray] = None,
                           element_name: Optional[str] = None) -> tuple:
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
    
    print(f"   [Auto-ML] 启动并行 LOO-CV 寻优 (Task: {task_type}, Max: {limit}, Scale: {scale})...")
    
    loo = LeaveOneOut()
    
    for n in components_range:
        model = PLSRegression(n_components=n, scale=scale)
        
        # 使用 cross_val_predict 获取所有样本的留一法预测值
        # 这比手动循环更高效且代码更整洁，同时支持并行
        try:
            y_cv = cross_val_predict(model, X, Y, cv=loo, n_jobs=-1)
        except Exception as e:
            print(f"      [Warning] CV failed for n={n}: {e}")
            score_list.append(np.inf if task_type == 'prediction' else -1.0)
            continue

        if task_type == 'prediction':
            # 1. 元素预测任务 (优化 RMSE)
            mse = mean_squared_error(Y, y_cv)
            rmse = np.sqrt(mse)
            score_list.append(rmse)
            
            # 计算 MSE 的标准误 (Standard Error) 用于 1-SE 准则
            # SE = std(squared_errors) / sqrt(N)
            sq_errors = (Y.flatten() - y_cv.flatten()) ** 2
            se = np.std(sq_errors, ddof=1) / np.sqrt(len(sq_errors))
            mse_values.append(mse)
            se_values.append(se)
            
        else: 
            # 2. 光谱校准任务 (优化相关性)
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

    # --- 优化主成分选择策略 (简约原则) ---
    if task_type == 'prediction':
        # 策略 A: 1-SE Rule (一倍标准误准则) - 更科学的防过拟合策略
        # 找到 MSE 最小的点
        best_idx = np.argmin(mse_values)
        min_mse = mse_values[best_idx]
        min_se = se_values[best_idx]
        best_score = scores[best_idx] # RMSE
        
        # 目标阈值 = 最小MSE + 它的标准误
        # 我们选择满足 MSE <= (Min_MSE + SE) 的最简单的模型
        target_threshold = min_mse + min_se
        
        optimal_idx = best_idx
        for i in range(best_idx):
            if mse_values[i] <= target_threshold:
                optimal_idx = i
                break
                
    else:
        # 策略 B: 相关性任务保持原有的百分比阈值策略
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
        if task_type == 'prediction':
            print(f"   [Auto-ML] 1-SE准则生效: 选择 n={optimal_n} (MSE: {mse_values[optimal_idx]:.4f}) 替代 n={components_range[best_idx]} (Best MSE: {mse_values[best_idx]:.4f}, SE: {se_values[best_idx]:.4f})")
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

    return optimal_n, val_score

def find_optimal_components_for_element(X, y, max_components=15, parsimony_threshold=0.01, scale=False, timestamp_dir=None, element_name=None):
    return find_optimal_components(X, y, max_components, task_type='prediction', parsimony_threshold=parsimony_threshold, scale=scale, timestamp_dir=timestamp_dir, element_name=element_name)