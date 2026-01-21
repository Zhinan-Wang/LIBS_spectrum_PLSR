import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import pickle
from joblib import Parallel, delayed, cpu_count
from joblib import cpu_count
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

def _fit_predict_loo_corr(train_idx, val_idx, X, Y, n_comp, scale=False):
    """辅助函数：单次 LOO 任务，用于校准任务（目标是相关性）"""
    pls = PLSRegression(n_components=n_comp, scale=scale)
    pls.fit(X[train_idx], Y[train_idx])
    
    # 校准模式下，输入是X(LQ), 输出是Y(HQ)
    pred = pls.predict(X[val_idx]) # (1, n_features)
    true_val = Y[val_idx]          # (1, n_features)
    
    # 计算两个向量(光谱)的相关性
    # flatten确保变成 1D 数组进行对比
    c = np.corrcoef(true_val.flatten(), pred.flatten())[0, 1]
    return c if not np.isnan(c) else 0.0

def find_optimal_components(X: np.ndarray, Y: np.ndarray, 
                           max_components: int = 15, 
                           task_type: str = 'calibration',
                           timestamp_dir: Optional[str] = None,
                           parsimony_threshold: float = 0.01,
                           scale: bool = False) -> int:
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
    scores = []
    
    print(f"   [Auto-ML] 启动并行 LOO-CV 寻优 (Task: {task_type}, Max: {limit}, Scale: {scale}, CPU: {cpu_count()})...")
    
    loo = LeaveOneOut()
    
    for n in components_range:
        # 1. 元素预测任务 (单目标数值回归，优化 RMSE)
        if task_type == 'prediction':
            # scoring='neg_root_mean_squared_error' 返回负的RMSE
            cv_scores = cross_val_score(
                PLSRegression(n_components=n, scale=scale), 
                X, Y, 
                cv=loo, 
                scoring='neg_mean_squared_error', 
                n_jobs=-1
            )
            # 修正：计算 RMSE (Root Mean Squared Error)
            # 之前代码计算的是 MAE (Mean Absolute Error): np.mean(np.sqrt(-cv_scores))
            # 正确逻辑: sqrt(mean(squared_errors))
            rmse = np.sqrt(np.mean(-cv_scores))
            scores.append(-rmse)
            
        # 2. 光谱校准任务 (多目标波形拟合，优化相关性)
        else:
            results = Parallel(n_jobs=-1)(
                delayed(_fit_predict_loo_corr)(train_idx, val_idx, X, Y, n, scale)
                for train_idx, val_idx in loo.split(X)
            )
            scores.append(np.mean(np.array(results)))

    # --- 优化主成分选择策略 (简约原则) ---
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    optimal_idx = best_idx
    
    # 阈值 (parsimony_threshold)。允许性能比最佳值差一定比例以换取更简单的模型
    threshold = parsimony_threshold
    
    # 从头遍历，找到第一个满足“性能接近最佳”的主成分数
    for i in range(best_idx):
        current_score = scores[i]
        
        if task_type == 'prediction':
            # scores 是负 RMSE (例如 best = -0.100)
            # 我们允许误差增加 1%: target = -0.100 * 1.01 = -0.101
            # 如果 current (-0.1005) > -0.101，说明误差在允许范围内，接受这个更简单的模型
            if current_score >= best_score * (1 + threshold):
                optimal_idx = i
                break
        else:
            # scores 是 Correlation (例如 best = 0.999)
            # 修正逻辑：将 "1 - Correlation" 视为误差 (Dissimilarity)
            # 允许误差比最佳误差增加 threshold %
            # 例如：Best=0.999 (Err=0.001), Th=0.01 => Allowed Err=0.00101 => Min Score=0.99899
            # 这样可以防止 n=1 (Score=0.99) 这种虽然相关性高但细节丢失的模型被误选
            best_error = 1.0 - best_score
            current_error = 1.0 - current_score
            
            if current_error <= best_error * (1 + threshold):
                optimal_idx = i
                break
    
    optimal_n = components_range[optimal_idx]
    
    if optimal_idx != best_idx:
        print(f"   [Auto-ML] 简约策略生效: 选择 n={optimal_n} (Score: {scores[optimal_idx]:.4f}) 替代 n={components_range[best_idx]} (Best: {best_score:.4f})")
    
    if timestamp_dir:
        plt.figure(figsize=(8, 4))
        plt.plot(components_range, scores, 'o-', markersize=4)
        
        # 绘制最佳点和选定点
        plt.plot(components_range[best_idx], scores[best_idx], 'r*', markersize=10, label=f'Global Best: {components_range[best_idx]}')
        if optimal_idx != best_idx:
            plt.plot(optimal_n, scores[optimal_idx], 'go', markersize=8, label=f'Selected: {optimal_n}')
            
        plt.axvline(x=optimal_n, color='g', linestyle='--', alpha=0.5)
        plt.title(f"Optimal Components (LOO-CV) - {task_type}")
        plt.xlabel("Components"); plt.ylabel("Score")
        plt.legend(); plt.grid(True, alpha=0.3)
        
        save_dir = os.path.join(timestamp_dir, "model_analysis")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"cv_optimization_{task_type}.png"))
        plt.close()

    return optimal_n

def find_optimal_components_for_element(X, y, max_components=15, parsimony_threshold=0.01, scale=False):
    return find_optimal_components(X, y, max_components, task_type='prediction', parsimony_threshold=parsimony_threshold, scale=scale)