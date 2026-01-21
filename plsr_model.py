import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple, Optional, Union
import warnings
import matplotlib.pyplot as plt
import os
from datetime import datetime


class PLSRSpectralModel:
    """
    PLSR光谱模型，用于光谱校准（LQ -> HQ）
    """
    
    def __init__(self, n_components: int = 25, scale: bool = True):
        self.n_components = n_components
        self.scale = scale
        self.model = PLSRegression(n_components=n_components, scale=scale)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """训练模型"""
        # 简单的数据清洗
        if np.any(np.isnan(X_train)):
            X_train = np.nan_to_num(X_train)
        if np.any(np.isnan(y_train)):
            y_train = np.nan_to_num(y_train)
            
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"   [Model] PLSR模型训练完成 (n_components={self.n_components})")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """预测光谱"""
        if not self.is_fitted:
            raise ValueError("模型尚未拟合")
        
        if np.any(np.isnan(X_test)):
            X_test = np.nan_to_num(X_test)
            
        return self.model.predict(X_test)

    def save(self, filepath: str):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"   [Model] 模型已保存至: {filepath}")

    def load(self, filepath: str):
        import pickle
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        self.n_components = self.model.n_components


def find_optimal_components(X: np.ndarray, Y: np.ndarray, 
                           max_components: int = 20, 
                           task_type: str = 'calibration',
                           timestamp_dir: Optional[str] = None) -> int:
    """
    功能 B: 自动参数寻优
    使用 LOO-CV 寻找最优主成分数
    
    Args:
        task_type: 'calibration' (多输出，使用相关性) 或 'prediction' (单输出，使用RMSE)
    """
    # 限制最大组件数
    n_samples, n_features = X.shape
    limit = min(n_samples - 1, n_features, max_components)
    if limit < 1: limit = 1
    
    components_range = range(1, limit + 1)
    scores = []
    
    loo = LeaveOneOut()
    
    print(f"   [Auto-ML] 正在进行 LOO-CV 寻优 (Task: {task_type}, Max: {limit})...")
    
    for n in components_range:
        try:
            pls = PLSRegression(n_components=n, scale=True)
            current_scores = []
            
            # LOO 循环
            for train_idx, val_idx in loo.split(X):
                X_tr, X_v = X[train_idx], X[val_idx]
                Y_tr, Y_v = Y[train_idx], Y[val_idx]
                
                pls.fit(X_tr, Y_tr)
                Y_pred = pls.predict(X_v)
                
                if task_type == 'calibration':
                    # 光谱校准：目标是波形相似，使用平均相关性
                    # Flatten用于处理多维光谱
                    corr = np.corrcoef(Y_v.flatten(), Y_pred.flatten())[0, 1]
                    if np.isnan(corr): corr = 0
                    current_scores.append(corr)
                else:
                    # 元素预测：目标是数值准确，使用负RMSE (为了统一“越大越好”)
                    rmse = np.sqrt(mean_squared_error(Y_v, Y_pred))
                    current_scores.append(-rmse) 
            
            avg_score = np.mean(current_scores)
            scores.append(avg_score)
            # print(f"      n={n}: score={avg_score:.4f}")
            
        except Exception as e:
            print(f"      n={n}: Error {e}")
            scores.append(-np.inf)
            
    # 选择最佳点
    best_idx = np.argmax(scores)
    optimal_n = components_range[best_idx]
    
    # 绘图保存
    if timestamp_dir:
        plt.figure(figsize=(8, 4))
        plt.plot(components_range, scores, 'o-')
        plt.axvline(x=optimal_n, color='r', linestyle='--', label=f'Optimal: {optimal_n}')
        plt.xlabel('Number of Components')
        plt.ylabel('CV Score (Correlation or -RMSE)')
        plt.title(f'Optimal Components Selection ({task_type})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_dir = os.path.join(timestamp_dir, "model_analysis")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"cv_optimization_{task_type}_{datetime.now().strftime('%H%M%S')}.png"))
        plt.close()

    return optimal_n

# 兼容旧代码接口
def find_optimal_components_for_element(X, y, max_components=20):
    return find_optimal_components(X, y, max_components, task_type='prediction')