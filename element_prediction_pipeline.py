"""
元素含量预测流水线模块
用于结合光谱校准和元素含量预测的完整流程
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

# 请确保这些模块在你的项目中存在
from plsr_model import PLSRSpectralModel, find_optimal_components_for_element
from spectral_preprocessing import SpectralPreprocessor
from evaluation_visualization import create_timestamp_directory


def load_element_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    加载元素含量数据
    """
    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            element_data = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            element_data = pd.read_csv(file_path)
        else:
            print(f"不支持的文件格式: {file_path}")
            return None
        
        print(f"成功加载元素含量数据，形状: {element_data.shape}")
        print(f"列名: {list(element_data.columns)}")
        
        return element_data
    except Exception as e:
        print(f"加载元素含量数据时出错: {str(e)}")
        return None


def save_results_to_run_folder(results_dict: Dict, timestamp_dir: str, method_name: str):
    """
    将评估结果保存到指定方法的文件夹中
    """
    if not results_dict:
        return
    
    # 确保run文件夹存在
    run_dir = os.path.join(timestamp_dir, "run")
    os.makedirs(run_dir, exist_ok=True)

    # 特殊处理：如果是基线模型结果
    if isinstance(results_dict, dict) and 'baseline_model' in results_dict:
         filepath = os.path.join(run_dir, f"element_prediction_{method_name}_baseline.txt")
         with open(filepath, 'w', encoding='utf-8') as f:
             for k, v in results_dict['baseline_model'].items():
                 f.write(f"{k}: {v}\n")
         return

    results_list = []
    for element, metrics in results_dict.items():
        if isinstance(metrics, dict):
            result_row = {
                'Element': element,
                'Model_Type': method_name,
                'R2': metrics.get('r2', 0),
                'RMSE': metrics.get('rmse', 0),
                'MRE': metrics.get('mre', 0)
            }
            results_list.append(result_row)
    
    if not results_list:
        return

    # 创建DataFrame并保存为CSV
    results_df = pd.DataFrame(results_list)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"element_prediction_{method_name}_{timestamp}.csv"
    filepath = os.path.join(run_dir, filename)
    
    # 保存文件
    results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"评估结果已保存至: {filepath}")


def save_optimal_components_log(optimal_components_info: Dict, timestamp_dir: str):
    """
    将最优主成分数信息保存到日志文件
    """
    run_dir = os.path.join(timestamp_dir, "run")
    os.makedirs(run_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimal_components_log_{timestamp}.log"
    filepath = os.path.join(run_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"最优主成分数记录 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for element, info in optimal_components_info.items():
            if isinstance(info, dict):
                n_components = info.get('n_components', 'N/A')
                r2_score = info.get('r2_score', 'N/A')
                rmse = info.get('rmse', 'N/A')
                f.write(f"{element:<10}: 最优主成分数 = {n_components}, 训练集 R² = {r2_score:.4f}, 训练集 RMSE = {rmse:.4f}\n")
            else:
                f.write(f"{element:<10}: 最优主成分数 = {info}\n")
    
    print(f"最优主成分数记录已保存至: {filepath}")


def save_element_prediction_log(results_dict: Dict, timestamp_dir: str, method_name: str):
    """
    将详细日志保存到run文件夹
    """
    run_dir = os.path.join(timestamp_dir, "run")
    os.makedirs(run_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"element_prediction_{method_name}_{timestamp}.log"
    filepath = os.path.join(run_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"元素预测结果 ({method_name}) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for element, metrics in results_dict.items():
            if isinstance(metrics, dict):
                r2 = metrics.get('r2', 0)
                rmse = metrics.get('rmse', 0)
                mre = metrics.get('mre', 0)
                f.write(f"{element:<10} ({method_name:>8}) 验证集: R² = {r2:.4f}, RMSE = {rmse:.4f}, MRE = {mre:.4f}%\n")
    
    print(f"评估日志已保存至: {filepath}")


class ElementPredictionPipeline:
    """
    结合光谱校准和元素含量预测的完整流程
    """
    
    def __init__(self, spectral_model: Optional[PLSRSpectralModel] = None):
        """
        初始化元素预测管道
        """
        self.spectral_model = spectral_model
        self.element_models = {}  # 存储每个元素的预测模型
        self.element_names = []   # 元素名称列表
        self.processed_spectra = None  # 存储预处理后的光谱
        
    def _apply_preprocessing(self, spectra: np.ndarray, steps: Optional[List[Dict]]) -> np.ndarray:
        """应用预处理步骤"""
        if not steps:
            return spectra
        print("正在应用预处理步骤...")
        preprocessor = SpectralPreprocessor()
        processed = spectra.copy()
        
        # 映射方法名到预处理器实例方法
        method_map = {
            'baseline_correction': preprocessor.baseline_correction,
            'smoothing': preprocessor.smoothing,
            'area_normalization': preprocessor.area_normalization,
            'snv_normalization': preprocessor.snv_normalization,
            'normalize_max': preprocessor.normalize_max,
            'derivative_spectrum': preprocessor.derivative_spectrum,
            'msc_normalization': preprocessor.msc_normalization,
            'vector_normalization': preprocessor.vector_normalization
        }

        for step in steps:
            method_name = step['method']
            params = step.get('params', {})
            
            if method_name in method_map:
                try:
                    processed = method_map[method_name](processed, **params)
                except Exception as e:
                    print(f"应用预处理 {method_name} 失败: {e}")
            else:
                print(f"警告: 未知的预处理方法 '{method_name}'")
                
        return processed

    def calibrate_spectra(self, lq_spectra: np.ndarray, preprocessing_steps: Optional[List[Dict]] = None):
        """
        使用已训练的光谱校准模型对LQ光谱进行校准
        """
        print("正在进行光谱校准...")
        
        if self.spectral_model is None:
            raise ValueError("光谱校准模型未提供")
        
        if not self.spectral_model.is_fitted:
            raise ValueError("光谱校准模型未训练")
        
        # 预处理光谱
        processed_spectra = self._apply_preprocessing(lq_spectra, preprocessing_steps)
        
        # 使用训练好的模型进行预测（校准）
        calibrated_spectra = self.spectral_model.predict(processed_spectra)
        
        self.processed_spectra = processed_spectra
        
        print(f"光谱校准完成，输出形状: {calibrated_spectra.shape}")
        return calibrated_spectra

    def evaluate_element_prediction_on_validation_set(self, 
                                                     calibrated_spectra: np.ndarray, 
                                                     element_data: pd.DataFrame, 
                                                     val_indices: list,
                                                     title_prefix: str = "", 
                                                     timestamp_dir: Optional[str] = None):
        """
        在验证集上评估元素预测模型性能
        """
        print(f"\n评估{title_prefix}元素预测模型在验证集上的性能...")
        
        results = {}
        
        # 确保目录存在
        if timestamp_dir is None:
            timestamp_dir = create_timestamp_directory("results")
        
        output_dir = os.path.join(timestamp_dir, "element_prediction")
        os.makedirs(output_dir, exist_ok=True)
        
        for element in self.element_names:
            if element not in self.element_models:
                continue
                
            model_info = self.element_models[element]
            pls_model = model_info['model']
            
            target_series = element_data[element].apply(
                lambda x: pd.to_numeric(str(x).strip(), errors='coerce')
            )
            
            # 获取验证集数据
            # 注意：这里的calibrated_spectra应该是已经对应val_indices的数据
            # 如果不是，调用者需要传入全量数据并正确索引
            # 在本类的调用逻辑中，通常传入的已经是切片后的数据，所以val_indices主要用于日志或对齐
            
            # 这里的处理假设 calibrated_spectra 和 element_data 已经是对齐的（仅包含验证集部分）
            # 或者 calibrated_spectra 是全量数据。为了通用性，我们假设传入的是已切分好的数据。
            
            # 增加安全检查
            if len(calibrated_spectra) != len(element_data):
                print(f"警告: 光谱数量 ({len(calibrated_spectra)}) 与 元素数据行数 ({len(element_data)}) 不一致，可能导致评估错误。")

            # 安全起见，我们重新提取有效数据
            if len(calibrated_spectra) != len(element_data):
                # 如果长度不一致，说明传入的可能是全量数据，但这里我们假设传入的是切片
                # 为避免混淆，这里假定 element_data 已经是切片后的
                pass

            valid_mask = ~pd.isna(target_series).values
            if not np.any(valid_mask):
                continue
            
            # 提取有效数据
            valid_spectra = calibrated_spectra[valid_mask]
            valid_targets = target_series.values[valid_mask]
            
            if len(valid_spectra) == 0:
                continue

            # 预测
            pred_values = pls_model.predict(valid_spectra).flatten()
            
            # 计算评估指标
            r2 = r2_score(valid_targets, pred_values)
            rmse = np.sqrt(mean_squared_error(valid_targets, pred_values))
            
            denom = np.abs(valid_targets)
            denom = np.where(denom == 0, 1e-10, denom)
            mre = np.mean(np.abs(pred_values - valid_targets) / denom) * 100
            
            results[element] = {
                'r2': r2,
                'rmse': rmse,
                'mre': mre,
                'true_values': valid_targets,
                'pred_values': pred_values
            }
            
            print(f"{element} ({title_prefix}) 验证集: R² = {r2:.4f}, RMSE = {rmse:.4f}, MRE = {mre:.4f}%")
            
            # 绘图
            plt.figure(figsize=(8, 6))
            plt.scatter(valid_targets, pred_values, alpha=0.7)
            plt.plot([valid_targets.min(), valid_targets.max()], 
                     [valid_targets.min(), valid_targets.max()], 'r--', lw=2)
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.title(f'{title_prefix}{element} 验证集预测效果 (R^2 = {r2:.4f})')
            plt.grid(True, alpha=0.3)
            
            plt.text(0.05, 0.95, f'R^2 = {r2:.4f}\nRMSE = {rmse:.4f}\nMRE = {mre:.2f}%', 
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"element_validation_{element}_{title_prefix.strip()}_{timestamp}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()
        
        save_results_to_run_folder(results, timestamp_dir, f"Validation-{title_prefix}-prediction")
        return results

    def train_element_models_with_calibrated_spectra(self, 
                                                   calibrated_spectra: np.ndarray, 
                                                   element_data: pd.DataFrame,
                                                   train_indices: list,
                                                   val_indices: list,
                                                   timestamp_dir: Optional[str] = None):
        """
        使用已校准的光谱数据训练元素含量预测模型
        """
        print("开始训练元素含量预测模型（基于校准光谱）...")
        
        # 1. 识别数值列
        all_columns = element_data.columns.tolist()
        numeric_columns = []
        for col in all_columns:
            if pd.api.types.is_numeric_dtype(element_data[col]):
                numeric_columns.append(col)
            else:
                try:
                    pd.to_numeric(element_data[col].apply(lambda x: str(x).strip() if pd.notnull(x) else x))
                    numeric_columns.append(col)
                except (ValueError, TypeError):
                    print(f"  跳过非数值列: {col}")
        
        self.element_names = [col for col in numeric_columns 
                              if col.lower() not in ['id', 'sample', 'sample_id', 'index', '元素', '名称']]
        
        print(f"检测到以下元素: {self.element_names}")
        
        optimal_components_info = {}
        
        # 2. 训练模型
        train_spectra = calibrated_spectra[train_indices]
        
        for element in self.element_names:
            print(f"正在训练 {element} 的预测模型...")
            
            target_series = element_data[element].apply(
                lambda x: pd.to_numeric(str(x).strip(), errors='coerce')
            )
            
            # 获取训练集数据
            train_targets_full = target_series.iloc[train_indices]
            valid_train_mask = ~pd.isna(train_targets_full).values
            
            if not np.any(valid_train_mask):
                print(f"  警告: {element} 训练集没有有效数据，跳过")
                continue
                
            valid_spectra = train_spectra[valid_train_mask]
            valid_targets = train_targets_full.values[valid_train_mask]
            
            # 寻找最优主成分
            optimal_n = find_optimal_components_for_element(
                valid_spectra, valid_targets, max_components=min(20, len(valid_targets)-1)
            )
            
            # 训练
            pls_model = PLSRegression(n_components=optimal_n)
            pls_model.fit(valid_spectra, valid_targets)
            
            # 训练集评估
            train_pred = pls_model.predict(valid_spectra).flatten()
            train_r2 = r2_score(valid_targets, train_pred)
            train_rmse = np.sqrt(mean_squared_error(valid_targets, train_pred))
            
            self.element_models[element] = {
                'model': pls_model,
                'n_components': optimal_n,
                'r2_score': train_r2,
                'rmse': train_rmse
            }
            
            optimal_components_info[element] = {
                'n_components': optimal_n, 'r2_score': train_r2, 'rmse': train_rmse
            }
            print(f"  {element}: Components={optimal_n}, Train R2={train_r2:.4f}, RMSE={train_rmse:.4f}")
            
            # 绘制训练集图表
            if timestamp_dir:
                output_dir = os.path.join(timestamp_dir, "element_prediction")
                os.makedirs(output_dir, exist_ok=True)
                plt.figure(figsize=(8, 6))
                plt.scatter(valid_targets, train_pred, alpha=0.7)
                plt.plot([valid_targets.min(), valid_targets.max()], [valid_targets.min(), valid_targets.max()], 'r--')
                plt.title(f'Calibrated {element} Train (R2={train_r2:.4f})')
                plt.savefig(os.path.join(output_dir, f"calib_train_{element}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
                plt.close()

        if timestamp_dir:
            save_optimal_components_log(optimal_components_info, timestamp_dir)
            
        # 3. 验证集评估
        val_spectra = calibrated_spectra[val_indices]
        return self.evaluate_element_prediction_on_validation_set(
            val_spectra, element_data.iloc[val_indices],
            val_indices=list(range(len(val_indices))),
            title_prefix="Calib-Spec",
            timestamp_dir=timestamp_dir
        )

    def train_element_models_with_lq_only(self, lq_spectra: np.ndarray, 
                                          element_data: pd.DataFrame,
                                          train_indices: List[int],
                                          val_indices: List[int],
                                          preprocessing_steps: Optional[List[Dict]] = None,
                                          timestamp_dir: Optional[str] = None):
        """
        使用LQ光谱数据训练元素含量预测模型
        """
        print("\n使用原始LQ光谱训练元素含量预测模型...")
        
        # 预处理
        processed_lq = self._apply_preprocessing(lq_spectra, preprocessing_steps)
        
        # 划分数据
        train_lq = processed_lq[train_indices]
        val_lq = processed_lq[val_indices]
        
        # 识别元素 (如果尚未识别)
        if not self.element_names:
            all_columns = element_data.columns.tolist()
            numeric_columns = []
            for col in all_columns:
                if pd.api.types.is_numeric_dtype(element_data[col]):
                    numeric_columns.append(col)
                else:
                    try:
                        pd.to_numeric(element_data[col].apply(lambda x: str(x).strip() if pd.notnull(x) else x))
                        numeric_columns.append(col)
                    except: pass
            self.element_names = [col for col in numeric_columns if col.lower() not in ['id', 'sample', 'sample_id', 'index', '元素', '名称']]

        optimal_components_info = {}
        
        for element in self.element_names:
            target_series = element_data[element].apply(lambda x: pd.to_numeric(str(x).strip(), errors='coerce'))
            
            # 训练数据
            train_targets = target_series.iloc[train_indices]
            valid_mask = ~pd.isna(train_targets).values
            
            if not np.any(valid_mask):
                continue
                
            valid_spectra = train_lq[valid_mask]
            valid_y = train_targets.values[valid_mask]
            
            optimal_n = find_optimal_components_for_element(
                valid_spectra, valid_y, max_components=min(15, len(valid_y)-1)
            )
            
            pls = PLSRegression(n_components=optimal_n)
            pls.fit(valid_spectra, valid_y)
            
            # 记录模型
            train_pred = pls.predict(valid_spectra).flatten()
            r2 = r2_score(valid_y, train_pred)
            rmse = np.sqrt(mean_squared_error(valid_y, train_pred))
            
            self.element_models[element] = {
                'model': pls, 'n_components': optimal_n, 'r2_score': r2, 'rmse': rmse
            }
            optimal_components_info[element] = {'n_components': optimal_n, 'r2_score': r2, 'rmse': rmse}
            
            # 绘图
            if timestamp_dir:
                out_dir = os.path.join(timestamp_dir, "element_prediction")
                os.makedirs(out_dir, exist_ok=True)
                plt.figure()
                plt.scatter(valid_y, train_pred)
                plt.plot([valid_y.min(), valid_y.max()], [valid_y.min(), valid_y.max()], 'r--')
                plt.title(f'LQ-only {element} Train')
                plt.savefig(os.path.join(out_dir, f"lq_train_{element}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
                plt.close()

        if timestamp_dir:
            save_optimal_components_log(optimal_components_info, timestamp_dir)
            
        # 验证
        return self.evaluate_element_prediction_on_validation_set(
            val_lq, element_data.iloc[val_indices],
            val_indices=list(range(len(val_indices))),
            title_prefix="LQ-only",
            timestamp_dir=timestamp_dir
        )

    def train_element_models_with_hq_only(self, hq_spectra: np.ndarray, 
                                         element_data: pd.DataFrame,
                                         train_indices: List[int],
                                         val_indices: List[int],
                                         preprocessing_steps: Optional[List[Dict]] = None,
                                         timestamp_dir: Optional[str] = None):
        """
        使用HQ光谱数据训练元素含量预测模型
        """
        print("\n使用HQ光谱训练元素含量预测模型...")
        processed_hq = self._apply_preprocessing(hq_spectra, preprocessing_steps)
        
        train_hq = processed_hq[train_indices]
        val_hq = processed_hq[val_indices]
        
        if not self.element_names:
            all_columns = element_data.columns.tolist()
            numeric_columns = []
            for col in all_columns:
                if pd.api.types.is_numeric_dtype(element_data[col]):
                    numeric_columns.append(col)
                else:
                    try:
                        pd.to_numeric(element_data[col].apply(lambda x: str(x).strip() if pd.notnull(x) else x))
                        numeric_columns.append(col)
                    except: pass
            self.element_names = [col for col in numeric_columns if col.lower() not in ['id', 'sample', 'sample_id', 'index', '元素', '名称']]

        optimal_components_info = {}
        
        for element in self.element_names:
            target_series = element_data[element].apply(lambda x: pd.to_numeric(str(x).strip(), errors='coerce'))
            train_targets = target_series.iloc[train_indices]
            valid_mask = ~pd.isna(train_targets).values
            
            if not np.any(valid_mask): continue
            
            valid_spectra = train_hq[valid_mask]
            valid_y = train_targets.values[valid_mask]
            
            optimal_n = find_optimal_components_for_element(valid_spectra, valid_y)
            pls = PLSRegression(n_components=optimal_n)
            pls.fit(valid_spectra, valid_y)
            
            train_pred = pls.predict(valid_spectra).flatten()
            r2 = r2_score(valid_y, train_pred)
            rmse = np.sqrt(mean_squared_error(valid_y, train_pred))
            
            self.element_models[element] = {'model': pls, 'n_components': optimal_n, 'r2_score': r2, 'rmse': rmse}
            optimal_components_info[element] = {'n_components': optimal_n, 'r2_score': r2, 'rmse': rmse}
            
            if timestamp_dir:
                out_dir = os.path.join(timestamp_dir, "element_prediction")
                os.makedirs(out_dir, exist_ok=True)
                plt.figure()
                plt.scatter(valid_y, train_pred)
                plt.plot([valid_y.min(), valid_y.max()], [valid_y.min(), valid_y.max()], 'r--')
                plt.title(f'HQ-only {element} Train')
                plt.savefig(os.path.join(out_dir, f"hq_train_{element}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
                plt.close()
                
        if timestamp_dir:
            save_optimal_components_log(optimal_components_info, timestamp_dir)

        return self.evaluate_element_prediction_on_validation_set(
            val_hq, element_data.iloc[val_indices],
            val_indices=list(range(len(val_indices))),
            title_prefix="HQ-only",
            timestamp_dir=timestamp_dir
        )

    def evaluate_element_models_with_dataset_name(self, calibrated_spectra: np.ndarray, element_data: pd.DataFrame, 
                              dataset_name: str = "验证集 ",
                              timestamp_dir: Optional[str] = None):
        """
        评估已训练的元素含量预测模型 (包装方法)
        """
        indices = list(range(len(calibrated_spectra)))
        self.evaluate_element_prediction_on_validation_set(
            calibrated_spectra, element_data, indices, 
            title_prefix=dataset_name, timestamp_dir=timestamp_dir
        )

if __name__ == "__main__":
    print("请使用 main_plsr_calibrator.py 运行完整流程。")