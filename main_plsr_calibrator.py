import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import matplotlib.pyplot as plt
import json

# 导入各模块
from plsr_model import PLSRSpectralModel, find_optimal_components
from spectral_preprocessing import SpectralPreprocessor, load_spectral_data_from_csv, resample_to_reference
from element_prediction_pipeline import ElementPredictionPipeline, load_element_data
from evaluation_visualization import create_timestamp_directory, plot_results
from preprocessing_visualization import visualize_complete_preprocessing_pipeline

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_config(config_path="config.json"):
    """加载配置文件"""
    if not os.path.exists(config_path):
        print(f"⚠️ 警告: 找不到配置文件 {config_path}，请确保文件存在。")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    print("="*80)
    print("      LIBS 光谱校准与元素预测系统 (完整架构版)")
    print("="*80)
    
    # 1. 加载配置
    config = load_config()
    if config is None:
        return

    # 配置路径 (从 config 读取)
    base_dir = config['paths']['base_dir']
    lq_dir = os.path.join(base_dir, config['paths']['lq_dir_name'])
    hq_dir = os.path.join(base_dir, config['paths']['hq_dir_name'])
    element_file_path = os.path.join(base_dir, config['paths']['element_file_name'])
    output_dir_name = config['paths'].get('output_dir', 'results')
    
    # 创建结果目录
    timestamp_dir = create_timestamp_directory(output_dir_name)
    print(f"📁 结果输出目录: {timestamp_dir}")

    # 2. 数据加载 (自动扫描文件)
    print("\n[Step 1] 数据加载与对齐...")
    if not os.path.exists(lq_dir):
        print(f"❌ 错误: 目录不存在 {lq_dir}")
        return

    sample_files = [f for f in os.listdir(lq_dir) if f.endswith('.csv')]
    sample_ids = sorted([os.path.splitext(f)[0] for f in sample_files])
    print(f"   检测到 {len(sample_ids)} 个样品文件")

    # 加载原始数据
    lq_raw, hq_raw, wl_common = load_spectral_data_from_csv(lq_dir, hq_dir, sample_ids)
    
    # 使用自动识别的公共波长范围 (无需手动裁剪)
    hq_wl_trim = wl_common
    hq_trim = hq_raw
    
    # 重采样 LQ -> HQ
    print("   正在执行三次样条重采样 (LQ -> HQ)...")
    resample_method = config['preprocessing'].get('resampling_method', 'cubic_spline')
    lq_resampled, _ = resample_to_reference(lq_raw, wl_common, hq_wl_trim, method=resample_method)
    
    # HQ 平均化 (Samples, Replicates, Pixels -> Samples, Pixels)
    if hq_trim.ndim == 3:
        hq_avg = np.mean(hq_trim, axis=1)
    else:
        hq_avg = hq_trim

    # 3. 预处理与可视化 (功能 F)
    print("\n[Step 2] 预处理可视化分析...")
    preprocessor = SpectralPreprocessor()
    # 定义预处理流
    steps = config['preprocessing']['steps']
    
    # 为可视化生成数据 (取第一个样品)
    viz_list = [(lq_resampled[0], "原始LQ")]
    temp = lq_resampled[0].copy()
    for s in steps:
        if s['method'] == 'baseline_correction':
            temp = preprocessor.baseline_correction(temp, **s['params'])
        elif s['method'] == 'smoothing':
            temp = preprocessor.smoothing(temp, **s['params'])
        elif s['method'] == 'snv_normalization':
            temp = preprocessor.snv_normalization(temp)
        viz_list.append((temp, s['name']))
        
    visualize_complete_preprocessing_pipeline(
        viz_list, hq_wl_trim, sample_idx=0, 
        output_dir=os.path.join(timestamp_dir, "overall_preprocessing")
    )
    print("   ✅ 预处理流水线图已生成")

    # 对全量数据应用预处理
    lq_proc = lq_resampled.copy()
    hq_proc = hq_avg.copy()
    # 简单循环处理所有样品
    for i in range(len(lq_proc)):
        for s in steps:
            if s['method'] == 'baseline_correction': lq_proc[i] = preprocessor.baseline_correction(lq_proc[i], **s['params'])
            elif s['method'] == 'smoothing': lq_proc[i] = preprocessor.smoothing(lq_proc[i], **s['params'])
            elif s['method'] == 'snv_normalization': lq_proc[i] = preprocessor.snv_normalization(lq_proc[i])
            
    # 4. 模型训练与寻优 (功能 B)
    print("\n[Step 3] 训练光谱校准模型...")
    # 划分数据集
    test_size = config['model'].get('test_size', 0.2)
    random_state = config['model'].get('random_state', 42)
    train_lq, val_lq, train_idx, val_idx = train_test_split(lq_proc, range(len(lq_proc)), test_size=test_size, random_state=random_state)
    train_hq, val_hq = hq_avg[train_idx], hq_avg[val_idx] # HQ通常不做复杂预处理作为Target，或者做同样的

    # 4.1 自动寻优
    print("   >>> 正在进行 LOO-CV 自动寻找最优主成分数...")
    max_comp = config['model'].get('max_components', 15)
    optimal_n = find_optimal_components(train_lq, train_hq, max_components=max_comp, task_type='calibration', timestamp_dir=timestamp_dir)
    print(f"   ✅ 最优主成分数: {optimal_n}")
    
    # 4.2 训练
    calib_model = PLSRSpectralModel(n_components=optimal_n)
    calib_model.fit(train_lq, train_hq)
    
    # 4.3 评估
    val_pred = calib_model.predict(val_lq)
    print(f"   光谱校准验证集 R²: {r2_score(val_hq.flatten(), val_pred.flatten()):.4f}")
    plot_results(val_lq, val_hq, val_pred, hq_wl_trim, sample_idx=0, title="校准效果", timestamp_dir=timestamp_dir)

    # 5. 多模式元素预测 (功能 C)
    print("\n[Step 4] 执行元素预测 (LQ-only vs Calib-Spec vs HQ-only)...")
    element_data = load_element_data(element_file_path)
    if element_data is None: return

    pipeline = ElementPredictionPipeline(spectral_model=calib_model)
    
    # 模式1: LQ-only
    print("\n   [Mode 1] LQ-only (基准)")
    res_lq = pipeline.train_element_models_with_lq_only(lq_proc, element_data, train_idx, val_idx, [], timestamp_dir)
    
    # 模式2: Calib-Spec
    print("\n   [Mode 2] Calib-Spec (核心)")
    # 生成全量校准光谱
    lq_calibrated = calib_model.predict(lq_proc)
    res_calib = pipeline.train_element_models_with_calibrated_spectra(lq_calibrated, element_data, train_idx, val_idx, timestamp_dir)
    
    # 模式3: HQ-only
    print("\n   [Mode 3] HQ-only (上限)")
    res_hq = pipeline.train_element_models_with_hq_only(hq_avg, element_data, train_idx, val_idx, [], timestamp_dir)

    # 6. 打印总结
    print("\n[Step 5] 总结 - 关键元素 (SiO2) R² 对比:")
    elem = 'SiO2 ' # 确保列名匹配
    if elem in res_lq:
        print(f"   LQ-only : {res_lq[elem]['r2']:.4f}")
        print(f"   Calib   : {res_calib[elem]['r2']:.4f}")
        print(f"   HQ-only : {res_hq[elem]['r2']:.4f}")
        
    print(f"\n✅ 完整流程结束！请查看结果文件夹: {timestamp_dir}")

if __name__ == "__main__":
    main()