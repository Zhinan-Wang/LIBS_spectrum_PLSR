import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import matplotlib.pyplot as plt
import json
from typing import List

# 导入各模块
from plsr_model import PLSRSpectralModel, train_calibration_model
from spectral_preprocessing import SpectralPreprocessor, load_spectral_data_from_csv, resample_to_reference, evaluate_resampling_reliability
from element_prediction_pipeline import ElementPredictionPipeline, load_element_data
from evaluation_visualization import create_timestamp_directory, plot_results, plot_performance_comparison, plot_prediction_scatter_comparison
from preprocessing_visualization import visualize_complete_preprocessing_pipeline, visualize_preprocessing_step, visualize_resampling_quality, visualize_data_alignment

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_config(config_path="config.json"):
    """加载配置文件"""
    if not os.path.exists(config_path):
        print(f"⚠️ 警告: 找不到配置文件 {config_path}，请确保文件存在。")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = ""
    # 尝试多种编码读取，优先 utf-8-sig 以处理 BOM (Windows 常见问题)
    for encoding in ['utf-8-sig', 'utf-8', 'gbk']:
        try:
            with open(config_path, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
            
    if not content:
        print(f"❌ 错误: 无法读取配置文件 {config_path} (尝试了 utf-8, gbk)")
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"❌ 配置文件 JSON 解析失败: {e}")
        # 尝试定位错误行
        lines = content.split('\n')
        if 0 <= e.lineno - 1 < len(lines):
            print(f"   错误位置: 第 {e.lineno} 行附近")
            print(f"   >> {lines[e.lineno - 1].strip()}")
        return None

def align_element_data(element_data: pd.DataFrame, sample_ids: List[str]) -> pd.DataFrame:
    """
    核心修复：根据 sample_ids 对 element_data 进行强制对齐
    防止因 Excel 顺序与文件名顺序不一致导致 X-Y 错位
    """
    # 1. 寻找 ID 列 (自动识别)
    id_col = None
    # 常见的 ID 列名候选
    candidates = ['sample', 'sample_id', 'id', 'no', 'name', '编号', '样品名称', '样品编号', 'index', '序号']
    
    # 策略A: 匹配列名
    for col in element_data.columns:
        if str(col).strip().lower() in candidates:
            id_col = col
            break
            
    # 策略B: 如果没找到，检查列值与 sample_ids 的重叠度
    if not id_col:
        for col in element_data.columns:
            try:
                col_values = element_data[col].astype(str).str.strip().values
                overlap = sum(1 for sid in sample_ids if str(sid).strip() in col_values)
                if overlap / len(sample_ids) > 0.5: # 超过50%匹配
                    id_col = col
                    break
            except Exception:
                continue
    
    if id_col:
        print(f"   [Data Alignment] 自动识别样品ID列: '{id_col}'")
        # 创建副本以免修改原始数据
        df_aligned = element_data.copy()
        # 统一转为字符串去空格
        df_aligned[id_col] = df_aligned[id_col].astype(str).str.strip()
        
        # 设为索引并按 sample_ids 重排
        df_aligned = df_aligned.set_index(id_col)
        
        # 关键步骤：reindex 会按照 sample_ids 的顺序重排数据
        # 如果某个 sample_id 在 Excel 中不存在，对应行会变成 NaN (后续会被过滤)
        # 确保 sample_ids 也是字符串格式
        sample_ids_str = [str(s).strip() for s in sample_ids]
        df_aligned = df_aligned.reindex(sample_ids_str)
        
        # 检查缺失情况
        missing_count = df_aligned.isnull().all(axis=1).sum()
        if missing_count > 0:
            print(f"   ⚠️ 警告: 对齐后有 {missing_count} 个样品在元素表中未找到数据 (将被跳过)")
        
        # 重置索引，保持 DataFrame 结构
        return df_aligned.reset_index()
    else:
        print("   ⚠️ 严重警告: 未能自动识别样品ID列！假设元素表顺序与光谱文件顺序一致。")
        print("      (如果 R2 很低，请检查 Excel 第一列是否为样品编号，且与文件名一致)")
        if len(element_data) != len(sample_ids):
            print(f"      注意: 元素表行数 ({len(element_data)}) 与 样品数 ({len(sample_ids)}) 不一致，极大概率错位！")
        return element_data

def plot_component_counts(res_lq, res_calib, res_hq, timestamp_dir, res_calib_self=None):
    """
    绘制各模式下各元素的主成分数对比图
    """
    if not timestamp_dir: return
    
    save_dir = os.path.join(timestamp_dir, "model_analysis")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 汇总数据
    models = [('LQ-only', res_lq), ('Calib-Spec', res_calib), ('HQ-only', res_hq)]
    if res_calib_self:
        models.append(('Calib-Self', res_calib_self))
        
    all_elements = set()
    for _, res in models:
        if res:
            all_elements.update(res.keys())
    
    sorted_elements = sorted(list(all_elements))
    
    # 准备绘图数据
    plot_data = {elem: [] for elem in sorted_elements}
    
    for name, res in models:
        for elem in sorted_elements:
            if res and elem in res:
                n = res[elem].get('n_components', 0)
                plot_data[elem].append(n)
            else:
                plot_data[elem].append(0)
                
    # --- 图1: 综合对比图 (Grouped Bar Chart) ---
    x = np.arange(len(sorted_elements))
    total_width = 0.8
    n_models = len(models)
    width = total_width / n_models
    
    plt.figure(figsize=(max(12, len(sorted_elements)*0.8), 6), dpi=300)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (name, _) in enumerate(models):
        vals = [plot_data[elem][i] for elem in sorted_elements]
        bar_x = x - (total_width / 2) + (i * width) + (width / 2)
        plt.bar(bar_x, vals, width, label=name, alpha=0.8, color=colors[i % len(colors)])
        
    plt.xlabel('Elements')
    plt.ylabel('Number of Components')
    plt.title('Optimal Components by Element and Model')
    plt.xticks(x, sorted_elements, rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "components_comparison_all.png"))
    plt.close()
    
    # --- 图2-5: 各模式单独图 ---
    for i, (name, res) in enumerate(models):
        if not res: continue
        
        elems = []
        comps = []
        for e in sorted_elements:
            if e in res:
                elems.append(e)
                comps.append(res[e].get('n_components', 0))
        
        if not elems: continue
        
        plt.figure(figsize=(max(10, len(elems)*0.6), 5), dpi=300)
        bars = plt.bar(elems, comps, color=colors[i % len(colors)], edgecolor='black', alpha=0.7)
        plt.xlabel('Elements')
        plt.ylabel('Number of Components')
        plt.title(f'Optimal Components - {name}')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{int(height)}',
                     ha='center', va='bottom', fontsize=9)
                     
        plt.tight_layout()
        safe_name = name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(save_dir, f"components_{safe_name}.png"))
        plt.close()
        
    print(f"   [Plot] 主成分数对比图已保存至: {save_dir}")

def plot_cv_curves(res_lq, res_calib, res_hq, timestamp_dir, res_calib_self=None):
    """
    生成各模式下各元素的 CV 寻优曲线 (RMSE vs Components)
    1. 生成分面图 (Grid Plot): 每个元素一张子图，显示原始 RMSE 值
    2. 生成综合图 (Combined Plot): 所有元素归一化后画在同一张图，便于比较趋势
    """
    if not timestamp_dir: return
    
    save_dir = os.path.join(timestamp_dir, "model_analysis", "cv_curves")
    os.makedirs(save_dir, exist_ok=True)
    
    models = [('LQ-only', res_lq), ('Calib-Spec', res_calib), ('HQ-only', res_hq)]
    if res_calib_self:
        models.append(('Calib-Self', res_calib_self))
        
    count = 0
    for mode_name, results in models:
        if not results: continue
        
        elements = sorted([e for e in results.keys() if 'cv_history' in results[e]])
        if not elements: continue
        
        # --- 1. 原始分面图 (Grid Plot) ---
        n_elems = len(elements)
        cols = 4
        rows = (n_elems + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows), dpi=200)
        axes = axes.flatten()
        
        for i, elem in enumerate(elements):
            ax = axes[i]
            data = results[elem]
            hist = data['cv_history']
            
            x = hist['components']
            y = hist['scores'] # RMSE
            opt_n = data.get('n_components', 0)
            
            # 绘制曲线
            ax.plot(x, y, 'b.-', alpha=0.7, linewidth=1)
            
            # 标记选定的点
            if opt_n in x:
                idx = x.index(opt_n)
                ax.plot(x[idx], y[idx], 'ro', markersize=6, label=f'Selected: {opt_n}')
            
            ax.set_title(f"{elem} (n={opt_n})")
            ax.set_xlabel("Components")
            ax.set_ylabel("CV RMSE")
            ax.grid(True, alpha=0.3)
            
        # 隐藏多余的子图
        for j in range(n_elems, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.suptitle(f"CV Optimization Curves - {mode_name}", y=1.02, fontsize=16)
        safe_name = mode_name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(save_dir, f"cv_curves_grid_{safe_name}.png"), bbox_inches='tight')
        plt.close()
        
        # --- 2. 综合对比图 (Combined Plot) ---
        # 为了在同一坐标轴显示，我们将 RMSE 归一化到 [0, 1] 区间
        # 这样可以比较不同元素的收敛趋势
        fig2, ax2 = plt.subplots(figsize=(14, 8), dpi=300)
        
        # 使用 Tab20 颜色表以支持更多元素区分 (最多20种颜色循环)
        cmap = plt.get_cmap('tab20')
        colors = cmap(np.linspace(0, 1, len(elements)))
        
        for idx, elem in enumerate(elements):
            data = results[elem]
            hist = data['cv_history']
            x = np.array(hist['components'])
            y = np.array(hist['scores'])
            opt_n = data.get('n_components', 0)
            
            # 过滤无效值 (inf/nan)
            mask = np.isfinite(y)
            if not np.any(mask): continue
            
            x_plot = x[mask]
            y_plot = y[mask]
            
            # 归一化: (y - min) / (max - min)
            y_min, y_max = np.min(y_plot), np.max(y_plot)
            if y_max > y_min:
                y_norm = (y_plot - y_min) / (y_max - y_min)
            else:
                y_norm = np.zeros_like(y_plot)
            
            # 绘图 (自动颜色循环)
            ax2.plot(x_plot, y_norm, '.-', alpha=0.7, linewidth=1.5, label=f"{elem} (n={opt_n})", color=colors[idx])
            
            # 标记最佳点
            if opt_n in x_plot:
                opt_idx = np.where(x_plot == opt_n)[0][0]
                # 使用对应颜色绘制星号
                ax2.plot(x_plot[opt_idx], y_norm[opt_idx], '*', color=colors[idx], markersize=12, markeredgecolor='white', markeredgewidth=1, zorder=10)

        ax2.set_xlabel("Number of Components", fontsize=12)
        ax2.set_ylabel("Normalized CV RMSE (0=Best, 1=Worst)", fontsize=12)
        ax2.set_title(f"CV Optimization Trends (Normalized) - {mode_name}", fontsize=14, fontweight='bold')
        # 将图例放在图外，防止遮挡
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=10)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, f"cv_curves_combined_{safe_name}.png"), bbox_inches='tight')
        plt.close()
        
        count += 1
        
    if count > 0:
        print(f"   [Plot] CV 寻优曲线图已保存至: {save_dir} (共 {count} 组)")
    else:
        print(f"   [Plot] ⚠️ 未生成 CV 曲线图 (可能是因为结果中缺少 cv_history 数据)")

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

    # 保存本次运行的配置快照
    config_save_path = os.path.join(timestamp_dir, "config_snapshot.json")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"   [Config] 配置快照已保存: {config_save_path}")

    # 2. 数据加载 (自动扫描文件)
    print("\n[Step 1] 数据加载与对齐...")
    if not os.path.exists(lq_dir):
        print(f"❌ 错误: 目录不存在 {lq_dir}")
        return

    sample_files = [f for f in os.listdir(lq_dir) if f.endswith('.csv')]
    sample_ids = sorted([os.path.splitext(f)[0] for f in sample_files])
    print(f"   检测到 {len(sample_ids)} 个样品文件")

    # 加载原始数据
    lq_raw, hq_raw, lq_wl_raw, wl_common, valid_ids = load_spectral_data_from_csv(lq_dir, hq_dir, sample_ids)
    sample_ids = valid_ids # 更新为实际加载成功的样品ID列表
    print(f"   成功加载 {len(sample_ids)} 对有效数据")

    # 使用自动识别的公共波长范围 (无需手动裁剪)
    hq_wl_trim = wl_common
    hq_trim = hq_raw
    
    # --- 智能重采样策略选择 ---
    config_method = config['preprocessing'].get('resampling_method', 'cubic_spline')
    
    candidates = []
    if isinstance(config_method, list):
        candidates = config_method
    elif config_method == "auto":
        candidates = ["cubic_spline", "pchip", "akima", "linear"]
    else:
        candidates = [str(config_method)]
    
    if not candidates: candidates = ["cubic_spline"]

    print(f"   [Resampling] 准备执行重采样 (配置模式: {config_method})...")
    
    best_method = candidates[0]
    best_metrics = {}
    best_rmse = float('inf')
    eval_idx = 0 # 选取第一个样品进行评估

    if len(candidates) > 1:
        print(f"   [Auto-Selection] 正在对比 {len(candidates)} 种重采样算法...")
        print(f"   {'Method':<15} | {'CV-RMSE':<10} | {'CV-MAPE':<10}")
        print("-" * 45)
        
        for method in candidates:
            metrics = evaluate_resampling_reliability(lq_wl_raw, lq_raw[eval_idx], method=method)
            print(f"   {method:<15} | {metrics['cv_rmse']:.4f}     | {metrics['cv_mape']:.2f}%")
            
            if metrics['cv_rmse'] < best_rmse:
                best_rmse = metrics['cv_rmse']
                best_method = method
                best_metrics = metrics
        print("-" * 45)
        print(f"   ✅ 自动选择最佳方法: {best_method} (RMSE={best_rmse:.4f})")
    else:
        best_method = candidates[0]
        print(f"   [Manual Mode] 已锁定重采样方法: {best_method} (正在评估保真度...)")
        best_metrics = evaluate_resampling_reliability(lq_wl_raw, lq_raw[eval_idx], method=best_method)
        print(f"      CV-RMSE: {best_metrics['cv_rmse']:.4f}, CV-MAPE: {best_metrics['cv_mape']:.2f}%")

    # 使用选定的最佳方法进行全量重采样
    print(f"   正在执行 {best_method} 重采样 (LQ -> HQ)...")
    lq_resampled, _ = resample_to_reference(lq_raw, lq_wl_raw, hq_wl_trim, method=best_method)
    
    # --- 可视化：原始 vs 裁剪 vs 重采样 ---
    print("   [Visualization] 生成数据加载与对齐图...")
    mask_trim = (lq_wl_raw >= np.min(wl_common)) & (lq_wl_raw <= np.max(wl_common))
    lq_wl_trim = lq_wl_raw[mask_trim]
    lq_spec_trim = lq_raw[eval_idx][mask_trim]
    
    visualize_data_alignment(
        lq_wl_raw, lq_raw[eval_idx],
        lq_wl_trim, lq_spec_trim,
        hq_wl_trim, lq_resampled[eval_idx],
        sample_idx=eval_idx,
        output_dir=os.path.join(timestamp_dir, "data_alignment")
    )
    
    # --- 生成评估图 ---
    visualize_resampling_quality(
        orig_wl=lq_wl_raw, orig_spec=lq_raw[eval_idx],
        resamp_wl=hq_wl_trim, resamp_spec=lq_resampled[eval_idx],
        metrics=best_metrics, sample_idx=eval_idx,
        output_dir=os.path.join(timestamp_dir, "resampling_evaluation")
    )
    
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
    
    step_viz_dir = os.path.join(timestamp_dir, "step_by_step_preprocessing")
    
    for s in steps:
        method_name = s['method']
        params = s.get('params', {})
        step_name = s['name']
        
        if hasattr(preprocessor, method_name):
            prev_temp = temp.copy()
            # 动态调用预处理方法，支持 config.json 中定义的所有方法
            temp = getattr(preprocessor, method_name)(temp, **params)
            
            # 记录到总列表
            viz_list.append((temp, step_name))
            
            # 生成单步对比图 (Before vs After)
            visualize_preprocessing_step(
                original_spectrum=prev_temp,
                processed_spectrum=temp,
                wavelengths=hq_wl_trim,
                step_name=step_name,
                sample_idx=0,
                output_dir=step_viz_dir
            )
        else:
            print(f"⚠️ 警告: 未知预处理方法 {method_name}，跳过可视化。")
        
    visualize_complete_preprocessing_pipeline(
        viz_list, hq_wl_trim, sample_idx=0, 
        output_dir=os.path.join(timestamp_dir, "overall_preprocessing")
    )
    print("   ✅ 预处理流水线图及单步对比图已生成")

    # 4. 模型训练与寻优 (功能 B)
    print("\n[Step 3] 训练光谱校准模型...")
    
    # 4.0 先划分数据集 (Indices)
    manual_split = config['model'].get('manual_split', {})
    
    if manual_split.get('enabled', False):
        print("   ⚠️ 使用手动数据集划分 (Configured in config.json)")
        train_names = manual_split.get('train_samples', [])
        test_names = manual_split.get('test_samples', [])
        
        # 建立名称到索引的映射
        name_to_idx = {name: i for i, name in enumerate(sample_ids)}
        all_indices = set(range(len(sample_ids)))
        
        train_idx_set = set()
        val_idx_set = set()
        
        # 解析配置中的样品名
        for name in train_names:
            name_str = str(name)
            if name_str in name_to_idx: train_idx_set.add(name_to_idx[name_str])
            else: print(f"   [Warning] 训练集样品未找到: {name}")
                
        for name in test_names:
            name_str = str(name)
            if name_str in name_to_idx: val_idx_set.add(name_to_idx[name_str])
            else: print(f"   [Warning] 测试集样品未找到: {name}")
        
        # 自动补全逻辑 (互斥补全)
        if train_idx_set and not val_idx_set:
            val_idx_set = all_indices - train_idx_set
            print(f"   自动分配剩余 {len(val_idx_set)} 个样品到测试集")
        elif val_idx_set and not train_idx_set:
            train_idx_set = all_indices - val_idx_set
            print(f"   自动分配剩余 {len(train_idx_set)} 个样品到训练集")
            
        if not train_idx_set or not val_idx_set:
            raise ValueError("手动划分配置错误：训练集或测试集为空，请检查 config.json 中的样品名称。")
            
        train_idx = sorted(list(train_idx_set))
        val_idx = sorted(list(val_idx_set))
    else:
        test_size = config['model'].get('test_size', 0.2)
        random_state = config['model'].get('random_state', 42)
        # 仅划分索引
        train_idx, val_idx = train_test_split(range(len(lq_resampled)), test_size=test_size, random_state=random_state)

    # 提取初始数据 (Raw)
    train_lq = lq_resampled[train_idx].copy()
    val_lq = lq_resampled[val_idx].copy()
    train_hq = hq_avg[train_idx].copy()
    val_hq = hq_avg[val_idx].copy()

    # 4.1 再应用预处理 (分别处理训练集和验证集，防止泄漏)
    for s in steps:
        method_name = s['method']
        params = s.get('params', {})
        if hasattr(preprocessor, method_name):
            print(f"   执行预处理: {s['name']} (Train/Val 分离处理)")
            train_lq = getattr(preprocessor, method_name)(train_lq, **params)
            val_lq = getattr(preprocessor, method_name)(val_lq, **params)
            train_hq = getattr(preprocessor, method_name)(train_hq, **params)
            val_hq = getattr(preprocessor, method_name)(val_hq, **params)

    # 4.2 重组全量数据 (用于后续 Pipeline)
    lq_proc = np.zeros((len(lq_resampled), train_lq.shape[1]))
    lq_proc[train_idx] = train_lq
    lq_proc[val_idx] = val_lq
    
    hq_proc = np.zeros((len(hq_avg), train_hq.shape[1]))
    hq_proc[train_idx] = train_hq
    hq_proc[val_idx] = val_hq

    # 4.1 自动寻优
    print("   >>> 正在训练光谱校准模型...")
    
    # 获取校准配置
    calib_config = config['model'].get('calibration', {})
    # 兼容旧配置 (如果 calibration 不存在)
    if not calib_config:
        calib_config = {
            "method": "PLSR",
            "params": config['model'] # 尝试使用顶层参数
        }
        
    calib_params = calib_config.get('params', {})
    learn_diff = calib_params.get('learn_difference', False)
    feature_selection_config = config['model'].get('feature_selection', {"enabled": False})
    mode_strategies = config['model'].get('mode_strategies', {})
    mode_fs_configs = config['model'].get('mode_feature_selection', {})
    
    if learn_diff:
        print("   [Strategy] 启用差异学习 (Difference Learning: HQ - LQ)...")
        train_target = train_hq - train_lq
        X_base_for_cv = train_lq
    else:
        print("   [Strategy] 标准直接学习 (Direct Learning: HQ)...")
        train_target = train_hq
        X_base_for_cv = None
        
    # 调用通用训练函数
    calib_model, calib_metrics = train_calibration_model(
        train_lq, train_target, 
        config=calib_config, 
        timestamp_dir=timestamp_dir, 
        X_base=X_base_for_cv
    )
    
    print(f"   ✅ 校准模型训练完成 (Params: {calib_metrics.get('n_components', 'N/A')}, Score: {calib_metrics.get('score', 0):.4f})")
    
    # 4.3 评估
    if learn_diff:
        val_diff_pred = calib_model.predict(val_lq)
        val_pred = val_lq + val_diff_pred # 重构: LQ + Predicted_Diff
    else:
        val_pred = calib_model.predict(val_lq)
        
    print(f"   光谱校准验证集 R²: {r2_score(val_hq.flatten(), val_pred.flatten()):.4f}")
    plot_results(val_lq, val_hq, val_pred, hq_wl_trim, sample_idx=0, title=f"校准效果 ({'Diff' if learn_diff else 'Direct'})", timestamp_dir=timestamp_dir)

    # 5. 多模式元素预测 (功能 C)
    print("\n[Step 4] 执行元素预测 (LQ-only vs Calib-Spec vs HQ-only)...")
    element_data = load_element_data(element_file_path)
    if element_data is None: return

    # --- 关键修复：执行数据对齐 ---
    element_data = align_element_data(element_data, sample_ids)

    # 获取预测配置
    pred_config = config['model'].get('prediction', {})
    if not pred_config:
        pred_config = {"method": "PLSR", "params": config['model']}

    pipeline = ElementPredictionPipeline(
        spectral_model=calib_model, 
        prediction_config=pred_config,
        feature_selection_config=feature_selection_config, 
        wavelengths=hq_wl_trim
    )
    
    # 模式1: LQ-only
    print("\n   [Mode 1] LQ-only (基准)")
    strat = mode_strategies.get('LQ-only', None)
    fs_cfg = mode_fs_configs.get('LQ-only', feature_selection_config)
    res_lq = pipeline.train_element_models_with_lq_only(lq_proc, element_data, train_idx, val_idx, timestamp_dir, selection_method=strat, feature_selection_config=fs_cfg)
    
    # 模式2: Calib-Spec
    print("\n   [Mode 2] Calib-Spec (核心: Train on HQ, Test on Calib-LQ)")
    strat = mode_strategies.get('Calib-Spec', None)
    # 生成全量校准光谱
    if learn_diff:
        lq_calibrated_diff = calib_model.predict(lq_proc)
        lq_calibrated = lq_proc + lq_calibrated_diff
    else:
        lq_calibrated = calib_model.predict(lq_proc)
        
    fs_cfg = mode_fs_configs.get('Calib-Spec', feature_selection_config)
    res_calib = pipeline.train_element_models_hq_train_calib_test(hq_proc, lq_calibrated, element_data, train_idx, val_idx, timestamp_dir, selection_method=strat, feature_selection_config=fs_cfg)
    
    # 模式3: HQ-only
    print("\n   [Mode 3] HQ-only (上限)")
    strat = mode_strategies.get('HQ-only', None)
    fs_cfg = mode_fs_configs.get('HQ-only', feature_selection_config)
    res_hq = pipeline.train_element_models_with_hq_only(hq_proc, element_data, train_idx, val_idx, timestamp_dir, selection_method=strat, feature_selection_config=fs_cfg)

    # 模式4: Calib-Self (实用模式)
    print("\n   [Mode 4] Calib-Self (实用: Train on Calib-LQ, Test on Calib-LQ)")
    strat = mode_strategies.get('Calib-Self', None)
    fs_cfg = mode_fs_configs.get('Calib-Self', feature_selection_config)
    res_calib_self = pipeline.train_element_models_with_calibrated_spectra(lq_calibrated, element_data, train_idx, val_idx, timestamp_dir, selection_method=strat, feature_selection_config=fs_cfg)

    # 5.1 生成对比图表
    print("\n[Step 5] 生成综合对比分析图...")
    plot_performance_comparison(res_lq, res_calib, res_hq, timestamp_dir, res_calib_self)
    plot_component_counts(res_lq, res_calib, res_hq, timestamp_dir, res_calib_self)
    plot_cv_curves(res_lq, res_calib, res_hq, timestamp_dir, res_calib_self)
    
    # 安全计算共同元素 (防止 res_calib_self 为 None)
    keys_sets = [set(res_lq.keys()), set(res_calib.keys()), set(res_hq.keys())]
    if res_calib_self:
        keys_sets.append(set(res_calib_self.keys()))
    common_elements = set.intersection(*keys_sets)
    
    for elem in common_elements:
        plot_prediction_scatter_comparison(res_lq, res_calib, res_hq, elem, timestamp_dir, res_calib_self)

    # 6. 打印总结
    print("\n[Summary] 关键元素 (SiO2) R² 对比:")
    # 尝试查找 SiO2 (处理可能的空格或命名差异)
    sio2_candidates = ['SiO2', 'SiO2 ', 'Si', 'Si ']
    target_elem = next((e for e in sio2_candidates if e in res_lq), None)
    
    if target_elem:
        print(f"   Element : {target_elem}")
        print(f"   LQ-only : {res_lq[target_elem]['r2']:.4f}")
        print(f"   Calib   : {res_calib[target_elem]['r2']:.4f} (Mode 2)")
        if res_calib_self and target_elem in res_calib_self:
            print(f"   Calib-S : {res_calib_self[target_elem]['r2']:.4f} (Mode 4)")
        print(f"   HQ-only : {res_hq[target_elem]['r2']:.4f}")
    else:
        print("   (未找到 SiO2 相关元素，跳过展示)")
        
    print(f"\n✅ 完整流程结束！请查看结果文件夹: {timestamp_dir}")

if __name__ == "__main__":
    main()