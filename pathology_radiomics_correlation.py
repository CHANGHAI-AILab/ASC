"""
病理特征与筛选后影像组学特征的相关性分析
分析11个病理特征与影像组学模型筛选后的影像特征之间的相关性
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("病理特征与影像组学模型筛选后特征的相关性分析")
print("=" * 80)

# ============================================================
# 1. 读取数据
# ============================================================
print("\n[步骤1] 读取数据...")

# 读取合并后的数据（包含影像特征）
df_merge = pd.read_excel(r'D:\JMC\病理影像特征汇总merge.xlsx')

# 读取病理特征数据
df_pathology = pd.read_excel(r'D:\JMC\病理特征汇总.xlsx')

# 11个病理特征
pathology_features = ['C_raw', 'D1_raw', 'D2_raw', 'D3_raw', 'D4_raw', 
                      'D5_raw', 'B_raw', 'S1_raw', 'S2_raw', 'S3_raw', 'V_raw']

print(f"病理特征数量: {len(pathology_features)}")
print(f"病理特征列表: {pathology_features}")

# 读取影像组学模型筛选后的特征
radiomics_selected = pd.read_csv('selected_features_radiomics_final.csv')
all_selected_radiomics = radiomics_selected['Feature'].tolist()

print(f"\n影像组学模型筛选特征数: {len(all_selected_radiomics)}")
print(f"筛选后的影像组学特征列表:")
for i, feat in enumerate(all_selected_radiomics, 1):
    print(f"  {i}. {feat}")

# ============================================================
# 2. 合并数据
# ============================================================
print("\n[步骤2] 合并病理特征和影像特征数据...")

# 确保两个数据框按patient_id对齐
df_analysis = df_merge[['patient_id', 'dataset'] + all_selected_radiomics].copy()
df_analysis = df_analysis.merge(df_pathology[['patient_id'] + pathology_features], 
                                 on='patient_id', how='inner')

print(f"合并后数据形状: {df_analysis.shape}")
print(f"样本数: {len(df_analysis)}")

# ============================================================
# 3. 计算相关性矩阵
# ============================================================
print("\n[步骤3] 计算相关性...")

# 提取数值数据
X_radiomics = df_analysis[all_selected_radiomics].values
X_pathology = df_analysis[pathology_features].values

# 计算Pearson相关系数
print("\n计算Pearson相关系数...")
pearson_corr_matrix = np.zeros((len(pathology_features), len(all_selected_radiomics)))
pearson_pval_matrix = np.zeros((len(pathology_features), len(all_selected_radiomics)))

for i, pathology_feat in enumerate(pathology_features):
    for j, radiomics_feat in enumerate(all_selected_radiomics):
        corr, pval = pearsonr(df_analysis[pathology_feat], df_analysis[radiomics_feat])
        pearson_corr_matrix[i, j] = corr
        pearson_pval_matrix[i, j] = pval

# 计算Spearman相关系数
print("计算Spearman相关系数...")
spearman_corr_matrix = np.zeros((len(pathology_features), len(all_selected_radiomics)))
spearman_pval_matrix = np.zeros((len(pathology_features), len(all_selected_radiomics)))

for i, pathology_feat in enumerate(pathology_features):
    for j, radiomics_feat in enumerate(all_selected_radiomics):
        corr, pval = spearmanr(df_analysis[pathology_feat], df_analysis[radiomics_feat])
        spearman_corr_matrix[i, j] = corr
        spearman_pval_matrix[i, j] = pval

print("相关性计算完成")

# ============================================================
# 4. 保存相关性结果
# ============================================================
print("\n[步骤4] 保存相关性结果...")

# 4.1 保存Pearson相关系数矩阵
pearson_df = pd.DataFrame(
    pearson_corr_matrix,
    index=pathology_features,
    columns=all_selected_radiomics
)
pearson_df.to_csv('pathology_radiomics_pearson_correlation.csv')
print("已保存: pathology_radiomics_pearson_correlation.csv")

# 4.2 保存Pearson p值矩阵
pearson_pval_df = pd.DataFrame(
    pearson_pval_matrix,
    index=pathology_features,
    columns=all_selected_radiomics
)
pearson_pval_df.to_csv('pathology_radiomics_pearson_pvalues.csv')
print("已保存: pathology_radiomics_pearson_pvalues.csv")

# 4.3 保存Spearman相关系数矩阵
spearman_df = pd.DataFrame(
    spearman_corr_matrix,
    index=pathology_features,
    columns=all_selected_radiomics
)
spearman_df.to_csv('pathology_radiomics_spearman_correlation.csv')
print("已保存: pathology_radiomics_spearman_correlation.csv")

# 4.4 保存Spearman p值矩阵
spearman_pval_df = pd.DataFrame(
    spearman_pval_matrix,
    index=pathology_features,
    columns=all_selected_radiomics
)
spearman_pval_df.to_csv('pathology_radiomics_spearman_pvalues.csv')
print("已保存: pathology_radiomics_spearman_pvalues.csv")

# ============================================================
# 5. 找出显著相关的特征对
# ============================================================
print("\n[步骤5] 识别显著相关的特征对...")

def find_significant_correlations(corr_matrix, pval_matrix, pathology_names, 
                                   radiomics_names, threshold_corr=0.3, threshold_pval=0.05):
    """找出显著相关的特征对"""
    significant_pairs = []
    
    for i, pathology_feat in enumerate(pathology_names):
        for j, radiomics_feat in enumerate(radiomics_names):
            corr = corr_matrix[i, j]
            pval = pval_matrix[i, j]
            
            if abs(corr) >= threshold_corr and pval < threshold_pval:
                significant_pairs.append({
                    'Pathology_Feature': pathology_feat,
                    'Radiomics_Feature': radiomics_feat,
                    'Correlation': corr,
                    'P_value': pval,
                    'Abs_Correlation': abs(corr)
                })
    
    return pd.DataFrame(significant_pairs)

# Pearson显著相关对
pearson_sig = find_significant_correlations(
    pearson_corr_matrix, pearson_pval_matrix,
    pathology_features, all_selected_radiomics,
    threshold_corr=0.3, threshold_pval=0.05
)

if len(pearson_sig) > 0:
    pearson_sig = pearson_sig.sort_values('Abs_Correlation', ascending=False)
    pearson_sig.to_csv('pathology_radiomics_pearson_significant.csv', index=False)
    print(f"\nPearson显著相关对数量 (|r|≥0.3, p<0.05): {len(pearson_sig)}")
    print("已保存: pathology_radiomics_pearson_significant.csv")
    print("\nTop 10 Pearson显著相关对:")
    print(pearson_sig.head(10)[['Pathology_Feature', 'Radiomics_Feature', 'Correlation', 'P_value']])
else:
    print("\n未发现Pearson显著相关对 (|r|≥0.3, p<0.05)")

# Spearman显著相关对
spearman_sig = find_significant_correlations(
    spearman_corr_matrix, spearman_pval_matrix,
    pathology_features, all_selected_radiomics,
    threshold_corr=0.3, threshold_pval=0.05
)

if len(spearman_sig) > 0:
    spearman_sig = spearman_sig.sort_values('Abs_Correlation', ascending=False)
    spearman_sig.to_csv('pathology_radiomics_spearman_significant.csv', index=False)
    print(f"\nSpearman显著相关对数量 (|ρ|≥0.3, p<0.05): {len(spearman_sig)}")
    print("已保存: pathology_radiomics_spearman_significant.csv")
    print("\nTop 10 Spearman显著相关对:")
    print(spearman_sig.head(10)[['Pathology_Feature', 'Radiomics_Feature', 'Correlation', 'P_value']])
else:
    print("\n未发现Spearman显著相关对 (|ρ|≥0.3, p<0.05)")

# ============================================================
# 6. 可视化相关性热图（使用层次聚类排序）
# ============================================================
print("\n[步骤6] 绘制相关性热图...")

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

def hierarchical_sort_by_correlation(corr_matrix, pathology_names, radiomics_names):
    """
    使用层次聚类对特征进行排序，使得相关性强的聚集在左上角
    """
    # 检查并处理NaN值
    corr_matrix_clean = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    try:
        # 使用绝对相关系数作为相似度，转换为距离
        # 距离 = 1 - |相关系数|，这样高相关的特征距离小
        
        # 对影像组学特征进行聚类
        # 计算每对影像组学特征之间的相似度（基于它们与病理特征的相关性模式）
        n_radiomics = len(radiomics_names)
        radiomics_sim = np.zeros((n_radiomics, n_radiomics))
        for i in range(n_radiomics):
            for j in range(n_radiomics):
                # 使用相关系数的相关性作为相似度
                radiomics_sim[i, j] = np.corrcoef(corr_matrix_clean[:, i], corr_matrix_clean[:, j])[0, 1]
        
        radiomics_sim = np.nan_to_num(radiomics_sim, nan=0.0)
        radiomics_dist = 1 - np.abs(radiomics_sim)
        np.fill_diagonal(radiomics_dist, 0)
        
        radiomics_dist_condensed = squareform(radiomics_dist, checks=False)
        radiomics_linkage = linkage(radiomics_dist_condensed, method='average')
        radiomics_dendrogram = dendrogram(radiomics_linkage, no_plot=True)
        radiomics_sorted_idx = radiomics_dendrogram['leaves']
        
        # 对病理特征进行聚类
        n_pathology = len(pathology_names)
        pathology_sim = np.zeros((n_pathology, n_pathology))
        for i in range(n_pathology):
            for j in range(n_pathology):
                pathology_sim[i, j] = np.corrcoef(corr_matrix_clean[i, :], corr_matrix_clean[j, :])[0, 1]
        
        pathology_sim = np.nan_to_num(pathology_sim, nan=0.0)
        pathology_dist = 1 - np.abs(pathology_sim)
        np.fill_diagonal(pathology_dist, 0)
        
        pathology_dist_condensed = squareform(pathology_dist, checks=False)
        pathology_linkage = linkage(pathology_dist_condensed, method='average')
        pathology_dendrogram = dendrogram(pathology_linkage, no_plot=True)
        pathology_sorted_idx = pathology_dendrogram['leaves']
        
        # 进一步优化：按照平均绝对相关性对聚类结果进行微调
        # 计算每个特征的平均绝对相关性
        radiomics_mean_abs_corr = np.mean(np.abs(corr_matrix_clean), axis=0)
        pathology_mean_abs_corr = np.mean(np.abs(corr_matrix_clean), axis=1)
        
        # 对聚类后的索引按平均相关性排序（强相关在前）
        radiomics_sorted_idx = sorted(radiomics_sorted_idx, 
                                      key=lambda x: radiomics_mean_abs_corr[x], 
                                      reverse=True)
        pathology_sorted_idx = sorted(pathology_sorted_idx, 
                                      key=lambda x: pathology_mean_abs_corr[x], 
                                      reverse=True)
        
    except Exception as e:
        print(f"  警告: 层次聚类失败 ({e})，使用简单排序")
        # 如果聚类失败，使用简单的平均相关性排序
        radiomics_mean_corr = np.mean(np.abs(corr_matrix_clean), axis=0)
        radiomics_sorted_idx = np.argsort(radiomics_mean_corr)[::-1]
        
        pathology_mean_corr = np.mean(np.abs(corr_matrix_clean), axis=1)
        pathology_sorted_idx = np.argsort(pathology_mean_corr)[::-1]
    
    # 重新排列矩阵
    sorted_matrix = corr_matrix[pathology_sorted_idx, :][:, radiomics_sorted_idx]
    sorted_pathology_names = [pathology_names[i] for i in pathology_sorted_idx]
    sorted_radiomics_names = [radiomics_names[i] for i in radiomics_sorted_idx]
    
    return sorted_matrix, sorted_pathology_names, sorted_radiomics_names

def plot_correlation_heatmap(corr_matrix, pathology_names, radiomics_names, 
                             title, filename, figsize=None):
    """绘制相关性热图（已排序）"""
    # 根据特征数量自动调整图片大小
    if figsize is None:
        width = max(14, len(radiomics_names) * 0.7)
        height = max(9, len(pathology_names) * 0.8)
        figsize = (width, height)
    
    plt.figure(figsize=figsize)
    
    # 创建热图
    ax = sns.heatmap(corr_matrix, 
                     xticklabels=radiomics_names,
                     yticklabels=pathology_names,
                     cmap='RdBu_r', 
                     center=0,
                     vmin=-1, vmax=1,
                     annot=True,
                     fmt='.2f',
                     annot_kws={'size': 7},
                     linewidths=0.5,
                     linecolor='gray',
                     cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8})
    
    plt.title(title, fontsize=15, pad=20, fontweight='bold')
    plt.xlabel('影像组学特征 (层次聚类排序)', fontsize=12, fontweight='bold')
    plt.ylabel('病理特征 (层次聚类排序)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=90, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=11)
    
    # 添加网格线使图更清晰
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {filename}")

# 对Pearson相关性矩阵进行层次聚类排序并绘图
print("\n对Pearson相关性矩阵进行层次聚类排序...")
pearson_sorted_matrix, pearson_sorted_pathology, pearson_sorted_radiomics = \
    hierarchical_sort_by_correlation(pearson_corr_matrix, pathology_features, all_selected_radiomics)

plot_correlation_heatmap(
    pearson_sorted_matrix,
    pearson_sorted_pathology,
    pearson_sorted_radiomics,
    'Pearson相关性: 病理特征 vs 影像组学模型筛选特征',
    'pathology_radiomics_pearson_heatmap_sorted.png'
)

# 对Spearman相关性矩阵进行层次聚类排序并绘图
print("\n对Spearman相关性矩阵进行层次聚类排序...")
spearman_sorted_matrix, spearman_sorted_pathology, spearman_sorted_radiomics = \
    hierarchical_sort_by_correlation(spearman_corr_matrix, pathology_features, all_selected_radiomics)

plot_correlation_heatmap(
    spearman_sorted_matrix,
    spearman_sorted_pathology,
    spearman_sorted_radiomics,
    'Spearman相关性: 病理特征 vs 影像组学模型筛选特征',
    'pathology_radiomics_spearman_heatmap_sorted.png'
)

# 保存排序后的相关性矩阵
pearson_sorted_df = pd.DataFrame(
    pearson_sorted_matrix,
    index=pearson_sorted_pathology,
    columns=pearson_sorted_radiomics
)
pearson_sorted_df.to_csv('pathology_radiomics_pearson_correlation_sorted.csv')
print("\n已保存排序后的Pearson相关性矩阵: pathology_radiomics_pearson_correlation_sorted.csv")

spearman_sorted_df = pd.DataFrame(
    spearman_sorted_matrix,
    index=spearman_sorted_pathology,
    columns=spearman_sorted_radiomics
)
spearman_sorted_df.to_csv('pathology_radiomics_spearman_correlation_sorted.csv')
print("已保存排序后的Spearman相关性矩阵: pathology_radiomics_spearman_correlation_sorted.csv")

# ============================================================
# 7. 统计摘要
# ============================================================
print("\n" + "=" * 80)
print("相关性分析统计摘要")
print("=" * 80)

print(f"\n分析的病理特征数: {len(pathology_features)}")
print(f"分析的影像组学特征数: {len(all_selected_radiomics)}")
print(f"总相关性对数: {len(pathology_features) * len(all_selected_radiomics)}")

# Pearson统计
print(f"\n【Pearson相关性统计】")
print(f"  平均相关系数: {np.mean(np.abs(pearson_corr_matrix)):.4f}")
print(f"  最大相关系数: {np.max(np.abs(pearson_corr_matrix)):.4f}")
print(f"  |r| ≥ 0.3 的对数: {np.sum(np.abs(pearson_corr_matrix) >= 0.3)}")
print(f"  |r| ≥ 0.5 的对数: {np.sum(np.abs(pearson_corr_matrix) >= 0.5)}")
print(f"  |r| ≥ 0.7 的对数: {np.sum(np.abs(pearson_corr_matrix) >= 0.7)}")
print(f"  p < 0.05 的对数: {np.sum(pearson_pval_matrix < 0.05)}")
print(f"  显著相关对数 (|r|≥0.3 且 p<0.05): {len(pearson_sig) if len(pearson_sig) > 0 else 0}")

# Spearman统计
print(f"\n【Spearman相关性统计】")
print(f"  平均相关系数: {np.mean(np.abs(spearman_corr_matrix)):.4f}")
print(f"  最大相关系数: {np.max(np.abs(spearman_corr_matrix)):.4f}")
print(f"  |ρ| ≥ 0.3 的对数: {np.sum(np.abs(spearman_corr_matrix) >= 0.3)}")
print(f"  |ρ| ≥ 0.5 的对数: {np.sum(np.abs(spearman_corr_matrix) >= 0.5)}")
print(f"  |ρ| ≥ 0.7 的对数: {np.sum(np.abs(spearman_corr_matrix) >= 0.7)}")
print(f"  p < 0.05 的对数: {np.sum(spearman_pval_matrix < 0.05)}")
print(f"  显著相关对数 (|ρ|≥0.3 且 p<0.05): {len(spearman_sig) if len(spearman_sig) > 0 else 0}")

# 每个病理特征的相关性统计
print(f"\n【各病理特征的平均相关性强度】")
print(f"{'病理特征':<15} {'Pearson平均|r|':<18} {'Spearman平均|ρ|':<18}")
print("-" * 55)
for i, feat in enumerate(pathology_features):
    pearson_mean = np.mean(np.abs(pearson_corr_matrix[i, :]))
    spearman_mean = np.mean(np.abs(spearman_corr_matrix[i, :]))
    print(f"{feat:<15} {pearson_mean:<18.4f} {spearman_mean:<18.4f}")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)

print("\n【生成的文件】")
print("相关性矩阵:")
print("  - pathology_radiomics_pearson_correlation.csv (原始顺序)")
print("  - pathology_radiomics_pearson_correlation_sorted.csv (层次聚类排序)")
print("  - pathology_radiomics_pearson_pvalues.csv")
print("  - pathology_radiomics_spearman_correlation.csv (原始顺序)")
print("  - pathology_radiomics_spearman_correlation_sorted.csv (层次聚类排序)")
print("  - pathology_radiomics_spearman_pvalues.csv")
print("\n显著相关对:")
if len(pearson_sig) > 0:
    print("  - pathology_radiomics_pearson_significant.csv")
if len(spearman_sig) > 0:
    print("  - pathology_radiomics_spearman_significant.csv")
print("\n可视化:")
print("  - pathology_radiomics_pearson_heatmap_sorted.png (层次聚类排序)")
print("  - pathology_radiomics_spearman_heatmap_sorted.png (层次聚类排序)")
