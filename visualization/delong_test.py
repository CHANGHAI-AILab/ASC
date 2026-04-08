"""
DeLong检验：比较联合模型 vs 临床模型 和 联合模型 vs 影像组学模型
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def compute_midrank(x):
    """计算中位秩"""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    """快速DeLong算法"""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def calc_pvalue(aucs, sigma):
    """计算p值"""
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + stats.norm.logsf(z, loc=0, scale=1) / np.log(10)

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    DeLong检验
    返回: p值
    """
    order = np.argsort(ground_truth)
    label_1_count = int(ground_truth.sum())
    
    predictions_sorted_transposed = np.vstack([predictions_one, predictions_two])[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    
    return 10**calc_pvalue(aucs, delongcov)[0][0]

# 读取预测结果
print("=" * 60)
print("DeLong检验：比较模型性能")
print("=" * 60)

predictions = pd.read_csv('model_predictions.csv')

# 分离数据集
datasets = {
    'Train': predictions[predictions['dataset'] == 'train'],
    'Validation': predictions[predictions['dataset'] == 'val'],
    'Test': predictions[predictions['dataset'] == 'test']
}

# 存储结果
delong_results = []

# 对每个数据集进行DeLong检验
for dataset_name, data in datasets.items():
    print(f"\n{dataset_name} Set:")
    print("-" * 60)
    
    y_true = data['true_label'].values
    pred_clinical = data['pred_clinical'].values
    pred_radiomics = data['pred_radiomics'].values
    pred_combined = data['pred_combined'].values
    
    # 1. 联合模型 vs 临床模型
    try:
        p_value_1 = delong_roc_test(y_true, pred_combined, pred_clinical)
        print(f"Combined vs Clinical: p-value = {p_value_1:.4f}")
        if p_value_1 < 0.001:
            significance_1 = "***"
        elif p_value_1 < 0.01:
            significance_1 = "**"
        elif p_value_1 < 0.05:
            significance_1 = "*"
        else:
            significance_1 = "ns"
    except Exception as e:
        print(f"Combined vs Clinical: Error - {e}")
        p_value_1 = np.nan
        significance_1 = "Error"
    
    # 2. 联合模型 vs 影像组学模型
    try:
        p_value_2 = delong_roc_test(y_true, pred_combined, pred_radiomics)
        print(f"Combined vs Radiomics: p-value = {p_value_2:.4f}")
        if p_value_2 < 0.001:
            significance_2 = "***"
        elif p_value_2 < 0.01:
            significance_2 = "**"
        elif p_value_2 < 0.05:
            significance_2 = "*"
        else:
            significance_2 = "ns"
    except Exception as e:
        print(f"Combined vs Radiomics: Error - {e}")
        p_value_2 = np.nan
        significance_2 = "Error"
    
    # 3. 临床模型 vs 影像组学模型
    try:
        p_value_3 = delong_roc_test(y_true, pred_radiomics, pred_clinical)
        print(f"Radiomics vs Clinical: p-value = {p_value_3:.4f}")
        if p_value_3 < 0.001:
            significance_3 = "***"
        elif p_value_3 < 0.01:
            significance_3 = "**"
        elif p_value_3 < 0.05:
            significance_3 = "*"
        else:
            significance_3 = "ns"
    except Exception as e:
        print(f"Radiomics vs Clinical: Error - {e}")
        p_value_3 = np.nan
        significance_3 = "Error"
    
    delong_results.append({
        'Dataset': dataset_name,
        'Comparison': 'Combined vs Clinical',
        'P-value': p_value_1,
        'Significance': significance_1
    })
    
    delong_results.append({
        'Dataset': dataset_name,
        'Comparison': 'Combined vs Radiomics',
        'P-value': p_value_2,
        'Significance': significance_2
    })
    
    delong_results.append({
        'Dataset': dataset_name,
        'Comparison': 'Radiomics vs Clinical',
        'P-value': p_value_3,
        'Significance': significance_3
    })

# 保存结果
delong_df = pd.DataFrame(delong_results)
delong_df.to_csv('delong_test_results.csv', index=False)

print("\n" + "=" * 60)
print("DeLong检验结果:")
print(delong_df.to_string(index=False))
print("\n已保存: delong_test_results.csv")
print("\n注: *** p<0.001, ** p<0.01, * p<0.05, ns: not significant")
print("=" * 60)
