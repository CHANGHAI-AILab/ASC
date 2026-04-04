"""
绘制ROC曲线和计算95%置信区间
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

def calculate_auc_ci(y_true, y_pred, n_bootstraps=2000, ci=95):
    """计算AUC的95%置信区间"""
    from sklearn.metrics import roc_auc_score
    from sklearn.utils import resample
    
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    
    for i in range(n_bootstraps):
        indices = resample(np.arange(len(y_true)), random_state=rng)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    alpha = (100 - ci) / 2
    lower = np.percentile(sorted_scores, alpha)
    upper = np.percentile(sorted_scores, 100 - alpha)
    
    return lower, upper

# 读取预测结果
print("=" * 60)
print("读取预测结果...")
predictions = pd.read_csv('model_predictions.csv')

# 分离数据集
train_data = predictions[predictions['dataset'] == 'train']
val_data = predictions[predictions['dataset'] == 'val']
test_data = predictions[predictions['dataset'] == 'test']

# 准备数据
datasets = {
    'Train': train_data,
    'Validation': val_data,
    'Test': test_data
}

models = {
    'Clinical': 'pred_clinical',
    'Radiomics': 'pred_radiomics',
    'Combined': 'pred_combined'
}

colors = {
    'Clinical': '#1f77b4',
    'Radiomics': '#ff7f0e',
    'Combined': '#2ca02c'
}

# ============================================================
# 绘制ROC曲线（三个数据集，每个一张图）
# ============================================================
print("\n绘制ROC曲线...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (dataset_name, data) in enumerate(datasets.items()):
    ax = axes[idx]
    
    y_true = data['true_label'].values
    
    for model_name, pred_col in models.items():
        y_pred = data[pred_col].values
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # 计算95%CI
        lower, upper = calculate_auc_ci(y_true, y_pred)
        
        # 绘制ROC曲线
        ax.plot(fpr, tpr, color=colors[model_name], lw=2,
                label=f'{model_name} (AUC={roc_auc:.3f}, 95%CI: {lower:.3f}-{upper:.3f})')
    
    # 绘制对角线
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{dataset_name} Set', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves_all_datasets.png', dpi=300, bbox_inches='tight')
plt.savefig('roc_curves_all_datasets.pdf', bbox_inches='tight')
print("已保存: roc_curves_all_datasets.png/pdf")
plt.close()

# ============================================================
# 绘制单独的ROC曲线（每个模型一张图，包含三个数据集）
# ============================================================
print("\n绘制每个模型的ROC曲线...")

dataset_colors = {
    'Train': '#1f77b4',
    'Validation': '#ff7f0e',
    'Test': '#2ca02c'
}

for model_name, pred_col in models.items():
    fig, ax = plt.subplots(figsize=(8, 7))
    
    for dataset_name, data in datasets.items():
        y_true = data['true_label'].values
        y_pred = data[pred_col].values
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # 计算95%CI
        lower, upper = calculate_auc_ci(y_true, y_pred)
        
        # 绘制ROC曲线
        ax.plot(fpr, tpr, color=dataset_colors[dataset_name], lw=2.5,
                label=f'{dataset_name} (AUC={roc_auc:.3f}, 95%CI: {lower:.3f}-{upper:.3f})')
    
    # 绘制对角线
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(f'{model_name} Model - ROC Curves', fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'roc_curve_{model_name.lower()}_model.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'roc_curve_{model_name.lower()}_model.pdf', bbox_inches='tight')
    print(f"已保存: roc_curve_{model_name.lower()}_model.png/pdf")
    plt.close()

# ============================================================
# 创建AUC结果表格（包含95%CI）
# ============================================================
print("\n计算AUC和95%CI...")

auc_table = []

for dataset_name, data in datasets.items():
    y_true = data['true_label'].values
    
    for model_name, pred_col in models.items():
        y_pred = data[pred_col].values
        
        # 计算AUC
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_true, y_pred)
        
        # 计算95%CI
        lower, upper = calculate_auc_ci(y_true, y_pred)
        
        auc_table.append({
            'Model': model_name,
            'Dataset': dataset_name,
            'AUC': roc_auc,
            '95% CI Lower': lower,
            '95% CI Upper': upper,
            'AUC (95% CI)': f'{roc_auc:.3f} ({lower:.3f}-{upper:.3f})'
        })

auc_df = pd.DataFrame(auc_table)
auc_df.to_csv('auc_with_ci.csv', index=False)
print("\n已保存: auc_with_ci.csv")
print("\nAUC Results with 95% CI:")
print(auc_df.to_string(index=False))

print("\n" + "=" * 60)
print("ROC曲线绘制完成！")
