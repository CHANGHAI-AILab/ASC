"""
SPPI风险预测模型 - 三种建模策略
1. 临床模型：直接使用所有临床特征，不做特征筛选
2. 影像组学模型：共线性分析 + LASSO特征选择
3. 联合模型：所有临床 + 影像组学特征，然后共线性分析 + LASSO特征选择
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置全局随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("SPPI风险预测模型构建 - 三种建模策略")
print("=" * 80)

# ============================================================
# 1. 数据读取
# ============================================================
print("\n[步骤1] 读取数据...")
df = pd.read_excel(r'D:\JMC\病理影像特征汇总merge.xlsx')

# 定义特征列
clinical_features = ['性别（男1女2）', '年龄', 'BMI', '位置.头1.体尾部2.', '肿瘤最大径CM.', 
                     '瘤内钙化', '瘤内囊变', '环形强化', '胰管扩张', '胆总管扩张', 
                     '肿块上游胰腺萎缩', '潴留囊肿.假性囊肿', '阻塞性胰腺炎', 
                     '淋巴结肿大', '血管侵犯', '栓子']

# 影像组学特征：以a_, c_, p_, v_开头的列
radiomics_features = [col for col in df.columns if col.startswith(('a_', 'c_', 'p_', 'v_'))]

# 目标变量
target = 'risk_median_numeric'

print(f"临床特征数量: {len(clinical_features)}")
print(f"影像组学特征数量: {len(radiomics_features)}")
print(f"总样本数: {len(df)}")

# 划分数据集
train_df = df[df['dataset'] == 'train'].copy()
val_df = df[df['dataset'] == 'val'].copy()
test_df = df[df['dataset'] == 'test'].copy()

y_train = train_df[target].values
y_val = val_df[target].values
y_test = test_df[target].values

print(f"训练集: {len(y_train)}, 验证集: {len(y_val)}, 测试集: {len(y_test)}")
print(f"训练集阳性率: {y_train.mean():.2%}")
print(f"验证集阳性率: {y_val.mean():.2%}")
print(f"测试集阳性率: {y_test.mean():.2%}")

# ============================================================
# 2. 数据预处理与标准化 (Z-score Normalization)
# ============================================================
print("\n" + "=" * 80)
print("[步骤2] 数据预处理与Z-score标准化")
print("=" * 80)

def zscore_normalize(X_train, X_val, X_test):
    """Z-score标准化: z = (x - μ) / σ"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# 临床特征标准化
X_train_clinical = train_df[clinical_features].values
X_val_clinical = val_df[clinical_features].values
X_test_clinical = test_df[clinical_features].values

X_train_clinical_scaled, X_val_clinical_scaled, X_test_clinical_scaled, scaler_clinical = \
    zscore_normalize(X_train_clinical, X_val_clinical, X_test_clinical)

print(f"临床特征标准化完成")
print(f"  训练集形状: {X_train_clinical_scaled.shape}")
print(f"  均值: {X_train_clinical_scaled.mean():.6f}, 标准差: {X_train_clinical_scaled.std():.6f}")

# 影像组学特征标准化
X_train_radiomics = train_df[radiomics_features].values
X_val_radiomics = val_df[radiomics_features].values
X_test_radiomics = test_df[radiomics_features].values

X_train_radiomics_scaled, X_val_radiomics_scaled, X_test_radiomics_scaled, scaler_radiomics = \
    zscore_normalize(X_train_radiomics, X_val_radiomics, X_test_radiomics)

print(f"影像组学特征标准化完成")
print(f"  训练集形状: {X_train_radiomics_scaled.shape}")
print(f"  均值: {X_train_radiomics_scaled.mean():.6f}, 标准差: {X_train_radiomics_scaled.std():.6f}")

# ============================================================
# 3. 特征筛选函数定义
# ============================================================
print("\n" + "=" * 80)
print("[步骤3] 定义特征筛选函数")
print("=" * 80)

def remove_collinear_features(X_train, feature_names, threshold=0.8):
    """
    移除高度相关的特征（Pearson相关系数 > threshold）
    保留第一个特征，移除后续相关的特征
    """
    corr_matrix = np.corrcoef(X_train.T)
    to_remove = set()
    n_features = len(feature_names)
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            if abs(corr_matrix[i, j]) > threshold:
                to_remove.add(j)
    
    to_keep = [i for i in range(n_features) if i not in to_remove]
    kept_features = [feature_names[i] for i in to_keep]
    removed_features = [feature_names[i] for i in to_remove]
    
    return to_keep, kept_features, removed_features, corr_matrix

def lasso_feature_selection(X_train, y_train, feature_names, cv=5):
    """
    使用LassoCV进行特征选择
    """
    lasso_cv = LassoCV(cv=cv, random_state=RANDOM_STATE, max_iter=10000, n_alphas=100)
    lasso_cv.fit(X_train, y_train)
    
    best_alpha = lasso_cv.alpha_
    coefficients = lasso_cv.coef_
    
    selected_idx = np.where(coefficients > 0.02)[0]
    selected_features = [feature_names[i] for i in selected_idx]
    selected_coef = coefficients[selected_idx]
    
    return selected_idx, selected_features, selected_coef, best_alpha, lasso_cv

print("特征筛选函数定义完成")

# ============================================================
# 4. 策略1: 临床模型 - 直接使用所有临床特征
# ============================================================
print("\n" + "=" * 80)
print("[策略1] 临床模型 - 使用所有临床特征，不做特征筛选")
print("=" * 80)

X_train_clinical_final = X_train_clinical_scaled
X_val_clinical_final = X_val_clinical_scaled
X_test_clinical_final = X_test_clinical_scaled
clinical_final_features = clinical_features

print(f"使用特征数: {len(clinical_final_features)}")
print(f"特征列表: {clinical_final_features}")

# 构建临床模型
model_clinical = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
model_clinical.fit(X_train_clinical_final, y_train)

y_train_pred_clinical = model_clinical.predict_proba(X_train_clinical_final)[:, 1]
y_val_pred_clinical = model_clinical.predict_proba(X_val_clinical_final)[:, 1]
y_test_pred_clinical = model_clinical.predict_proba(X_test_clinical_final)[:, 1]

auc_train_clinical = roc_auc_score(y_train, y_train_pred_clinical)
auc_val_clinical = roc_auc_score(y_val, y_val_pred_clinical)
auc_test_clinical = roc_auc_score(y_test, y_test_pred_clinical)

print(f"\n临床模型性能:")
print(f"  训练集AUC: {auc_train_clinical:.4f}")
print(f"  验证集AUC: {auc_val_clinical:.4f}")
print(f"  测试集AUC: {auc_test_clinical:.4f}")

# 保存临床特征系数
clinical_coef_df = pd.DataFrame({
    'Feature': clinical_final_features,
    'Coefficient': model_clinical.coef_[0]
})
clinical_coef_df = clinical_coef_df.sort_values('Coefficient', key=abs, ascending=False)
print(f"\n临床特征系数 (Top 10):")
print(clinical_coef_df.head(10))

# ============================================================
# 5. 策略2: 影像组学模型 - 共线性分析 + LASSO特征选择
# ============================================================
print("\n" + "=" * 80)
print("[策略2] 影像组学模型 - 共线性分析 + LASSO特征选择")
print("=" * 80)

# 5.1 共线性分析
print("\n[5.1] 共线性分析 (Pearson > 0.8)")
radiomics_keep_idx, radiomics_kept, radiomics_removed, radiomics_corr = \
    remove_collinear_features(X_train_radiomics_scaled, radiomics_features, threshold=0.5)

print(f"  原始特征数: {len(radiomics_features)}")
print(f"  移除特征数: {len(radiomics_removed)}")
print(f"  保留特征数: {len(radiomics_kept)}")

X_train_radiomics_filtered = X_train_radiomics_scaled[:, radiomics_keep_idx]
X_val_radiomics_filtered = X_val_radiomics_scaled[:, radiomics_keep_idx]
X_test_radiomics_filtered = X_test_radiomics_scaled[:, radiomics_keep_idx]

# 5.2 LASSO特征选择
print("\n[5.2] LASSO特征选择")
radiomics_lasso_idx, radiomics_lasso_features, radiomics_lasso_coef, radiomics_alpha, lasso_radiomics = \
    lasso_feature_selection(X_train_radiomics_filtered, y_train, radiomics_kept, cv=5)

print(f"  LASSO最优alpha: {radiomics_alpha:.6f}")
print(f"  LASSO选择特征数: {len(radiomics_lasso_features)}/{len(radiomics_kept)}")

X_train_radiomics_final = X_train_radiomics_filtered[:, radiomics_lasso_idx]
X_val_radiomics_final = X_val_radiomics_filtered[:, radiomics_lasso_idx]
X_test_radiomics_final = X_test_radiomics_filtered[:, radiomics_lasso_idx]

if len(radiomics_lasso_features) > 0:
    print(f"\n  选择的前10个特征及其LASSO系数:")
    for i in range(min(10, len(radiomics_lasso_features))):
        print(f"    {i+1}. {radiomics_lasso_features[i]}: {radiomics_lasso_coef[i]:.6f}")

# 构建影像组学模型
model_radiomics = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
model_radiomics.fit(X_train_radiomics_final, y_train)

y_train_pred_radiomics = model_radiomics.predict_proba(X_train_radiomics_final)[:, 1]
y_val_pred_radiomics = model_radiomics.predict_proba(X_val_radiomics_final)[:, 1]
y_test_pred_radiomics = model_radiomics.predict_proba(X_test_radiomics_final)[:, 1]

auc_train_radiomics = roc_auc_score(y_train, y_train_pred_radiomics)
auc_val_radiomics = roc_auc_score(y_val, y_val_pred_radiomics)
auc_test_radiomics = roc_auc_score(y_test, y_test_pred_radiomics)

print(f"\n影像组学模型性能:")
print(f"  训练集AUC: {auc_train_radiomics:.4f}")
print(f"  验证集AUC: {auc_val_radiomics:.4f}")
print(f"  测试集AUC: {auc_test_radiomics:.4f}")

# ============================================================
# 6. 策略3: 联合模型 - 所有特征合并后共线性分析 + LASSO
# ============================================================
print("\n" + "=" * 80)
print("[策略3] 联合模型 - 所有临床+影像组学特征，然后共线性分析 + LASSO")
print("=" * 80)

# 6.1 合并所有特征
print("\n[6.1] 合并所有特征")
combined_features = clinical_features + radiomics_features
X_train_combined_raw = np.hstack([X_train_clinical_scaled, X_train_radiomics_scaled])
X_val_combined_raw = np.hstack([X_val_clinical_scaled, X_val_radiomics_scaled])
X_test_combined_raw = np.hstack([X_test_clinical_scaled, X_test_radiomics_scaled])

print(f"  临床特征: {len(clinical_features)}")
print(f"  影像组学特征: {len(radiomics_features)}")
print(f"  合并后总特征数: {len(combined_features)}")

# 6.2 共线性分析
print("\n[6.2] 共线性分析 (Pearson > 0.8)")
combined_keep_idx, combined_kept_features, combined_removed_features, combined_corr = \
    remove_collinear_features(X_train_combined_raw, combined_features, threshold=0.5)

print(f"  原始特征数: {len(combined_features)}")
print(f"  移除特征数: {len(combined_removed_features)}")
print(f"  保留特征数: {len(combined_kept_features)}")

X_train_combined_filtered = X_train_combined_raw[:, combined_keep_idx]
X_val_combined_filtered = X_val_combined_raw[:, combined_keep_idx]
X_test_combined_filtered = X_test_combined_raw[:, combined_keep_idx]

# 6.3 LASSO特征选择
print("\n[6.3] LASSO特征选择")
combined_lasso_idx, combined_lasso_features, combined_lasso_coef, combined_alpha, lasso_combined = \
    lasso_feature_selection(X_train_combined_filtered, y_train, combined_kept_features, cv=5)

print(f"  LASSO最优alpha: {combined_alpha:.6f}")
print(f"  LASSO选择特征数: {len(combined_lasso_features)}/{len(combined_kept_features)}")

X_train_combined_final = X_train_combined_filtered[:, combined_lasso_idx]
X_val_combined_final = X_val_combined_filtered[:, combined_lasso_idx]
X_test_combined_final = X_test_combined_filtered[:, combined_lasso_idx]

if len(combined_lasso_features) > 0:
    print(f"\n  选择的前15个特征及其LASSO系数:")
    for i in range(min(15, len(combined_lasso_features))):
        print(f"    {i+1}. {combined_lasso_features[i]}: {combined_lasso_coef[i]:.6f}")

# 构建联合模型
model_combined = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
model_combined.fit(X_train_combined_final, y_train)

y_train_pred_combined = model_combined.predict_proba(X_train_combined_final)[:, 1]
y_val_pred_combined = model_combined.predict_proba(X_val_combined_final)[:, 1]
y_test_pred_combined = model_combined.predict_proba(X_test_combined_final)[:, 1]

auc_train_combined = roc_auc_score(y_train, y_train_pred_combined)
auc_val_combined = roc_auc_score(y_val, y_val_pred_combined)
auc_test_combined = roc_auc_score(y_test, y_test_pred_combined)

print(f"\n联合模型性能:")
print(f"  训练集AUC: {auc_train_combined:.4f}")
print(f"  验证集AUC: {auc_val_combined:.4f}")
print(f"  测试集AUC: {auc_test_combined:.4f}")

# ============================================================
# 7. 保存初步结果（模型和预测）
# ============================================================
print("\n" + "=" * 80)
print("[步骤7] 保存模型和预测结果")
print("=" * 80)

# 7.1 保存模型
joblib.dump(model_clinical, 'model_clinical_final.pkl')
joblib.dump(model_radiomics, 'model_radiomics_final.pkl')
joblib.dump(model_combined, 'model_combined_final.pkl')
joblib.dump(scaler_clinical, 'scaler_clinical_final.pkl')
joblib.dump(scaler_radiomics, 'scaler_radiomics_final.pkl')
print("已保存模型文件")

# 7.2 保存预测结果
predictions = pd.DataFrame({
    'patient_id': df['patient_id'],
    'dataset': df['dataset'],
    'true_label': df[target],
    'pred_clinical': np.concatenate([y_train_pred_clinical, y_val_pred_clinical, y_test_pred_clinical]),
    'pred_radiomics': np.concatenate([y_train_pred_radiomics, y_val_pred_radiomics, y_test_pred_radiomics]),
    'pred_combined': np.concatenate([y_train_pred_combined, y_val_pred_combined, y_test_pred_combined])
})
predictions.to_csv('model_predictions_final.csv', index=False)
print("已保存预测结果: model_predictions_final.csv")

# ============================================================
# 8. 计算最优阈值和性能指标
# ============================================================
print("\n" + "=" * 80)
print("[步骤8] 计算最优阈值和性能指标")
print("=" * 80)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def find_optimal_cutoff(y_true, y_pred_proba):
    """
    基于Youden指数找到最优阈值
    Youden指数 = Sensitivity + Specificity - 1
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def calculate_metrics(y_true, y_pred_proba, threshold):
    """计算各种性能指标"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)  # TPR
    specificity = tn / (tn + fp)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    return {
        'AUC': auc,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1-Score': f1,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }

# 8.1 临床模型 - 基于验证集找最优阈值
print("\n[8.1] 临床模型性能评估")
cutoff_clinical = find_optimal_cutoff(y_val, y_val_pred_clinical)
print(f"  验证集最优阈值: {cutoff_clinical:.4f}")

metrics_clinical_train = calculate_metrics(y_train, y_train_pred_clinical, cutoff_clinical)
metrics_clinical_val = calculate_metrics(y_val, y_val_pred_clinical, cutoff_clinical)
metrics_clinical_test = calculate_metrics(y_test, y_test_pred_clinical, cutoff_clinical)

print(f"\n  训练集 - AUC: {metrics_clinical_train['AUC']:.4f}, Acc: {metrics_clinical_train['Accuracy']:.4f}, "
      f"Sen: {metrics_clinical_train['Sensitivity']:.4f}, Spe: {metrics_clinical_train['Specificity']:.4f}")
print(f"  验证集 - AUC: {metrics_clinical_val['AUC']:.4f}, Acc: {metrics_clinical_val['Accuracy']:.4f}, "
      f"Sen: {metrics_clinical_val['Sensitivity']:.4f}, Spe: {metrics_clinical_val['Specificity']:.4f}")
print(f"  测试集 - AUC: {metrics_clinical_test['AUC']:.4f}, Acc: {metrics_clinical_test['Accuracy']:.4f}, "
      f"Sen: {metrics_clinical_test['Sensitivity']:.4f}, Spe: {metrics_clinical_test['Specificity']:.4f}")

# 8.2 影像组学模型 - 基于验证集找最优阈值
print("\n[8.2] 影像组学模型性能评估")
cutoff_radiomics = find_optimal_cutoff(y_val, y_val_pred_radiomics)
print(f"  验证集最优阈值: {cutoff_radiomics:.4f}")

metrics_radiomics_train = calculate_metrics(y_train, y_train_pred_radiomics, cutoff_radiomics)
metrics_radiomics_val = calculate_metrics(y_val, y_val_pred_radiomics, cutoff_radiomics)
metrics_radiomics_test = calculate_metrics(y_test, y_test_pred_radiomics, cutoff_radiomics)

print(f"\n  训练集 - AUC: {metrics_radiomics_train['AUC']:.4f}, Acc: {metrics_radiomics_train['Accuracy']:.4f}, "
      f"Sen: {metrics_radiomics_train['Sensitivity']:.4f}, Spe: {metrics_radiomics_train['Specificity']:.4f}")
print(f"  验证集 - AUC: {metrics_radiomics_val['AUC']:.4f}, Acc: {metrics_radiomics_val['Accuracy']:.4f}, "
      f"Sen: {metrics_radiomics_val['Sensitivity']:.4f}, Spe: {metrics_radiomics_val['Specificity']:.4f}")
print(f"  测试集 - AUC: {metrics_radiomics_test['AUC']:.4f}, Acc: {metrics_radiomics_test['Accuracy']:.4f}, "
      f"Sen: {metrics_radiomics_test['Sensitivity']:.4f}, Spe: {metrics_radiomics_test['Specificity']:.4f}")

# 8.3 联合模型 - 基于验证集找最优阈值
print("\n[8.3] 联合模型性能评估")
cutoff_combined = find_optimal_cutoff(y_val, y_val_pred_combined)
print(f"  验证集最优阈值: {cutoff_combined:.4f}")

metrics_combined_train = calculate_metrics(y_train, y_train_pred_combined, cutoff_combined)
metrics_combined_val = calculate_metrics(y_val, y_val_pred_combined, cutoff_combined)
metrics_combined_test = calculate_metrics(y_test, y_test_pred_combined, cutoff_combined)

print(f"\n  训练集 - AUC: {metrics_combined_train['AUC']:.4f}, Acc: {metrics_combined_train['Accuracy']:.4f}, "
      f"Sen: {metrics_combined_train['Sensitivity']:.4f}, Spe: {metrics_combined_train['Specificity']:.4f}")
print(f"  验证集 - AUC: {metrics_combined_val['AUC']:.4f}, Acc: {metrics_combined_val['Accuracy']:.4f}, "
      f"Sen: {metrics_combined_val['Sensitivity']:.4f}, Spe: {metrics_combined_val['Specificity']:.4f}")
print(f"  测试集 - AUC: {metrics_combined_test['AUC']:.4f}, Acc: {metrics_combined_test['Accuracy']:.4f}, "
      f"Sen: {metrics_combined_test['Sensitivity']:.4f}, Spe: {metrics_combined_test['Specificity']:.4f}")

# ============================================================
# 8.4 保存性能指标和特征选择摘要
# ============================================================
print("\n[8.4] 保存性能指标和特征选择摘要")

# 保存性能指标
performance_metrics = pd.DataFrame({
    'Model': ['Clinical', 'Clinical', 'Clinical',
              'Radiomics', 'Radiomics', 'Radiomics',
              'Combined', 'Combined', 'Combined'],
    'Dataset': ['Train', 'Val', 'Test'] * 3,
    'Cutoff': [cutoff_clinical] * 3 + [cutoff_radiomics] * 3 + [cutoff_combined] * 3,
    'AUC': [metrics_clinical_train['AUC'], metrics_clinical_val['AUC'], metrics_clinical_test['AUC'],
            metrics_radiomics_train['AUC'], metrics_radiomics_val['AUC'], metrics_radiomics_test['AUC'],
            metrics_combined_train['AUC'], metrics_combined_val['AUC'], metrics_combined_test['AUC']],
    'Accuracy': [metrics_clinical_train['Accuracy'], metrics_clinical_val['Accuracy'], metrics_clinical_test['Accuracy'],
                 metrics_radiomics_train['Accuracy'], metrics_radiomics_val['Accuracy'], metrics_radiomics_test['Accuracy'],
                 metrics_combined_train['Accuracy'], metrics_combined_val['Accuracy'], metrics_combined_test['Accuracy']],
    'Sensitivity': [metrics_clinical_train['Sensitivity'], metrics_clinical_val['Sensitivity'], metrics_clinical_test['Sensitivity'],
                    metrics_radiomics_train['Sensitivity'], metrics_radiomics_val['Sensitivity'], metrics_radiomics_test['Sensitivity'],
                    metrics_combined_train['Sensitivity'], metrics_combined_val['Sensitivity'], metrics_combined_test['Sensitivity']],
    'Specificity': [metrics_clinical_train['Specificity'], metrics_clinical_val['Specificity'], metrics_clinical_test['Specificity'],
                    metrics_radiomics_train['Specificity'], metrics_radiomics_val['Specificity'], metrics_radiomics_test['Specificity'],
                    metrics_combined_train['Specificity'], metrics_combined_val['Specificity'], metrics_combined_test['Specificity']],
    'Precision': [metrics_clinical_train['Precision'], metrics_clinical_val['Precision'], metrics_clinical_test['Precision'],
                  metrics_radiomics_train['Precision'], metrics_radiomics_val['Precision'], metrics_radiomics_test['Precision'],
                  metrics_combined_train['Precision'], metrics_combined_val['Precision'], metrics_combined_test['Precision']],
    'F1-Score': [metrics_clinical_train['F1-Score'], metrics_clinical_val['F1-Score'], metrics_clinical_test['F1-Score'],
                 metrics_radiomics_train['F1-Score'], metrics_radiomics_val['F1-Score'], metrics_radiomics_test['F1-Score'],
                 metrics_combined_train['F1-Score'], metrics_combined_val['F1-Score'], metrics_combined_test['F1-Score']]
})
performance_metrics.to_csv('model_performance_metrics_final.csv', index=False)
print("  已保存性能指标: model_performance_metrics_final.csv")

# 保存特征选择摘要
feature_selection_summary = pd.DataFrame({
    'Model': ['Clinical', 'Clinical', 
              'Radiomics', 'Radiomics', 'Radiomics',
              'Combined', 'Combined', 'Combined'],
    'Step': ['Original', 'Final',
             'Original', 'After Collinearity', 'After LASSO',
             'Original', 'After Collinearity', 'After LASSO'],
    'N_Features': [
        len(clinical_features), len(clinical_final_features),
        len(radiomics_features), len(radiomics_kept), len(radiomics_lasso_features),
        len(combined_features), len(combined_kept_features), len(combined_lasso_features)
    ]
})
feature_selection_summary.to_csv('feature_selection_summary_final.csv', index=False)
print("  已保存特征选择摘要: feature_selection_summary_final.csv")

# 保存选中的特征及其系数
# 临床模型特征
pd.DataFrame({
    'Feature': clinical_final_features,
    'LR_Coefficient': model_clinical.coef_[0]
}).to_csv('selected_features_clinical_final.csv', index=False)

# 影像组学模型特征
pd.DataFrame({
    'Feature': radiomics_lasso_features,
    'LASSO_Coefficient': radiomics_lasso_coef,
    'LR_Coefficient': model_radiomics.coef_[0]
}).to_csv('selected_features_radiomics_final.csv', index=False)

# 联合模型特征
pd.DataFrame({
    'Feature': combined_lasso_features,
    'LASSO_Coefficient': combined_lasso_coef,
    'LR_Coefficient': model_combined.coef_[0]
}).to_csv('selected_features_combined_final.csv', index=False)

print("  已保存选中的特征及系数文件")

# ============================================================
# 9. 绘制ROC曲线
# ============================================================
print("\n" + "=" * 80)
print("[步骤9] 绘制ROC曲线")
print("=" * 80)

def plot_roc_curves(y_true, y_pred_dict, dataset_name, save_path):
    """绘制多个模型的ROC曲线"""
    plt.figure(figsize=(10, 8))
    
    for model_name, y_pred in y_pred_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {dataset_name}', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")

# 绘制训练集ROC曲线
plot_roc_curves(
    y_train,
    {
        'Clinical Model': y_train_pred_clinical,
        'Radiomics Model': y_train_pred_radiomics,
        'Combined Model': y_train_pred_combined
    },
    'Training Set',
    'roc_curves_train_final.png'
)

# 绘制验证集ROC曲线
plot_roc_curves(
    y_val,
    {
        'Clinical Model': y_val_pred_clinical,
        'Radiomics Model': y_val_pred_radiomics,
        'Combined Model': y_val_pred_combined
    },
    'Validation Set',
    'roc_curves_val_final.png'
)

# 绘制测试集ROC曲线
plot_roc_curves(
    y_test,
    {
        'Clinical Model': y_test_pred_clinical,
        'Radiomics Model': y_test_pred_radiomics,
        'Combined Model': y_test_pred_combined
    },
    'Test Set',
    'roc_curves_test_final.png'
)

# ============================================================
# 10. 最终结果总结
# ============================================================
print("\n" + "=" * 80)
print("模型构建完成！")
print("=" * 80)

print("\n" + "=" * 80)
print("最终结果总结")
print("=" * 80)

print("\n【模型性能对比 - AUC】")
print(f"{'模型':<15} {'训练集AUC':<12} {'验证集AUC':<12} {'测试集AUC':<12}")
print("-" * 55)
print(f"{'临床模型':<15} {metrics_clinical_train['AUC']:<12.4f} {metrics_clinical_val['AUC']:<12.4f} {metrics_clinical_test['AUC']:<12.4f}")
print(f"{'影像组学模型':<15} {metrics_radiomics_train['AUC']:<12.4f} {metrics_radiomics_val['AUC']:<12.4f} {metrics_radiomics_test['AUC']:<12.4f}")
print(f"{'联合模型':<15} {metrics_combined_train['AUC']:<12.4f} {metrics_combined_val['AUC']:<12.4f} {metrics_combined_test['AUC']:<12.4f}")

print("\n【最优阈值（基于验证集Youden指数）】")
print(f"临床模型阈值: {cutoff_clinical:.4f}")
print(f"影像组学模型阈值: {cutoff_radiomics:.4f}")
print(f"联合模型阈值: {cutoff_combined:.4f}")

print("\n【测试集详细性能指标】")
print(f"{'指标':<15} {'临床模型':<12} {'影像组学模型':<15} {'联合模型':<12}")
print("-" * 60)
print(f"{'AUC':<15} {metrics_clinical_test['AUC']:<12.4f} {metrics_radiomics_test['AUC']:<15.4f} {metrics_combined_test['AUC']:<12.4f}")
print(f"{'Accuracy':<15} {metrics_clinical_test['Accuracy']:<12.4f} {metrics_radiomics_test['Accuracy']:<15.4f} {metrics_combined_test['Accuracy']:<12.4f}")
print(f"{'Sensitivity':<15} {metrics_clinical_test['Sensitivity']:<12.4f} {metrics_radiomics_test['Sensitivity']:<15.4f} {metrics_combined_test['Sensitivity']:<12.4f}")
print(f"{'Specificity':<15} {metrics_clinical_test['Specificity']:<12.4f} {metrics_radiomics_test['Specificity']:<15.4f} {metrics_combined_test['Specificity']:<12.4f}")
print(f"{'Precision':<15} {metrics_clinical_test['Precision']:<12.4f} {metrics_radiomics_test['Precision']:<15.4f} {metrics_combined_test['Precision']:<12.4f}")
print(f"{'F1-Score':<15} {metrics_clinical_test['F1-Score']:<12.4f} {metrics_radiomics_test['F1-Score']:<15.4f} {metrics_combined_test['F1-Score']:<12.4f}")

print("\n【特征选择总结】")
print(f"{'模型':<15} {'原始特征':<12} {'共线性后':<12} {'LASSO后':<12}")
print("-" * 55)
print(f"{'临床模型':<15} {len(clinical_features):<12} {'-':<12} {len(clinical_final_features):<12}")
print(f"{'影像组学模型':<15} {len(radiomics_features):<12} {len(radiomics_kept):<12} {len(radiomics_lasso_features):<12}")
print(f"{'联合模型':<15} {len(combined_features):<12} {len(combined_kept_features):<12} {len(combined_lasso_features):<12}")

print("\n【建模策略说明】")
print("1. 临床模型：直接使用所有16个临床特征，不做特征筛选")
print("2. 影像组学模型：先进行共线性分析(Pearson>0.8)，再用LASSO特征选择")
print("3. 联合模型：合并所有临床+影像组学特征后，先共线性分析，再LASSO选择")
print("4. 阈值选择：基于验证集Youden指数（Sensitivity + Specificity - 1）最大化")

print("\n【保存的文件】")
print("模型文件:")
print("  - model_clinical_final.pkl")
print("  - model_radiomics_final.pkl")
print("  - model_combined_final.pkl")
print("  - scaler_clinical_final.pkl")
print("  - scaler_radiomics_final.pkl")
print("\n结果文件:")
print("  - model_predictions_final.csv")
print("  - model_performance_metrics_final.csv")
print("  - feature_selection_summary_final.csv")
print("  - selected_features_clinical_final.csv")
print("  - selected_features_radiomics_final.csv")
print("  - selected_features_combined_final.csv")
print("\n图表文件:")
print("  - roc_curves_train_final.png")
print("  - roc_curves_val_final.png")
print("  - roc_curves_test_final.png")

print("\n" + "=" * 80)
print("所有任务完成！")
print("=" * 80)
