"""
SPPI风险预测模型分析
包括：临床影像模型、影像组学模型、联合模型
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
print("=" * 60)
print("读取数据...")
df_image = pd.read_excel(r'D:\JMC\病理影像特征汇总.xlsx')
df_pathology = pd.read_excel(r'D:\JMC\病理特征汇总.xlsx')

# 合并数据
df = pd.merge(df_image, df_pathology[['patient_id', 'SPPI_ridge_B5', 'SPPI_z']], 
              on='patient_id', how='inner')

print(f"合并后数据形状: {df.shape}")
print(f"训练集: {sum(df['dataset']=='train')}")
print(f"验证集: {sum(df['dataset']=='val')}")
print(f"测试集: {sum(df['dataset']=='test')}")

# 2. 创建SPPI风险分组标签（基于中位数）
sppi_median = df['SPPI_z'].median()
df['SPPI_risk'] = (df['SPPI_z'] > sppi_median).astype(int)  # 1=高风险, 0=低风险
print(f"\nSPPI_z中位数: {sppi_median:.4f}")
print(f"高风险样本数: {sum(df['SPPI_risk']==1)}")
print(f"低风险样本数: {sum(df['SPPI_risk']==0)}")

# 3. 定义特征列
# 临床影像特征（前4列之后到影像特征之前）
clinical_features = ['性别（男1女2）', '年龄', 'BMI', '位置.头1.体尾部2.', '肿瘤最大径CM.', 
                     '瘤内钙化', '瘤内囊变', '环形强化', '胰管扩张', '胆总管扩张', 
                     '肿块上游胰腺萎缩', '潴留囊肿.假性囊肿', '阻塞性胰腺炎', 
                     '淋巴结肿大', '血管侵犯', '栓子']

# 影像组学特征（所有以a_、c_、p_、v_开头的列）
radiomics_features = [col for col in df.columns if col.startswith(('a_', 'c_', 'p_', 'v_'))]

print(f"\n临床特征数量: {len(clinical_features)}")
print(f"影像组学特征数量: {len(radiomics_features)}")

# 4. 数据预处理 - 处理缺失值
print("\n" + "=" * 60)
print("数据预处理...")
for col in clinical_features + radiomics_features:
    if col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

# 5. 划分数据集
train_df = df[df['dataset'] == 'train'].copy()
val_df = df[df['dataset'] == 'val'].copy()
test_df = df[df['dataset'] == 'test'].copy()

print(f"训练集: {len(train_df)}, 高风险: {sum(train_df['SPPI_risk']==1)}")
print(f"验证集: {len(val_df)}, 高风险: {sum(val_df['SPPI_risk']==1)}")
print(f"测试集: {len(test_df)}, 高风险: {sum(test_df['SPPI_risk']==1)}")

# 保存处理后的数据
df.to_csv(r'D:\JMC\processed_data.csv', index=False)
print("\n处理后的数据已保存到 processed_data.csv")
print("=" * 60)
