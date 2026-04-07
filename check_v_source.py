import pandas as pd

# 查CSV列名
df_csv = pd.read_csv('D:/LF/jmc_vis/4_complete_sample_analysis.csv', encoding='gbk')
print('CSV columns:', df_csv.columns.tolist())
print()

# 找包含tumor/tissue/stroma的列
cols = [c for c in df_csv.columns if any(k in c.lower() for k in ['tumor','tissue','stroma','同病理'])]
print('Relevant cols:', cols)
print()

# 看两个样本的值
for pid_prefix in ['2202521A6', '2314986A5']:
    rows = df_csv[df_csv.iloc[:,0].astype(str).str.startswith(pid_prefix)]
    if not rows.empty:
        print(f'{pid_prefix}:')
        for c in cols:
            print(f'  {c}: {rows[c].values}')
        print()

# 也看Excel里的V_raw对应的原始值
df_xl = pd.read_excel('D:/LF/jmc_vis/病理特征汇总.xlsx')
df_xl['patient_id'] = df_xl['patient_id'].astype(str)
import numpy as np
for pid in ['2202521A6', '2314986A5']:
    v = float(df_xl[df_xl['patient_id']==pid]['V_raw'].values[0])
    # 反算 logit^-1
    orig = 1 / (1 + np.exp(-v))
    print(f'{pid}: V_raw={v:.4f} -> Tumor_tissue_wavg={orig:.4f} ({int(round(orig*100))}%)')
