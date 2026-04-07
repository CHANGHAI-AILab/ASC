"""
分析CSV原始数据和Excel特征的对应关系
找出wavg的计算逻辑
"""
import pandas as pd, numpy as np

df_csv = pd.read_csv('D:/LF/jmc_vis/4_complete_sample_analysis.csv', encoding='gbk')
df_xl  = pd.read_excel('D:/LF/jmc_vis/病理特征汇总.xlsx')
df_xl['patient_id'] = df_xl['patient_id'].astype(str)

EPS = 1e-6
def logit(p): return np.log(np.clip(p,EPS,1-EPS)/(1-np.clip(p,EPS,1-EPS)))

print('CSV columns:')
for c in df_csv.columns: print(f'  {c}')
print()

# 找两个患者在CSV里的所有切片行
for pid in ['2202521A6','2314986A5']:
    # CSV样本名是 pid_predictions，patient_id是前缀
    # 先找病理号
    xl_row = df_xl[df_xl['patient_id']==pid].iloc[0]
    print(f'=== {pid} ===')
    print(f'  Excel features:')
    for col in ['C_raw','D1_raw','D3_raw','D4_raw','D5_raw','B_raw','S1_raw','S3_raw','V_raw']:
        print(f'    {col}: {xl_row[col]:.4f}')

    # 找CSV中该患者的所有切片
    mask = df_csv.iloc[:,0].astype(str).str.startswith(pid)
    rows = df_csv[mask]
    print(f'  CSV rows for {pid}: {len(rows)}')
    if len(rows) > 0:
        print(f'  Sample names: {rows.iloc[:,0].tolist()}')
        # 打印关键列
        key_cols = ['腺癌比例','鳞癌比例','SCC最大连通域占比(DCR)','SCC形状因子',
                    'SCC连通域数量','SCC显著病灶数量','SCC第二大连通域占比',
                    'SCC与ADC接触边界比例','SCC靠近浸润前沿比例','SCC到前沿平均距离(um)',
                    '同病理号平均鳞腺比例']
        for c in key_cols:
            if c in rows.columns:
                vals = rows[c].values
                print(f'    {c}: {vals}')

        # 尝试从CSV直接算C_raw
        scc_col = rows['鳞癌比例'].values
        adc_col = rows['腺癌比例'].values
        tumor = scc_col + adc_col
        scc_in_tumor = np.where(tumor>0, scc_col/tumor, 0)
        c_raw_calc = logit(scc_in_tumor)
        print(f'  Calc C_raw per slice: {c_raw_calc}')
        print(f'  Excel C_raw: {xl_row["C_raw"]:.4f}')

        # 尝试加权平均（按SCC面积加权）
        scc_area = rows['SCC总面积'].values if 'SCC总面积' in rows.columns else np.ones(len(rows))
        w = scc_area / scc_area.sum() if scc_area.sum()>0 else np.ones(len(rows))/len(rows)
        c_wavg = float(np.sum(w * c_raw_calc))
        print(f'  Weighted avg C_raw (by SCC area): {c_wavg:.4f}')

        # 简单平均
        c_mean = float(np.mean(c_raw_calc))
        print(f'  Simple mean C_raw: {c_mean:.4f}')
    print()
