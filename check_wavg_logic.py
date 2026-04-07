"""
验证wavg = 同病理号所有切片的加权平均
"""
import pandas as pd, numpy as np

df_csv = pd.read_csv('D:/LF/jmc_vis/4_complete_sample_analysis.csv', encoding='gbk')
df_xl  = pd.read_excel('D:/LF/jmc_vis/病理特征汇总.xlsx')
df_xl['patient_id'] = df_xl['patient_id'].astype(str)

EPS = 1e-6
def logit(p): return np.log(np.clip(p,EPS,1-EPS)/(1-np.clip(p,EPS,1-EPS)))

# 2202521A6 的病理号
for pid in ['2202521A6','2314986A5']:
    xl_row = df_xl[df_xl['patient_id']==pid].iloc[0]
    # 找CSV中该病理号的所有切片（病理号列）
    pathology_id = str(xl_row.get('dataset', ''))  # check what columns exist
    print(f'Excel columns: {df_xl.columns.tolist()}')
    break

# 看CSV里病理号列
print('CSV 病理号 sample:', df_csv['病理号'].head(10).tolist())
print()

for pid in ['2202521A6','2314986A5']:
    xl_row = df_xl[df_xl['patient_id']==pid].iloc[0]
    # 找该patient在CSV里的病理号
    csv_row = df_csv[df_csv['样本名称'].astype(str).str.startswith(pid)]
    if csv_row.empty:
        print(f'{pid}: not found in CSV'); continue
    pathno = csv_row['病理号'].values[0]
    print(f'{pid} -> 病理号: {pathno}')

    # 找同病理号所有切片
    same = df_csv[df_csv['病理号'] == pathno]
    print(f'  Same pathology slices ({len(same)}): {same["样本名称"].tolist()}')

    scc_areas = same['SCC总面积'].values.astype(float)
    w = scc_areas / scc_areas.sum() if scc_areas.sum()>0 else np.ones(len(same))/len(same)

    # C_raw: logit(SCC/(SCC+ADC)) per slice, then wavg
    scc_r = same['鳞癌比例'].values.astype(float)
    adc_r = same['腺癌比例'].values.astype(float)
    tumor_r = scc_r + adc_r
    scc_in_tumor = np.where(tumor_r>0, scc_r/tumor_r, EPS)
    c_per = logit(scc_in_tumor)
    c_wavg = float(np.sum(w * c_per))
    print(f'  C_raw wavg={c_wavg:.4f}  excel={xl_row["C_raw"]:.4f}')

    # D1: 1-DCR wavg
    dcr = same['SCC最大连通域占比(DCR)'].values.astype(float)
    d1_wavg = float(np.sum(w * (1-dcr)))
    print(f'  D1_raw wavg={d1_wavg:.4f}  excel={xl_row["D1_raw"]:.4f}')

    # D3: log(1+CCCount) wavg
    cc = same['SCC连通域数量'].values.astype(float)
    d3_wavg = float(np.sum(w * np.log(1+cc)))
    print(f'  D3_raw wavg={d3_wavg:.4f}  excel={xl_row["D3_raw"]:.4f}')

    # D4: log(1+LesionCount) max (not wavg?)
    lc = same['SCC显著病灶数量'].values.astype(float)
    d4_wavg = float(np.sum(w * np.log(1+lc)))
    d4_max  = float(np.log(1+lc.max()))
    print(f'  D4_raw wavg={d4_wavg:.4f}  max={d4_max:.4f}  excel={xl_row["D4_raw"]:.4f}')

    # D5: SecondClusterFrac max
    sc = same['SCC第二大连通域占比'].values.astype(float)
    d5_wavg = float(np.sum(w * sc))
    d5_max  = float(sc.max())
    print(f'  D5_raw wavg={d5_wavg:.4f}  max={d5_max:.4f}  excel={xl_row["D5_raw"]:.4f}')

    # B: log(ShapeIndex) wavg
    si = same['SCC形状因子'].values.astype(float)
    b_wavg = float(np.sum(w * np.log(si)))
    print(f'  B_raw wavg={b_wavg:.4f}  excel={xl_row["B_raw"]:.4f}')

    # S1: Interface wavg
    iface = same['SCC与ADC接触边界比例'].values.astype(float)
    s1_wavg = float(np.sum(w * iface))
    print(f'  S1_raw wavg={s1_wavg:.4f}  excel={xl_row["S1_raw"]:.4f}')

    # S3: -log(1+FrontDist) wavg
    fd = same['SCC到前沿平均距离(um)'].values.astype(float)
    s3_wavg = float(np.sum(w * (-np.log(1+fd))))
    print(f'  S3_raw wavg={s3_wavg:.4f}  excel={xl_row["S3_raw"]:.4f}')

    # V: logit(tumor/total) -- need total pixels, use (SCC+ADC proportions * total)
    # total pixels unknown from CSV, try logit of (scc+adc proportion) directly
    tumor_prop = scc_r + adc_r  # fraction of total image
    v_wavg = float(np.sum(w * logit(tumor_prop)))
    print(f'  V_raw wavg={v_wavg:.4f}  excel={xl_row["V_raw"]:.4f}')
    print()
