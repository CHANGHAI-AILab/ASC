"""
按SPPI公式精确验证：
- _pt  : 患者级合并（所有切片像素合并后计算）
- _wavg: SCC面积加权平均
- _max : 取最大值
- _any : 任意切片=1则为1
"""
import pandas as pd, numpy as np

df_csv = pd.read_csv('D:/LF/jmc_vis/4_complete_sample_analysis.csv', encoding='gbk')
df_xl  = pd.read_excel('D:/LF/jmc_vis/病理特征汇总.xlsx')
df_xl['patient_id'] = df_xl['patient_id'].astype(str)

EPS = 1e-6
def logit(p): return float(np.log(np.clip(p,EPS,1-EPS)/(1-np.clip(p,EPS,1-EPS))))

for pid in ['2202521A6','2314986A5']:
    xl_row = df_xl[df_xl['patient_id']==pid].iloc[0]
    csv_row = df_csv[df_csv['样本名称'].astype(str).str.startswith(pid)]
    pathno  = csv_row['病理号'].values[0]
    same    = df_csv[df_csv['病理号'] == pathno].copy()

    scc_area = same['SCC总面积'].values.astype(float)
    w = scc_area / scc_area.sum() if scc_area.sum()>0 else np.ones(len(same))/len(same)

    scc_r   = same['鳞癌比例'].values.astype(float)
    adc_r   = same['腺癌比例'].values.astype(float)
    tumor_r = scc_r + adc_r

    print(f'=== {pid}  (病理号:{pathno}, {len(same)} slices) ===')
    print(f'  {"Feature":8s}  {"Calc":>10s}  {"Excel":>10s}  {"Diff":>8s}  Status')
    print(f'  {"-"*58}')

    results = {}

    # C_raw = logit(SCC_in_tumor_pt)
    # _pt: 患者级合并 = sum(SCC像素)/sum(tumor像素) across all slices
    # 用比例×总像素近似：SCC_in_tumor_pt = sum(scc_r) / sum(tumor_r)
    # 但更准确：各切片SCC比例和ADC比例都是占全图，所以sum(scc)/sum(tumor)
    scc_sum   = scc_r.sum()
    tumor_sum = tumor_r.sum()
    C = logit(scc_sum/tumor_sum) if tumor_sum>0 else 0
    results['C_raw'] = C

    # D1_raw = 1 - DCR_wavg  (SCC面积加权平均)
    dcr = same['SCC最大连通域占比(DCR)'].values.astype(float)
    D1 = float(np.sum(w*(1-dcr)))
    results['D1_raw'] = D1

    # D2_raw = Multifocal_any
    multi = same['SCC多灶性'].values
    D2 = 1 if any(str(m).strip().lower()=='multifocal' for m in multi) else 0
    results['D2_raw'] = D2

    # D3_raw = log(1+CCCount_max)
    cc = same['SCC连通域数量'].values.astype(float)
    D3 = float(np.log(1+cc.max()))
    results['D3_raw'] = D3

    # D4_raw = log(1+LesionCount_max)
    lc = same['SCC显著病灶数量'].values.astype(float)
    D4 = float(np.log(1+lc.max()))
    results['D4_raw'] = D4

    # D5_raw = SecondClusterFrac_max
    sc = same['SCC第二大连通域占比'].values.astype(float)
    D5 = float(sc.max())
    results['D5_raw'] = D5

    # B_raw = log(ShapeIndex_wavg)
    si = same['SCC形状因子'].values.astype(float)
    B = float(np.log(np.sum(w*si)))
    results['B_raw'] = B

    # S1_raw = Interface_wavg
    iface = same['SCC与ADC接触边界比例'].values.astype(float)
    S1 = float(np.sum(w*iface))
    results['S1_raw'] = S1

    # S2_raw = FrontSCC_wavg
    front_scc = same['SCC靠近浸润前沿比例'].values.astype(float)
    S2 = float(np.sum(w*front_scc))
    results['S2_raw'] = S2

    # S3_raw = -log(1+FrontDist_um_wavg)
    fd = same['SCC到前沿平均距离(um)'].values.astype(float)
    S3 = float(-np.log(1+np.sum(w*fd)))
    results['S3_raw'] = S3

    # V_raw = logit(Tumor_tissue_wavg)
    # Tumor_tissue = tumor比例（占全图），wavg by SCC area
    V = logit(np.sum(w*tumor_r))
    results['V_raw'] = V

    COL_MAP = {'C_raw':'C_raw','D1_raw':'D1_raw','D2_raw':'D2_raw',
               'D3_raw':'D3_raw','D4_raw':'D4_raw','D5_raw':'D5_raw',
               'B_raw':'B_raw','S1_raw':'S1_raw','S2_raw':'S2_raw',
               'S3_raw':'S3_raw','V_raw':'V_raw'}

    all_ok = True
    for fk, col in COL_MAP.items():
        cv = results[fk]
        ev = float(xl_row[col])
        diff = abs(cv-ev)
        ok = diff < 0.001
        if not ok: all_ok = False
        flag = 'OK' if ok else f'MISMATCH (calc={cv:.4f})'
        print(f'  {fk:8s}  {cv:>10.4f}  {ev:>10.4f}  {diff:>8.4f}  {flag}')

    print(f'  {"Overall":8s}  {"ALL OK" if all_ok else "HAS MISMATCHES"}')
    print()
