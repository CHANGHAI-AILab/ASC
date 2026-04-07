"""
全特征数值一致性验证
从原始mask重新计算所有11个SPPI特征，与Excel对比
"""
import cv2, numpy as np, pandas as pd
from scipy import ndimage
from scipy.ndimage import distance_transform_edt

XLA_PATH = r"D:\JMC\xla_total"
YXA_PATH = r"D:\JMC\yxa_total"
STR_PATH = r"D:\JMC\stroma_total\stroma_total"
EXCEL    = r"D:\LF\jmc_vis\病理特征汇总.xlsx"
EPS      = 1e-6

def load(pid):
    xla = cv2.imread(f'{XLA_PATH}/{pid}_predictions.jpg', cv2.IMREAD_GRAYSCALE)
    yxa = cv2.imread(f'{YXA_PATH}/{pid}_predictions.jpg', cv2.IMREAD_GRAYSCALE)
    stm = cv2.imread(f'{STR_PATH}/{pid}_predictions.jpg', cv2.IMREAD_GRAYSCALE)
    scc = (xla > 0).astype(np.uint8)
    adc = np.where(scc==1, 0, (yxa>0).astype(np.uint8)).astype(np.uint8)
    h, w = scc.shape
    if stm is not None and stm.shape != (h,w):
        stm = cv2.resize(stm, (w,h), interpolation=cv2.INTER_NEAREST)
    return scc, adc, stm

def logit(p): return np.log(np.clip(p,EPS,1-EPS) / (1-np.clip(p,EPS,1-EPS)))

def compute_all(scc, adc, stm):
    h, w = scc.shape
    total_px = h * w
    tumor = ((scc>0)|(adc>0)).astype(np.uint8)
    scc_px = int(np.sum(scc))
    tumor_px = int(np.sum(tumor))

    # ── C: logit(SCC / tumor) ──────────────────────────────
    C = float(logit(scc_px / tumor_px)) if tumor_px > 0 else 0

    # ── D1: 1 - DCR ────────────────────────────────────────
    labeled, n = ndimage.label(scc, structure=np.ones((3,3)))
    areas = [np.sum(labeled==i) for i in range(1,n+1)]
    valid = [a for a in areas if a >= 10]
    if valid and scc_px > 0:
        DCR = max(valid) / scc_px
    else:
        DCR = 0
    D1 = 1 - DCR

    # ── D2: Multifocal_any ─────────────────────────────────
    sig = [a for a in valid if a >= 0.05*scc_px]
    D2 = 1 if (len(sig) >= 2 or (len(valid)>=2 and sorted(valid,reverse=True)[1]/scc_px >= 0.05)) else 0

    # ── D3: log(1+CCCount) ─────────────────────────────────
    D3 = float(np.log(1 + len(valid)))

    # ── D4: log(1+LesionCount) ─────────────────────────────
    D4 = float(np.log(1 + len(sig)))

    # ── D5: SecondClusterFrac ──────────────────────────────
    sorted_areas = sorted(valid, reverse=True)
    D5 = sorted_areas[1]/scc_px if len(sorted_areas)>=2 and scc_px>0 else 0

    # ── B: log(ShapeIndex) ─────────────────────────────────
    kernel = np.ones((3,3), np.uint8)
    smoothed = cv2.morphologyEx(scc, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c,True) for c in cnts) if cnts else 0
    shape_idx = perim / (2*np.sqrt(np.pi*scc_px)) if scc_px>0 else 1
    B = float(np.log(shape_idx)) if shape_idx>0 else 0

    # ── S1: Interface_wavg ─────────────────────────────────
    scc_bnd = cv2.Canny(scc*255, 50, 150).astype(bool)
    kern5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    adc_dil = cv2.dilate(adc, kern5, iterations=1)
    bnd_px = int(np.sum(scc_bnd))
    iface_px = int(np.sum(scc_bnd & (adc_dil>0)))
    S1 = iface_px/bnd_px if bnd_px>0 else 0

    # ── S2: FrontSCC_wavg ──────────────────────────────────
    dist_map = distance_transform_edt(tumor)
    front_thresh = 500  # pixels (original scale)
    front_zone = (dist_map>0) & (dist_map<=front_thresh)
    scc_front = int(np.sum((scc>0) & front_zone))
    S2 = scc_front/scc_px if scc_px>0 else 0

    # ── S3: -log(1+FrontDist) ──────────────────────────────
    scc_dists = dist_map[scc>0]
    mean_dist = float(np.mean(scc_dists)) if len(scc_dists)>0 else 0
    S3 = -float(np.log(1+mean_dist))

    # ── V: logit(tumor/total) ──────────────────────────────
    V = float(logit(tumor_px/total_px))

    return dict(C=C,D1=D1,D2=D2,D3=D3,D4=D4,D5=D5,B=B,S1=S1,S2=S2,S3=S3,V=V)

# ── Main ───────────────────────────────────────────────────
df = pd.read_excel(EXCEL)
df['patient_id'] = df['patient_id'].astype(str)
COL = {'C':'C_raw','D1':'D1_raw','D2':'D2_raw','D3':'D3_raw','D4':'D4_raw',
       'D5':'D5_raw','B':'B_raw','S1':'S1_raw','S2':'S2_raw','S3':'S3_raw','V':'V_raw'}

THRESH = {'C':0.05,'D1':0.05,'D2':0.01,'D3':0.05,'D4':0.05,
          'D5':0.01,'B':0.05,'S1':0.05,'S2':0.05,'S3':0.05,'V':0.05}

for pid in ['2202521A6','2314986A5']:
    scc, adc, stm = load(pid)
    calc = compute_all(scc, adc, stm)
    row  = df[df['patient_id']==pid].iloc[0]
    print(f'=== {pid} ===')
    print(f'  {"Feat":4s}  {"Calc":>10s}  {"Excel":>10s}  {"Diff":>10s}  Status')
    print(f'  {"-"*55}')
    for fk, col in COL.items():
        cv = calc[fk]
        ev = float(row[col])
        diff = abs(cv-ev)
        ok = diff < THRESH[fk]
        flag = 'OK' if ok else '*** MISMATCH ***'
        print(f'  {fk:4s}  {cv:>10.4f}  {ev:>10.4f}  {diff:>10.4f}  {flag}')
    print()
