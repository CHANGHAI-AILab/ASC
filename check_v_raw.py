import cv2, numpy as np, pandas as pd

df = pd.read_excel('D:/LF/jmc_vis/病理特征汇总.xlsx')
df['patient_id'] = df['patient_id'].astype(str)

for pid in ['2202521A6', '2314986A5']:
    xla = cv2.imread(f'D:/JMC/xla_total/{pid}_predictions.jpg', cv2.IMREAD_GRAYSCALE)
    yxa = cv2.imread(f'D:/JMC/yxa_total/{pid}_predictions.jpg', cv2.IMREAD_GRAYSCALE)
    str_img = cv2.imread(f'D:/JMC/stroma_total/stroma_total/{pid}_predictions.jpg', cv2.IMREAD_GRAYSCALE)
    h, w = xla.shape
    str_orig_shape = str_img.shape
    if str_img.shape != (h, w):
        str_img = cv2.resize(str_img, (w, h), interpolation=cv2.INTER_NEAREST)

    scc    = (xla > 0).astype(np.uint8)
    adc    = np.where(scc==1, 0, (yxa>0).astype(np.uint8))
    tumor  = ((scc>0)|(adc>0)).astype(np.uint8)
    stroma = (str_img > 0).astype(np.uint8)
    str_only = ((stroma>0) & (tumor==0)).astype(np.uint8)

    tissue_px = int(np.sum(tumor)) + int(np.sum(str_only))
    t_frac = np.sum(tumor) / tissue_px
    s_frac = np.sum(str_only) / tissue_px
    v_excel = float(df[df['patient_id']==pid]['V_raw'].values[0])
    v_from_mask = float(np.log(t_frac / (1 - t_frac)))

    print(f'=== {pid} ===')
    print(f'  xla shape: {xla.shape},  stroma orig shape: {str_orig_shape}')
    print(f'  tumor_px : {int(np.sum(tumor)):>8}')
    print(f'  stroma_px: {int(np.sum(str_only)):>8}')
    print(f'  tissue_px: {tissue_px:>8}')
    print(f'  tumor_frac : {t_frac:.4f}  ({int(round(t_frac*100))}%)')
    print(f'  stroma_frac: {s_frac:.4f}  ({int(round(s_frac*100))}%)')
    print(f'  V_raw (Excel)    : {v_excel:.4f}')
    print(f'  logit(tumor_frac): {v_from_mask:.4f}')
    print(f'  Match: {abs(v_excel - v_from_mask) < 0.01}')
    print()
