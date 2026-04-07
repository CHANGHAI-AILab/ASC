#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPPI Patient-Level Feature Visualization
参照参考图风格：每个患者生成一张包含5个域(C/D/B/S/V)的直观可视化图
基于实际分割mask，不使用直方图
"""

import os, sys
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 路径配置
# ============================================================================
XLA_PATH   = r"D:\JMC\xla_total"       # SCC mask
YXA_PATH   = r"D:\JMC\yxa_total"       # ADC mask
EXCEL_PATH = r"D:\LF\jmc_vis\病理特征汇总.xlsx"
OUTPUT_ROOT= r"D:\LF\jmc_vis\sppi_patient_vis"

BG   = '#0A0A18'
DARK = '#10102A'

# ============================================================================
# Mask 加载
# ============================================================================
def load_masks(pid):
    xf = os.path.join(XLA_PATH, pid + '_predictions.jpg')
    yf = os.path.join(YXA_PATH, pid + '_predictions.jpg')
    if not os.path.exists(xf) or not os.path.exists(yf):
        return None, None, None
    xla = cv2.imread(xf, cv2.IMREAD_GRAYSCALE)
    yxa = cv2.imread(yf, cv2.IMREAD_GRAYSCALE)
    if xla is None or yxa is None:
        return None, None, None
    scc = (xla > 0).astype(np.uint8)
    adc_raw = (yxa > 0).astype(np.uint8)
    adc = np.where(scc == 1, 0, adc_raw).astype(np.uint8)  # ADC去除SCC覆盖
    fusion = np.zeros_like(scc, dtype=np.uint8)
    fusion[adc == 1] = 1
    fusion[scc == 1] = 2
    return scc, adc, fusion

def downsample(mask, max_dim=600):
    h, w = mask.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        nh, nw = int(h * scale), int(w * scale)
        if mask.ndim == 2:
            return cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        else:
            return cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return mask

# ============================================================================
# 域可视化函数
# ============================================================================

def make_seg_rgb(scc, adc):
    """基础分割RGB图：ADC=蓝, SCC=红, 背景=深色"""
    rgb = np.zeros((*scc.shape, 3), dtype=np.uint8)
    rgb[adc == 1] = [30, 80, 200]    # 蓝 ADC
    rgb[scc == 1] = [200, 40, 40]    # 红 SCC
    return rgb


def panel_C(ax, scc, adc, row):
    """C域: Composition - 饼图叠加在分割图上"""
    rgb = make_seg_rgb(scc, adc)
    ax.imshow(rgb)
    ax.set_facecolor(DARK)

    scc_pt = float(row['C_raw']) if 'C_raw' in row.index else 0
    # 反算原始比例: logit^-1
    scc_frac = 1 / (1 + np.exp(-scc_pt))
    adc_frac = 1 - scc_frac

    # 饼图嵌入右上角
    ax_inset = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
    wedge_colors = ['#C82828', '#1E50C8']
    ax_inset.pie([scc_frac, adc_frac], colors=wedge_colors,
                 startangle=90, counterclock=False,
                 wedgeprops=dict(linewidth=0.5, edgecolor='white'))
    ax_inset.set_facecolor(DARK)

    pct = int(round(adc_frac * 100))
    ax_inset.text(0, 0, f'{pct}%', ha='center', va='center',
                  color='white', fontsize=8, fontweight='bold')

    ax.text(0.98, 0.98, 'SCC | Adeno', transform=ax.transAxes,
            color='white', fontsize=6.5, ha='right', va='top')

    ax.text(0.02, 0.02, f'C_raw = logit(SCC_in_tumor_pt)\nSCC proportion within tumor area',
            transform=ax.transAxes, color='#AAAACC', fontsize=6, va='bottom')

    _domain_badge(ax, 'C', '#E05C5C')
    ax.set_title('Composition', color='#E05C5C', fontsize=9, pad=3, loc='left', x=0.12)
    ax.axis('off')


def panel_D(ax, scc, adc, row):
    """D域: Dispersion - 连通域标注+编号"""
    rgb = make_seg_rgb(scc, adc)
    ax.imshow(rgb)
    ax.set_facecolor(DARK)

    labeled, n = ndimage.label(scc, structure=np.ones((3, 3)))
    # 过滤小连通域
    areas = [(i, np.sum(labeled == i)) for i in range(1, n + 1)]
    areas = [(i, a) for i, a in areas if a >= 20]
    areas.sort(key=lambda x: -x[1])

    # 为每个连通域画彩色轮廓+编号
    colors_cycle = ['#FF6B6B','#FFD93D','#6BCB77','#4D96FF','#FF922B',
                    '#CC5DE8','#20C997','#F06595','#74C0FC','#A9E34B']
    h, w = scc.shape
    for rank, (idx, area) in enumerate(areas[:20]):
        comp = (labeled == idx).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = colors_cycle[rank % len(colors_cycle)]
        cr, cg, cb = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        cv2.drawContours(rgb, contours, -1, (cr, cg, cb), max(1, h // 300))
        # 质心标号
        ys, xs = np.where(labeled == idx)
        cy_c, cx_c = int(np.mean(ys)), int(np.mean(xs))
        ax.text(cx_c / w, 1 - cy_c / h, f'#{rank+1}',
                transform=ax.transAxes, color=c, fontsize=5.5,
                ha='center', va='center', fontweight='bold')

    ax.imshow(rgb)  # 重绘带轮廓的图

    n_clusters = len(areas)
    d1 = float(row.get('D1_raw', 0))
    d2 = int(row.get('D2_raw', 0))
    d3 = float(row.get('D3_raw', 0))
    d4 = float(row.get('D4_raw', 0))
    d5 = float(row.get('D5_raw', 0))

    info = (f'{n_clusters} SCC clusters identified\n'
            f'D1: 1-DCR_wavg  |  D2: Multifocal_any\n'
            f'D3: log(1+CCCount)  |  D4: log(1+LesionCount)\n'
            f'D5: SecondClusterFrac_max')
    ax.text(0.02, 0.02, info, transform=ax.transAxes,
            color='#AAAACC', fontsize=5.5, va='bottom', linespacing=1.5)

    _domain_badge(ax, 'D', '#E08C3C')
    ax.set_title('Dispersion', color='#E08C3C', fontsize=9, pad=3, loc='left', x=0.12)
    ax.axis('off')


def panel_B(ax, scc, adc, row):
    """B域: Boundary - 边界轮廓高亮 + 局部放大框"""
    rgb = make_seg_rgb(scc, adc)
    h, w = scc.shape

    # 提取SCC轮廓，橙色高亮
    kernel = np.ones((3, 3), np.uint8)
    smoothed = cv2.morphologyEx(scc, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thick = max(1, h // 250)
    cv2.drawContours(rgb, contours, -1, (255, 165, 0), thick)

    ax.imshow(rgb)
    ax.set_facecolor(DARK)

    # 找最大连通域的bbox，画放大框
    labeled, n = ndimage.label(scc, structure=np.ones((3, 3)))
    if n > 0:
        areas = [(i, np.sum(labeled == i)) for i in range(1, n + 1)]
        largest_idx = max(areas, key=lambda x: x[1])[0]
        ys, xs = np.where(labeled == largest_idx)
        y1, y2 = max(0, ys.min() - 10), min(h, ys.max() + 10)
        x1, x2 = max(0, xs.min() - 10), min(w, xs.max() + 10)

        # 橙色矩形框
        rect = mpatches.Rectangle((x1 / w, 1 - y2 / h),
                                   (x2 - x1) / w, (y2 - y1) / h,
                                   linewidth=1.5, edgecolor='#FF8C00',
                                   facecolor='none', transform=ax.transAxes)
        ax.add_patch(rect)

        # 右上角放大inset
        crop = rgb[y1:y2, x1:x2]
        if crop.size > 0:
            ax_ins = ax.inset_axes([0.55, 0.45, 0.44, 0.52])
            ax_ins.imshow(crop)
            ax_ins.set_facecolor(DARK)
            ax_ins.axis('off')
            ax_ins.set_title('Boundary Detail', color='#FF8C00', fontsize=6, pad=2)

    b_val = float(row.get('B_raw', 0))
    info = (f'B_raw = log(ShapeIndex_wavg)\n'
            f'ShapeIndex = Perimeter / (2√(π·Area))\n'
            f'Higher value = more irregular boundary')
    ax.text(0.02, 0.02, info, transform=ax.transAxes,
            color='#AAAACC', fontsize=6, va='bottom', linespacing=1.5)

    _domain_badge(ax, 'B', '#F5A623')
    ax.set_title('Boundary', color='#F5A623', fontsize=9, pad=3, loc='left', x=0.12)
    ax.axis('off')


def panel_S(ax, scc, adc, row):
    """S域: Spatial Relationship - 接触边界+前沿距离热图"""
    h, w = scc.shape
    rgb = make_seg_rgb(scc, adc)

    tumor = ((scc > 0) | (adc > 0)).astype(np.uint8)

    # 接触边界（interface zone）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    scc_dilated = cv2.dilate(scc, kernel, iterations=2)
    adc_dilated = cv2.dilate(adc, kernel, iterations=2)
    interface = ((scc_dilated > 0) & (adc > 0)) | ((adc_dilated > 0) & (scc > 0))
    interface = interface & (tumor > 0)

    # 前沿区域（距肿瘤边界500px内）
    dist_map = distance_transform_edt(tumor)
    front_zone = (dist_map > 0) & (dist_map <= max(30, h // 20))

    # 叠加颜色
    rgb[interface] = [0, 220, 100]      # 绿色：接触边界
    rgb[front_zone & (scc > 0)] = [255, 220, 0]  # 黄色：SCC在前沿

    ax.imshow(rgb)
    ax.set_facecolor(DARK)

    # 图例
    legend_items = [
        mpatches.Patch(color='#1E50C8', label='Adenocarcinoma'),
        mpatches.Patch(color='#C82828', label='Squamous (SCC)'),
        mpatches.Patch(color='#00DC64', label='Interface zone'),
        mpatches.Patch(color='#FFDC00', label='Contact edge'),
    ]
    ax.legend(handles=legend_items, loc='upper right', fontsize=5.5,
              facecolor='#1A1A3A', labelcolor='white', framealpha=0.8,
              handlelength=1.2, borderpad=0.5)

    # FrontDist 标注
    s3_val = float(row.get('S3_raw', 0))
    # 在肿瘤质心附近标注
    ys, xs = np.where(tumor > 0)
    if len(ys) > 0:
        cy_c, cx_c = int(np.mean(ys)), int(np.mean(xs))
        ax.annotate('FrontDist↓', xy=(cx_c / w, 1 - cy_c / h),
                    xytext=(cx_c / w + 0.05, 1 - cy_c / h + 0.05),
                    xycoords='axes fraction', textcoords='axes fraction',
                    color='#FFD700', fontsize=6.5,
                    arrowprops=dict(arrowstyle='->', color='#FFD700', lw=0.8))

    info = ('S1: Interface_wavg (SCC-Adeno contact fraction)\n'
            'S2: FrontSCC_wavg (SCC at invasion front)\n'
            'S3: −log(1 + FrontDist_um_wavg) (proximity)')
    ax.text(0.02, 0.02, info, transform=ax.transAxes,
            color='#AAAACC', fontsize=5.5, va='bottom', linespacing=1.5)

    _domain_badge(ax, 'S', '#4CAF50')
    ax.set_title('Spatial Relationship', color='#4CAF50', fontsize=9, pad=3, loc='left', x=0.12)
    ax.axis('off')


def panel_V(ax, scc, adc, row):
    """V域: Scale/Consistency - 肿瘤/间质分区色块 + 横向比例条"""
    h, w = scc.shape

    tumor = ((scc > 0) | (adc > 0)).astype(np.uint8)
    stroma = np.ones_like(tumor) - tumor  # 背景视为间质

    # 橙色=肿瘤, 蓝灰=间质
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[tumor == 1] = [220, 120, 50]   # 橙色 Tumor
    rgb[stroma == 1] = [60, 100, 140]  # 蓝灰 Stroma

    ax.imshow(rgb)
    ax.set_facecolor(DARK)

    # 比例条 inset（右上）
    tumor_frac = np.sum(tumor) / tumor.size
    stroma_frac = 1 - tumor_frac
    tumor_pct = int(round(tumor_frac * 100))

    ax_bar = ax.inset_axes([0.35, 0.72, 0.62, 0.22])
    ax_bar.set_facecolor(DARK)
    ax_bar.barh([0, 1], [tumor_frac, stroma_frac],
                color=['#DC7832', '#3C6488'], height=0.6)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_yticks([0, 1])
    ax_bar.set_yticklabels(['Tumor', 'Stroma'], color='white', fontsize=6)
    ax_bar.tick_params(colors='white', labelsize=6)
    ax_bar.set_title('Tissue Composition', color='white', fontsize=6.5, pad=2)
    for sp in ax_bar.spines.values():
        sp.set_edgecolor('#444466')
    ax_bar.xaxis.label.set_color('white')
    ax_bar.text(tumor_frac + 0.02, 0, f'{tumor_pct}%',
                color='white', fontsize=7, va='center')

    v_val = float(row.get('V_raw', 0))
    info = (f'V_raw = logit(Tumor_tissue_wavg)\n'
            f'Tumor_tissue = area(Tumor) / area(Tissue) = {tumor_pct}%')
    ax.text(0.02, 0.02, info, transform=ax.transAxes,
            color='#AAAACC', fontsize=6, va='bottom', linespacing=1.5)

    # 图例
    legend_items = [
        mpatches.Patch(color='#DC7832', label='Tumor (SCC + Adeno)'),
        mpatches.Patch(color='#3C6488', label='Stroma'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=6,
              facecolor='#1A1A3A', labelcolor='white', framealpha=0.8)

    _domain_badge(ax, 'V', '#7E57C2')
    ax.set_title('Scale / Consistency', color='#7E57C2', fontsize=9, pad=3, loc='left', x=0.12)
    ax.axis('off')


def _domain_badge(ax, letter, color):
    """左上角域字母徽章"""
    ax.text(0.02, 0.97, letter, transform=ax.transAxes,
            fontsize=14, color=color, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=DARK,
                      edgecolor=color, linewidth=1.5))


# ============================================================================
# 主可视化函数：单患者
# ============================================================================

def visualize_patient(pid, row, output_dir):
    scc, adc, fusion = load_masks(pid)
    if scc is None:
        print(f'  [SKIP] {pid}: mask not found')
        return

    # 降采样加速
    scc  = downsample(scc,  600)
    adc  = downsample(adc,  600)

    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # 标题
    fig.text(0.5, 0.975, 'SPPI: Histologic Feature Domains on H&E Whole-Slide Images',
             ha='center', va='top', fontsize=13, color='white', fontweight='bold')
    fig.text(0.5, 0.952,
             '5 Domains · 11 Patient-Level Descriptors · DeepLab-v3+ Segmentation  '
             '|  blue = Adenocarcinoma  |  red = SCC  |  green = Stroma',
             ha='center', va='top', fontsize=7.5, color='#AAAACC')
    fig.text(0.5, 0.935, f'Patient: {pid}',
             ha='center', va='top', fontsize=9, color='#88CCFF')

    gs = GridSpec(2, 3, figure=fig,
                  top=0.91, bottom=0.04,
                  left=0.03, right=0.97,
                  hspace=0.06, wspace=0.04)

    ax_c = fig.add_subplot(gs[0, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_s = fig.add_subplot(gs[1, 0:2])
    ax_v = fig.add_subplot(gs[1, 2])

    for ax in [ax_c, ax_d, ax_b, ax_s, ax_v]:
        ax.set_facecolor(DARK)
        for sp in ax.spines.values():
            sp.set_edgecolor('#333355')

    panel_C(ax_c, scc, adc, row)
    panel_D(ax_d, scc, adc, row)
    panel_B(ax_b, scc, adc, row)
    panel_S(ax_s, scc, adc, row)
    panel_V(ax_v, scc, adc, row)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{pid}_SPPI_domains.jpg')
    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    return out_path


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    df = pd.read_excel(EXCEL_PATH)
    df['patient_id'] = df['patient_id'].astype(str)

    # 可指定单个患者测试: python visualize_sppi_patient.py 1509454C1
    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = df['patient_id'].tolist()

    print(f'Processing {len(targets)} patients...')
    ok, skip = 0, 0
    for pid in targets:
        rows = df[df['patient_id'] == pid]
        if rows.empty:
            print(f'  [SKIP] {pid}: not in Excel')
            skip += 1
            continue
        row = rows.iloc[0]
        out = visualize_patient(pid, row, OUTPUT_ROOT)
        if out:
            print(f'  OK: {out}')
            ok += 1
        else:
            skip += 1

    print(f'\nDone. {ok} saved, {skip} skipped.')
    print(f'Output: {OUTPUT_ROOT}')
