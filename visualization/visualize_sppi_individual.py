#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPPI 单特征可视化 - 每个特征单独一张图
11个特征: C, D1, D2, D3, D4, D5, B, S1, S2, S3, V
每张图: 左=分割mask直观可视化, 右=特征说明+数值
"""

import os, sys
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
XLA_PATH    = r"D:\JMC\xla_total"
YXA_PATH    = r"D:\JMC\yxa_total"
STR_PATH    = r"D:\JMC\stroma_total\stroma_total"
EXCEL_PATH  = r"D:\LF\jmc_vis\病理特征汇总.xlsx"
OUTPUT_ROOT = r"D:\LF\jmc_vis\sppi_individual_vis"

BG   = '#0A0A18'
DARK = '#10102A'

FEAT_META = {
    'C':  {'domain':'C', 'color':'#E05C5C', 'title':'Composition',
           'formula':'C_raw = logit(SCC_in_tumor_pt)',
           'desc':'SCC proportion within tumor area (logit-transformed)'},
    'D1': {'domain':'D', 'color':'#E08C3C', 'title':'Dispersion: 1−DCR',
           'formula':'D1_raw = 1 − DCR_wavg',
           'desc':'1 minus dominant cluster ratio\n(higher = more dispersed)'},
    'D2': {'domain':'D', 'color':'#E0A03C', 'title':'Dispersion: Multifocality',
           'formula':'D2_raw = Multifocal_any',
           'desc':'Binary multifocality flag\n(1 = multifocal, 0 = unifocal)'},
    'D3': {'domain':'D', 'color':'#C8B040', 'title':'Dispersion: CC Count',
           'formula':'D3_raw = log(1 + CCCount_max)',
           'desc':'Log-transformed connected component count'},
    'D4': {'domain':'D', 'color':'#A0B840', 'title':'Dispersion: Lesion Count',
           'formula':'D4_raw = log(1 + LesionCount_max)',
           'desc':'Log-transformed significant lesion count'},
    'D5': {'domain':'D', 'color':'#70B840', 'title':'Dispersion: 2nd Cluster Frac',
           'formula':'D5_raw = SecondClusterFrac_max',
           'desc':'Second largest cluster fraction\n(relative to total SCC area)'},
    'B':  {'domain':'B', 'color':'#F5A623', 'title':'Boundary Complexity',
           'formula':'B_raw = log(ShapeIndex_wavg)',
           'desc':'Log shape index = log(Perimeter / 2√(π·Area))\nHigher = more irregular boundary'},
    'S1': {'domain':'S', 'color':'#4CAF50', 'title':'Spatial: Interface Fraction',
           'formula':'S1_raw = Interface_wavg',
           'desc':'SCC-ADC contact boundary fraction\n(proportion of SCC boundary touching ADC)'},
    'S2': {'domain':'S', 'color':'#26A69A', 'title':'Spatial: Front SCC Fraction',
           'formula':'S2_raw = FrontSCC_wavg',
           'desc':'Fraction of SCC at invasion front\n(proximity to tumor edge)'},
    'S3': {'domain':'S', 'color':'#29B6F6', 'title':'Spatial: Front Distance',
           'formula':'S3_raw = −log(1 + FrontDist_um_wavg)',
           'desc':'Negative log mean distance to invasion front\n(higher = closer to front)'},
    'V':  {'domain':'V', 'color':'#7E57C2', 'title':'Scale / Consistency',
           'formula':'V_raw = logit(Tumor_tissue_wavg)',
           'desc':'Tumor tissue fraction (logit-transformed)\nTumor_tissue = area(Tumor)/area(Tissue)'},
}

COL_MAP = {
    'C':'C_raw','D1':'D1_raw','D2':'D2_raw','D3':'D3_raw','D4':'D4_raw',
    'D5':'D5_raw','B':'B_raw','S1':'S1_raw','S2':'S2_raw','S3':'S3_raw','V':'V_raw'
}

# ============================================================================
# Mask 工具
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
    adc = np.where(scc == 1, 0, (yxa > 0).astype(np.uint8)).astype(np.uint8)
    # stroma mask（概率图，>0即为间质预测区域）
    sf = os.path.join(STR_PATH, pid + '_predictions.jpg')
    if os.path.exists(sf):
        str_img = cv2.imread(sf, cv2.IMREAD_GRAYSCALE)
        stroma = (str_img > 0).astype(np.uint8) if str_img is not None else None
    else:
        stroma = None
    return scc, adc, stroma

def downsample(mask, max_dim=700):
    h, w = mask.shape
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        return cv2.resize(mask, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
    return mask

def base_rgb(scc, adc):
    rgb = np.zeros((*scc.shape, 3), dtype=np.uint8)
    rgb[adc == 1] = [30, 80, 200]
    rgb[scc == 1] = [200, 40, 40]
    return rgb

def dark_ax(ax):
    ax.set_facecolor(DARK)
    ax.tick_params(colors='white', labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333355')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')


def info_panel(ax, fkey, meta, value, pid):
    """右侧信息面板"""
    ax.set_facecolor(DARK)
    for sp in ax.spines.values():
        sp.set_edgecolor(meta['color'])
        sp.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])

    color = meta['color']
    y = 0.93
    ax.text(0.08, y, meta['domain'], transform=ax.transAxes,
            fontsize=28, color=color, fontweight='bold', va='top')
    ax.text(0.28, y, meta['title'], transform=ax.transAxes,
            fontsize=12, color=color, fontweight='bold', va='top')

    ax.text(0.08, 0.72, meta['formula'], transform=ax.transAxes,
            fontsize=9.5, color='#DDDDFF', va='top', family='monospace')
    ax.text(0.08, 0.60, meta['desc'], transform=ax.transAxes,
            fontsize=8.5, color='#AAAACC', va='top', linespacing=1.6)

    # 数值框
    val_str = f'{value:.4f}' if isinstance(value, float) else str(value)
    ax.text(0.08, 0.42, 'Value:', transform=ax.transAxes,
            fontsize=9, color='#888899', va='top')
    ax.text(0.08, 0.33, val_str, transform=ax.transAxes,
            fontsize=20, color='#FFD700', fontweight='bold', va='top')

    ax.text(0.08, 0.18, f'Patient: {pid}', transform=ax.transAxes,
            fontsize=9, color='#88CCFF', va='top')
    ax.text(0.08, 0.10, f'Feature key: {COL_MAP[fkey]}', transform=ax.transAxes,
            fontsize=8, color='#666688', va='top', family='monospace')


# ============================================================================
# 各特征可视化函数  render_<key>(ax_vis, scc, adc, row)
# ============================================================================

def render_C(ax, scc, adc, row):
    rgb = base_rgb(scc, adc)
    ax.imshow(rgb); ax.axis('off')
    h, w = scc.shape
    # 饼图 inset
    val = float(row.get('C_raw', 0))
    scc_frac = 1 / (1 + np.exp(-val))
    adc_frac = 1 - scc_frac
    ax_pie = ax.inset_axes([0.55, 0.52, 0.43, 0.43])
    ax_pie.pie([scc_frac, adc_frac], colors=['#C82828','#1E50C8'],
               startangle=90, counterclock=False,
               wedgeprops=dict(linewidth=0.8, edgecolor='white'))
    ax_pie.set_facecolor(DARK)
    ax_pie.text(0, 0, f'{int(round(adc_frac*100))}%\nAdeno',
                ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    ax.text(0.97, 0.97, 'SCC | Adeno', transform=ax.transAxes,
            color='white', fontsize=7, ha='right', va='top')
    legend = [mpatches.Patch(color='#C82828', label='SCC'),
              mpatches.Patch(color='#1E50C8', label='Adenocarcinoma')]
    ax.legend(handles=legend, loc='lower left', fontsize=7,
              facecolor='#1A1A3A', labelcolor='white', framealpha=0.8)



def render_D1(ax, scc, adc, row):
    """1-DCR: 最大连通域高亮，其余灰色"""
    rgb = np.zeros((*scc.shape, 3), dtype=np.uint8)
    rgb[adc == 1] = [20, 50, 130]
    labeled, n = ndimage.label(scc, structure=np.ones((3,3)))
    if n > 0:
        areas = [(i, np.sum(labeled==i)) for i in range(1, n+1)]
        largest_idx = max(areas, key=lambda x: x[1])[0]
        for idx, _ in areas:
            if idx == largest_idx:
                rgb[labeled == idx] = [220, 40, 40]   # 最大: 亮红
            else:
                rgb[labeled == idx] = [120, 60, 60]   # 其余: 暗红
    ax.imshow(rgb); ax.axis('off')
    legend = [mpatches.Patch(color='#DC2828', label='Largest cluster (DCR)'),
              mpatches.Patch(color='#783C3C', label='Other clusters'),
              mpatches.Patch(color='#143282', label='Adenocarcinoma')]
    ax.legend(handles=legend, loc='lower left', fontsize=7,
              facecolor='#1A1A3A', labelcolor='white', framealpha=0.8)



def render_D2(ax, scc, adc, row):
    """Multifocality: 连通域彩色标注"""
    rgb = base_rgb(scc, adc)
    labeled, n = ndimage.label(scc, structure=np.ones((3,3)))
    areas = sorted([(i, np.sum(labeled==i)) for i in range(1,n+1)
                    if np.sum(labeled==i) >= 20], key=lambda x: -x[1])
    colors_c = ['#FF6B6B','#FFD93D','#6BCB77','#4D96FF','#FF922B',
                '#CC5DE8','#20C997','#F06595','#74C0FC','#A9E34B']
    h, w = scc.shape
    for rank, (idx, area) in enumerate(areas[:15]):
        comp = (labeled == idx).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = colors_c[rank % len(colors_c)]
        cr,cg,cb = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
        cv2.drawContours(rgb, cnts, -1, (cr,cg,cb), max(1, h//300))
        ys, xs = np.where(labeled == idx)
        ax.text(np.mean(xs)/w, 1-np.mean(ys)/h, f'#{rank+1}',
                transform=ax.transAxes, color=c, fontsize=6, fontweight='bold',
                ha='center', va='center')
    ax.imshow(rgb); ax.axis('off')
    val = int(row.get('D2_raw', 0))
    label_str = 'MULTIFOCAL' if val == 1 else 'UNIFOCAL'
    label_col  = '#FF6B6B' if val == 1 else '#6BCB77'
    ax.text(0.5, 0.04, label_str, transform=ax.transAxes,
            color=label_col, fontsize=13, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=DARK, edgecolor=label_col, lw=1.5))



def render_D3(ax, scc, adc, row):
    """CC Count: 每个连通域不同颜色"""
    rgb = np.zeros((*scc.shape, 3), dtype=np.uint8)
    rgb[adc == 1] = [20, 50, 130]
    labeled, n = ndimage.label(scc, structure=np.ones((3,3)))
    np.random.seed(42)
    for i in range(1, n+1):
        if np.sum(labeled==i) >= 10:
            col = np.random.randint(80, 255, 3)
            rgb[labeled == i] = col
    ax.imshow(rgb); ax.axis('off')
    ax.text(0.5, 0.04, f'{n} connected components',
            transform=ax.transAxes, color='#FFD700', fontsize=10,
            fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=DARK, edgecolor='#FFD700', lw=1.2))



def render_D4(ax, scc, adc, row):
    """Lesion Count: 显著病灶（≥5%总面积）高亮"""
    rgb = base_rgb(scc, adc)
    labeled, n = ndimage.label(scc, structure=np.ones((3,3)))
    total_scc = max(np.sum(scc), 1)
    areas = [(i, np.sum(labeled==i)) for i in range(1,n+1)]
    sig = [(i,a) for i,a in areas if a >= 0.05*total_scc]
    sig.sort(key=lambda x: -x[1])
    h, w = scc.shape
    colors_c = ['#FF6B6B','#FFD93D','#6BCB77','#4D96FF','#FF922B','#CC5DE8']
    for rank, (idx, area) in enumerate(sig):
        comp = (labeled==idx).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = colors_c[rank % len(colors_c)]
        cr,cg,cb = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
        cv2.drawContours(rgb, cnts, -1, (cr,cg,cb), max(2, h//200))
        ys, xs = np.where(labeled==idx)
        ax.text(np.mean(xs)/w, 1-np.mean(ys)/h, f'L{rank+1}',
                transform=ax.transAxes, color=c, fontsize=7, fontweight='bold',
                ha='center', va='center')
    ax.imshow(rgb); ax.axis('off')
    ax.text(0.5, 0.04, f'{len(sig)} significant lesions (≥5% SCC area)',
            transform=ax.transAxes, color='#FFD700', fontsize=9,
            fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=DARK, edgecolor='#FFD700', lw=1.2))



def render_D5(ax, scc, adc, row):
    """Second cluster fraction: 最大+第二大连通域对比"""
    rgb = np.zeros((*scc.shape, 3), dtype=np.uint8)
    rgb[adc == 1] = [20, 50, 130]
    labeled, n = ndimage.label(scc, structure=np.ones((3,3)))
    areas = sorted([(i, np.sum(labeled==i)) for i in range(1,n+1)
                    if np.sum(labeled==i)>=10], key=lambda x: -x[1])
    color_map = {0: [220,40,40], 1: [255,200,0]}   # 1st=红, 2nd=黄
    for rank, (idx, _) in enumerate(areas[:2]):
        rgb[labeled==idx] = color_map.get(rank, [100,100,100])
    for rank, (idx, _) in enumerate(areas[2:]):
        rgb[labeled==idx] = [80, 60, 60]
    ax.imshow(rgb); ax.axis('off')
    legend = [mpatches.Patch(color='#DC2828', label='1st cluster'),
              mpatches.Patch(color='#FFC800', label='2nd cluster'),
              mpatches.Patch(color='#504040', label='Others')]
    ax.legend(handles=legend, loc='lower left', fontsize=7,
              facecolor='#1A1A3A', labelcolor='white', framealpha=0.8)



def render_B(ax, scc, adc, row):
    """Boundary: 橙色轮廓高亮 + 局部放大"""
    rgb = base_rgb(scc, adc)
    h, w = scc.shape
    kernel = np.ones((3,3), np.uint8)
    smoothed = cv2.morphologyEx(scc, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, cnts, -1, (255,165,0), max(1, h//250))
    ax.imshow(rgb); ax.axis('off')
    # 最大连通域放大框
    labeled, n = ndimage.label(scc, structure=np.ones((3,3)))
    if n > 0:
        areas = [(i, np.sum(labeled==i)) for i in range(1,n+1)]
        li = max(areas, key=lambda x: x[1])[0]
        ys, xs = np.where(labeled==li)
        y1,y2 = max(0,ys.min()-15), min(h,ys.max()+15)
        x1,x2 = max(0,xs.min()-15), min(w,xs.max()+15)
        rect = mpatches.Rectangle((x1/w, 1-y2/h), (x2-x1)/w, (y2-y1)/h,
                                   lw=1.8, edgecolor='#FF8C00', facecolor='none',
                                   transform=ax.transAxes)
        ax.add_patch(rect)
        crop = rgb[y1:y2, x1:x2]
        if crop.size > 0:
            ins = ax.inset_axes([0.52, 0.44, 0.46, 0.52])
            ins.imshow(crop); ins.axis('off')
            ins.set_title('Boundary Detail', color='#FF8C00', fontsize=7, pad=2)
            for sp in ins.spines.values():
                sp.set_edgecolor('#FF8C00'); sp.set_linewidth(1.2)



def render_S1(ax, scc, adc, row):
    """Interface fraction: 接触边界绿色高亮"""
    rgb = base_rgb(scc, adc)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    adc_dil = cv2.dilate(adc, kernel, iterations=2)
    scc_bnd = cv2.Canny(scc*255, 50, 150).astype(bool)
    interface = scc_bnd & (adc_dil > 0)
    # 接触边界加粗显示
    h, w = scc.shape
    thick = max(2, h//200)
    scc_bnd_img = (scc_bnd.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(scc_bnd_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 接触像素绿色，非接触白色
    for y in range(h):
        for x in range(w):
            if scc_bnd[y,x]:
                rgb[y,x] = [0,220,80] if interface[y,x] else [200,200,200]
    ax.imshow(rgb); ax.axis('off')
    legend = [mpatches.Patch(color='#00DC50', label='SCC-ADC interface'),
              mpatches.Patch(color='#C8C8C8', label='SCC boundary (non-contact)'),
              mpatches.Patch(color='#1E50C8', label='Adenocarcinoma'),
              mpatches.Patch(color='#C82828', label='SCC')]
    ax.legend(handles=legend, loc='lower left', fontsize=7,
              facecolor='#1A1A3A', labelcolor='white', framealpha=0.8)



def render_S2(ax, scc, adc, row):
    """Front SCC fraction: 前沿SCC黄色高亮"""
    rgb = base_rgb(scc, adc)
    h, w = scc.shape
    tumor = ((scc>0)|(adc>0)).astype(np.uint8)
    dist_map = distance_transform_edt(tumor)
    front_thresh = max(30, h//18)
    front_zone = (dist_map > 0) & (dist_map <= front_thresh)
    scc_front = (scc > 0) & front_zone
    scc_inner = (scc > 0) & ~front_zone
    rgb[scc_inner] = [160, 30, 30]
    rgb[scc_front] = [255, 220, 0]
    # 前沿轮廓线
    front_edge = (dist_map > front_thresh-2) & (dist_map <= front_thresh+2) & (tumor>0)
    rgb[front_edge] = [255, 100, 0]
    ax.imshow(rgb); ax.axis('off')
    legend = [mpatches.Patch(color='#FFDC00', label='SCC at invasion front'),
              mpatches.Patch(color='#A01E1E', label='SCC (interior)'),
              mpatches.Patch(color='#FF6400', label='Front boundary'),
              mpatches.Patch(color='#1E50C8', label='Adenocarcinoma')]
    ax.legend(handles=legend, loc='lower left', fontsize=7,
              facecolor='#1A1A3A', labelcolor='white', framealpha=0.8)



def render_S3(ax, scc, adc, row):
    """Front distance: 距前沿距离热图"""
    h, w = scc.shape
    tumor = ((scc>0)|(adc>0)).astype(np.uint8)
    dist_map = distance_transform_edt(tumor)
    # 热图：仅在SCC区域显示距离
    heat = np.zeros((h, w), dtype=np.float32)
    heat[scc>0] = dist_map[scc>0]
    # 归一化
    mx = heat.max()
    if mx > 0:
        heat_norm = (heat / mx * 255).astype(np.uint8)
    else:
        heat_norm = heat.astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    # 背景用暗色
    bg_rgb = np.full((h, w, 3), 15, dtype=np.uint8)
    bg_rgb[adc>0] = [20, 50, 130]
    mask3 = np.stack([scc>0]*3, axis=-1)
    result = np.where(mask3, heat_color, bg_rgb)
    ax.imshow(result); ax.axis('off')
    # 颜色条
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, mx))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label('Distance to front (px)', color='white', fontsize=7)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=6)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    legend = [mpatches.Patch(color='#1E50C8', label='Adenocarcinoma')]
    ax.legend(handles=legend, loc='lower left', fontsize=7,
              facecolor='#1A1A3A', labelcolor='white', framealpha=0.8)



def render_V(ax, scc, adc, row, stroma=None):
    """Scale/Consistency: 肿瘤/间质/背景三类色块 + 比例条
    V_raw = logit(Tumor_tissue_wavg)
    Tumor_tissue_wavg = tumor_px / total_px (全图像素，含背景)
    """
    h, w = scc.shape
    total_px = h * w
    tumor = ((scc > 0) | (adc > 0)).astype(np.uint8)

    # 三类：肿瘤=橙, 间质=蓝灰, 背景=近黑
    rgb = np.full((h, w, 3), 12, dtype=np.uint8)
    if stroma is not None:
        if stroma.shape != (h, w):
            stroma = cv2.resize(stroma, (w, h), interpolation=cv2.INTER_NEAREST)
        str_only = ((stroma > 0) & (tumor == 0)).astype(np.uint8)
        rgb[str_only == 1] = [60, 100, 140]
    else:
        str_only = None
    rgb[tumor == 1] = [220, 120, 50]

    ax.imshow(rgb); ax.axis('off')

    # 比例：分母=全图总像素（与Excel V_raw定义一致）
    tumor_frac  = np.sum(tumor) / total_px
    bg_frac     = 1 - tumor_frac - (np.sum(str_only) / total_px if str_only is not None else 0)
    stroma_frac = np.sum(str_only) / total_px if str_only is not None else (1 - tumor_frac)

    tumor_pct  = round(tumor_frac * 100, 1)
    stroma_pct = round(stroma_frac * 100, 1)

    # 验证与Excel V_raw的一致性
    eps = 1e-6
    t_clip = np.clip(tumor_frac, eps, 1 - eps)
    v_check = float(np.log(t_clip / (1 - t_clip)))
    v_excel = float(row.get('V_raw', 0))

    # 比例条 inset（只显示肿瘤和间质，背景不在条内）
    ax_bar = ax.inset_axes([0.30, 0.74, 0.68, 0.20])
    ax_bar.set_facecolor(DARK)
    ax_bar.barh([0, 1], [tumor_frac, stroma_frac],
                color=['#DC7832', '#3C6488'], height=0.55)
    ax_bar.set_xlim(0, max(tumor_frac, stroma_frac) * 1.3)
    ax_bar.set_yticks([0, 1])
    ax_bar.set_yticklabels(['Tumor', 'Stroma'], color='white', fontsize=7)
    ax_bar.tick_params(colors='white', labelsize=6)
    ax_bar.set_title('Fraction of Total Slide Area', color='white', fontsize=7, pad=2)
    for sp in ax_bar.spines.values():
        sp.set_edgecolor('#444466')
    ax_bar.text(tumor_frac  * 1.05, 0, f'{tumor_pct}%',
                color='white', fontsize=8, va='center', fontweight='bold')
    ax_bar.text(stroma_frac * 1.05, 1, f'{stroma_pct}%',
                color='white', fontsize=8, va='center', fontweight='bold')

    legend = [mpatches.Patch(color='#DC7832', label='Tumor (SCC+Adeno)'),
              mpatches.Patch(color='#3C6488', label='Stroma'),
              mpatches.Patch(color='#0C0C0C', label='Background (slide)')]
    ax.legend(handles=legend, loc='lower left', fontsize=7,
              facecolor='#1A1A3A', labelcolor='white', framealpha=0.8)



RENDER_FN = {
    'C': render_C, 'D1': render_D1, 'D2': render_D2,
    'D3': render_D3, 'D4': render_D4, 'D5': render_D5,
    'B': render_B, 'S1': render_S1, 'S2': render_S2,
    'S3': render_S3, 'V': render_V,
}

# ============================================================================
# 单特征单患者出图
# ============================================================================

def make_feature_fig(pid, fkey, scc, adc, row, stroma=None):
    meta  = FEAT_META[fkey]
    col   = COL_MAP[fkey]
    value = float(row.get(col, 0)) if col in row.index else 0.0

    fig, (ax_vis, ax_info) = plt.subplots(1, 2, figsize=(14, 7),
                                           gridspec_kw={'width_ratios': [3, 1.2]},
                                           facecolor=BG)
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.97,
             'SPPI: Histologic Feature Domains on H&E Whole-Slide Images',
             ha='center', va='top', fontsize=11, color='white', fontweight='bold')
    fig.text(0.5, 0.945,
             '5 Domains · 11 Patient-Level Descriptors · DeepLab-v3+ Segmentation  '
             '|  blue = Adenocarcinoma  |  red = SCC',
             ha='center', va='top', fontsize=7, color='#AAAACC')

    ax_vis.set_facecolor(DARK)
    for sp in ax_vis.spines.values():
        sp.set_edgecolor(meta['color']); sp.set_linewidth(1.5)

    # V域需要传入stroma
    if fkey == 'V':
        render_V(ax_vis, scc, adc, row, stroma=stroma)
    else:
        RENDER_FN[fkey](ax_vis, scc, adc, row)

    info_panel(ax_info, fkey, meta, value, pid)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def process_patient(pid, row, output_root):
    scc, adc, stroma = load_masks(pid)
    if scc is None:
        print(f'  [SKIP] {pid}: mask not found')
        return 0
    scc    = downsample(scc,    700)
    adc    = downsample(adc,    700)
    stroma = downsample(stroma, 700) if stroma is not None else None

    saved = 0
    for fkey in FEAT_META:
        out_dir = os.path.join(output_root, fkey)
        os.makedirs(out_dir, exist_ok=True)
        col = COL_MAP[fkey]
        val = row.get(col, 0)
        val_str = f'{float(val):.4f}' if col in row.index else '0'
        out_path = os.path.join(out_dir, f'{pid}_{fkey}_{val_str}.jpg')

        try:
            fig = make_feature_fig(pid, fkey, scc, adc, row, stroma=stroma)
            fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor=BG)
            plt.close(fig)
            print(f'    {fkey}: {out_path}')
            saved += 1
        except Exception as e:
            print(f'    [ERR] {fkey}: {e}')
            plt.close('all')
    return saved


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    df = pd.read_excel(EXCEL_PATH)
    df['patient_id'] = df['patient_id'].astype(str)

    targets = sys.argv[1:] if len(sys.argv) > 1 else df['patient_id'].tolist()

    print(f'Processing {len(targets)} patients × {len(FEAT_META)} features...')
    total = 0
    for pid in targets:
        rows = df[df['patient_id'] == pid]
        if rows.empty:
            print(f'  [SKIP] {pid}: not in Excel')
            continue
        print(f'  {pid}')
        total += process_patient(pid, rows.iloc[0], OUTPUT_ROOT)

    print(f'\nDone. {total} images saved to: {OUTPUT_ROOT}')
