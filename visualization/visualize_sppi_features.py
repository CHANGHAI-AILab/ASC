#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPPI特征可视化 - 基于病理特征汇总.xlsx
参照参考图风格，为每个SPPI特征生成分布可视化
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 配置
# ============================================================================
EXCEL_PATH = r"D:\LF\jmc_vis\病理特征汇总.xlsx"
OUTPUT_ROOT = r"D:\LF\jmc_vis\sppi_feature_visualizations"

# 特征定义
FEATURES = {
    'C_raw':  {'label': 'C (Composition)',      'formula': 'C_raw = logit(SCC_in_tumor_pt)',
               'desc': 'SCC proportion within tumor area', 'color': '#E05C5C', 'domain': 'C'},
    'D1_raw': {'label': 'D1 (1−DCR)',            'formula': 'D1_raw = 1 − DCR_wavg',
               'desc': 'SCC dispersion: 1 minus dominant cluster ratio', 'color': '#E08C3C', 'domain': 'D'},
    'D2_raw': {'label': 'D2 (Multifocal)',        'formula': 'D2_raw = Multifocal_any',
               'desc': 'Multifocality (binary: 0/1)', 'color': '#E0A03C', 'domain': 'D'},
    'D3_raw': {'label': 'D3 (log CCCount)',       'formula': 'D3_raw = log(1+CCCount_max)',
               'desc': 'Log-transformed connected component count', 'color': '#C8B040', 'domain': 'D'},
    'D4_raw': {'label': 'D4 (log LesionCount)',   'formula': 'D4_raw = log(1+LesionCount_max)',
               'desc': 'Log-transformed significant lesion count', 'color': '#A0B840', 'domain': 'D'},
    'D5_raw': {'label': 'D5 (SecondFrac)',        'formula': 'D5_raw = SecondClusterFrac_max',
               'desc': 'Second largest cluster fraction', 'color': '#70B840', 'domain': 'D'},
    'B_raw':  {'label': 'B (Boundary)',           'formula': 'B_raw = log(ShapeIndex_wavg)',
               'desc': 'Log shape index: boundary irregularity', 'color': '#F5A623', 'domain': 'B'},
    'S1_raw': {'label': 'S1 (Interface)',         'formula': 'S1_raw = Interface_wavg',
               'desc': 'SCC-ADC contact boundary fraction', 'color': '#4CAF50', 'domain': 'S'},
    'S2_raw': {'label': 'S2 (FrontSCC)',          'formula': 'S2_raw = FrontSCC_wavg',
               'desc': 'SCC at invasion front fraction', 'color': '#26A69A', 'domain': 'S'},
    'S3_raw': {'label': 'S3 (−log FrontDist)',    'formula': 'S3_raw = −log(1+FrontDist_um_wavg)',
               'desc': 'Negative log front distance (proximity)', 'color': '#29B6F6', 'domain': 'S'},
    'V_raw':  {'label': 'V (Scale/Consistency)',  'formula': 'V_raw = logit(Tumor_tissue_wavg)',
               'desc': 'Tumor tissue fraction (logit)', 'color': '#7E57C2', 'domain': 'V'},
}

DOMAIN_COLORS = {'C': '#E05C5C', 'D': '#E08C3C', 'B': '#F5A623', 'S': '#4CAF50', 'V': '#7E57C2'}
BG_COLOR = '#0D0D1A'
PANEL_COLOR = '#12122A'


# ============================================================================
# 工具函数
# ============================================================================

def setup_dark_axes(ax, bg=PANEL_COLOR):
    ax.set_facecolor(bg)
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')


def plot_distribution(ax, values, color, feature_key, feat_info, df):
    """直方图 + KDE + 均值线，按train/test分组"""
    from scipy.stats import gaussian_kde

    train_vals = df.loc[df['dataset'] == 'train', feature_key].dropna().values
    test_vals  = df.loc[df['dataset'] == 'test',  feature_key].dropna().values

    # 如果是二值变量，用条形图
    unique_vals = np.unique(values)
    if len(unique_vals) <= 2:
        counts_train = [np.sum(train_vals == v) for v in unique_vals]
        counts_test  = [np.sum(test_vals  == v) for v in unique_vals]
        x = np.arange(len(unique_vals))
        w = 0.35
        ax.bar(x - w/2, counts_train, w, color=color,       alpha=0.85, label='Train')
        ax.bar(x + w/2, counts_test,  w, color='#88CCFF',   alpha=0.85, label='Test')
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in unique_vals], color='white')
        ax.set_ylabel('Count', color='white', fontsize=8)
        ax.legend(fontsize=7, facecolor='#1A1A3A', labelcolor='white', framealpha=0.7)
        setup_dark_axes(ax)
        return

    # 连续变量
    bins = min(30, max(10, len(values) // 8))
    ax.hist(train_vals, bins=bins, color=color,     alpha=0.6, density=True, label='Train')
    ax.hist(test_vals,  bins=bins, color='#88CCFF', alpha=0.5, density=True, label='Test')

    # KDE
    for vals, c in [(train_vals, color), (test_vals, '#88CCFF')]:
        if len(vals) > 3:
            try:
                kde = gaussian_kde(vals, bw_method=0.3)
                xs = np.linspace(vals.min(), vals.max(), 200)
                ax.plot(xs, kde(xs), color=c, linewidth=1.5, alpha=0.9)
            except Exception:
                pass

    # 均值线
    mean_val = np.mean(values)
    ax.axvline(mean_val, color='#FFD700', linewidth=1.2, linestyle='--', alpha=0.9)
    ax.text(mean_val, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.1,
            f'μ={mean_val:.3f}', color='#FFD700', fontsize=7, ha='center', va='bottom')

    ax.set_ylabel('Density', color='white', fontsize=8)
    ax.legend(fontsize=7, facecolor='#1A1A3A', labelcolor='white', framealpha=0.7)
    setup_dark_axes(ax)


def plot_survival_scatter(ax, df, feature_key, color):
    """OS月份 vs 特征值散点图，按死亡状态着色"""
    alive = df[df['death'] == 0]
    dead  = df[df['death'] == 1]

    ax.scatter(alive[feature_key], alive['OS_months'], c='#4FC3F7', s=18, alpha=0.7,
               label='Alive', zorder=3)
    ax.scatter(dead[feature_key],  dead['OS_months'],  c='#EF5350', s=18, alpha=0.7,
               label='Dead',  zorder=3)

    # 趋势线
    try:
        from numpy.polynomial.polynomial import polyfit
        x_all = df[feature_key].dropna().values
        y_all = df.loc[df[feature_key].notna(), 'OS_months'].values
        if len(x_all) > 5:
            c0, c1 = polyfit(x_all, y_all, 1)
            xs = np.linspace(x_all.min(), x_all.max(), 100)
            ax.plot(xs, c0 + c1 * xs, color='#FFD700', linewidth=1.2, alpha=0.8, linestyle='--')
    except Exception:
        pass

    ax.set_xlabel(feature_key, color='white', fontsize=8)
    ax.set_ylabel('OS (months)', color='white', fontsize=8)
    ax.legend(fontsize=7, facecolor='#1A1A3A', labelcolor='white', framealpha=0.7)
    setup_dark_axes(ax)


def plot_boxplot_by_death(ax, df, feature_key, color):
    """按死亡状态分组箱线图"""
    alive_vals = df.loc[df['death'] == 0, feature_key].dropna().values
    dead_vals  = df.loc[df['death'] == 1, feature_key].dropna().values

    bp = ax.boxplot([alive_vals, dead_vals],
                    patch_artist=True,
                    widths=0.5,
                    medianprops=dict(color='#FFD700', linewidth=2),
                    whiskerprops=dict(color='#AAAACC'),
                    capprops=dict(color='#AAAACC'),
                    flierprops=dict(marker='o', color=color, alpha=0.4, markersize=3))

    bp['boxes'][0].set_facecolor('#4FC3F7')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#EF5350')
    bp['boxes'][1].set_alpha(0.7)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Alive', 'Dead'], color='white', fontsize=8)
    ax.set_ylabel(feature_key, color='white', fontsize=8)

    # p值（Mann-Whitney）
    try:
        from scipy.stats import mannwhitneyu
        if len(alive_vals) > 1 and len(dead_vals) > 1:
            _, p = mannwhitneyu(alive_vals, dead_vals, alternative='two-sided')
            ax.set_title(f'p={p:.3f}', color='#FFD700', fontsize=8)
    except Exception:
        pass

    setup_dark_axes(ax)


# ============================================================================
# 单特征可视化主函数
# ============================================================================

def visualize_feature(df, feature_key, feat_info, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    values = df[feature_key].dropna().values
    domain = feat_info['domain']
    color  = feat_info['color']
    domain_color = DOMAIN_COLORS[domain]

    fig = plt.figure(figsize=(14, 9), facecolor=BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    # 标题区
    fig.text(0.5, 0.96, 'SPPI: Histologic Feature Domains on H&E Whole-Slide Images',
             ha='center', va='top', fontsize=13, color='white', fontweight='bold')
    fig.text(0.5, 0.925, '5 Domains · 11 Patient-Level Descriptors · DeepLab-v3+ Segmentation',
             ha='center', va='top', fontsize=8, color='#AAAACC')

    gs = GridSpec(2, 3, figure=fig, top=0.88, bottom=0.08,
                  left=0.07, right=0.97, hspace=0.45, wspace=0.35)

    # ── 面板1: 域标识 + 特征信息 ──────────────────────────────────────────
    ax_info = fig.add_subplot(gs[0, 0])
    ax_info.set_facecolor(PANEL_COLOR)
    for spine in ax_info.spines.values():
        spine.set_edgecolor(domain_color)
        spine.set_linewidth(2)
    ax_info.set_xticks([])
    ax_info.set_yticks([])

    # 域字母大图标
    ax_info.text(0.12, 0.88, domain, transform=ax_info.transAxes,
                 fontsize=32, color=domain_color, fontweight='bold', va='top')
    ax_info.text(0.32, 0.88, feat_info['label'], transform=ax_info.transAxes,
                 fontsize=11, color=domain_color, fontweight='bold', va='top')

    # 公式
    ax_info.text(0.08, 0.62, feat_info['formula'], transform=ax_info.transAxes,
                 fontsize=9, color='#DDDDFF', va='top', family='monospace')
    ax_info.text(0.08, 0.48, feat_info['desc'], transform=ax_info.transAxes,
                 fontsize=8, color='#AAAACC', va='top', wrap=True)

    # 统计摘要
    stats_text = (f"n = {len(values)}\n"
                  f"mean = {np.mean(values):.4f}\n"
                  f"median = {np.median(values):.4f}\n"
                  f"std = {np.std(values):.4f}\n"
                  f"[{np.min(values):.3f}, {np.max(values):.3f}]")
    ax_info.text(0.08, 0.30, stats_text, transform=ax_info.transAxes,
                 fontsize=8, color='#CCCCEE', va='top', family='monospace',
                 linespacing=1.6)

    # train/test 样本数
    n_train = (df['dataset'] == 'train').sum()
    n_test  = (df['dataset'] == 'test').sum()
    ax_info.text(0.08, 0.05, f'Train: {n_train}  |  Test: {n_test}',
                 transform=ax_info.transAxes, fontsize=8, color='#88CCFF', va='bottom')

    # ── 面板2: 分布直方图 ─────────────────────────────────────────────────
    ax_dist = fig.add_subplot(gs[0, 1])
    plot_distribution(ax_dist, values, color, feature_key, feat_info, df)
    ax_dist.set_title(f'Distribution  ({feature_key})', color='white', fontsize=9, pad=4)
    setup_dark_axes(ax_dist)

    # ── 面板3: 箱线图（Alive vs Dead）────────────────────────────────────
    ax_box = fig.add_subplot(gs[0, 2])
    plot_boxplot_by_death(ax_box, df, feature_key, color)
    ax_box.set_title('Alive vs Dead', color='white', fontsize=9, pad=4)
    setup_dark_axes(ax_box)

    # ── 面板4: OS散点图 ───────────────────────────────────────────────────
    ax_scatter = fig.add_subplot(gs[1, 0:2])
    plot_survival_scatter(ax_scatter, df, feature_key, color)
    ax_scatter.set_title(f'{feature_key}  vs  Overall Survival', color='white', fontsize=9, pad=4)
    setup_dark_axes(ax_scatter)

    # ── 面板5: 所有域特征相关热图（当前特征高亮）────────────────────────
    ax_corr = fig.add_subplot(gs[1, 2])
    feat_cols = [k for k in FEATURES.keys() if k in df.columns]
    corr_vals = [df[feature_key].corr(df[k]) for k in feat_cols]
    colors_bar = [DOMAIN_COLORS[FEATURES[k]['domain']] for k in feat_cols]
    short_labels = [k.replace('_raw', '') for k in feat_cols]

    bars = ax_corr.barh(short_labels, corr_vals, color=colors_bar, alpha=0.75)
    # 高亮当前特征
    cur_idx = feat_cols.index(feature_key) if feature_key in feat_cols else -1
    if cur_idx >= 0:
        bars[cur_idx].set_alpha(1.0)
        bars[cur_idx].set_edgecolor('#FFD700')
        bars[cur_idx].set_linewidth(2)

    ax_corr.axvline(0, color='#AAAACC', linewidth=0.8)
    ax_corr.set_xlabel('Pearson r', color='white', fontsize=8)
    ax_corr.set_title(f'Correlation with {feature_key.replace("_raw","")}',
                      color='white', fontsize=9, pad=4)
    ax_corr.tick_params(axis='y', labelsize=7)
    setup_dark_axes(ax_corr)

    # 保存
    safe_name = feature_key.replace('/', '_')
    out_path = os.path.join(output_dir, f'{safe_name}_distribution.jpg')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f'  Saved: {out_path}')
    return out_path


# ============================================================================
# 汇总总览图（所有11特征）
# ============================================================================

def visualize_overview(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    feat_cols = [k for k in FEATURES.keys() if k in df.columns]
    n = len(feat_cols)

    fig, axes = plt.subplots(3, 4, figsize=(20, 14), facecolor=BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle('SPPI: All Feature Distributions  (n=170)',
                 color='white', fontsize=14, fontweight='bold', y=0.98)

    axes_flat = axes.flatten()

    for i, fk in enumerate(feat_cols):
        ax = axes_flat[i]
        ax.set_facecolor(PANEL_COLOR)
        info = FEATURES[fk]
        color = info['color']
        domain_color = DOMAIN_COLORS[info['domain']]

        vals = df[fk].dropna().values
        unique_vals = np.unique(vals)

        if len(unique_vals) <= 2:
            counts = [np.sum(vals == v) for v in unique_vals]
            ax.bar([str(int(v)) for v in unique_vals], counts, color=color, alpha=0.85)
        else:
            bins = min(25, max(8, len(vals) // 8))
            ax.hist(vals, bins=bins, color=color, alpha=0.75, density=True)
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(vals, bw_method=0.3)
                xs = np.linspace(vals.min(), vals.max(), 200)
                ax.plot(xs, kde(xs), color='white', linewidth=1.2, alpha=0.8)
            except Exception:
                pass
            ax.axvline(np.mean(vals), color='#FFD700', linewidth=1, linestyle='--')

        # 域标签
        ax.text(0.04, 0.95, info['domain'], transform=ax.transAxes,
                fontsize=14, color=domain_color, fontweight='bold', va='top')
        ax.set_title(fk.replace('_raw', ''), color='white', fontsize=9, pad=3)
        ax.text(0.98, 0.95, f'μ={np.mean(vals):.3f}', transform=ax.transAxes,
                fontsize=7, color='#FFD700', va='top', ha='right')

        for spine in ax.spines.values():
            spine.set_edgecolor(domain_color)
        ax.tick_params(colors='white', labelsize=7)

    # 隐藏多余子图
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, '00_SPPI_all_features_overview.jpg')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f'  Saved overview: {out_path}')
    return out_path


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    print('Loading data...')
    df = pd.read_excel(EXCEL_PATH)
    print(f'  Loaded {len(df)} rows, columns: {df.columns.tolist()}')

    # 确保dataset列存在
    if 'dataset' not in df.columns:
        df['dataset'] = 'train'

    print('\nGenerating overview...')
    visualize_overview(df, OUTPUT_ROOT)

    print('\nGenerating per-feature visualizations...')
    for fk, finfo in FEATURES.items():
        if fk not in df.columns:
            print(f'  Skipping {fk} (not in data)')
            continue
        print(f'  Processing {fk}...')
        visualize_feature(df, fk, finfo, OUTPUT_ROOT)

    print(f'\nDone. Output saved to: {OUTPUT_ROOT}')
