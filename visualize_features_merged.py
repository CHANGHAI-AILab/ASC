#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
癌症特征合并可视化工具

合并自两个脚本：
1. visualize_5_features.py - 特征1-5（使用融合mask确保与表格数据一致）
2. visualize_features_individual_no1112_nobili.py - 特征6-17

关键逻辑：
- 使用融合mask：SCC覆盖ADC的重叠区域
- fusion_mask: 0=背景, 1=ADC(不含被SCC覆盖的), 2=SCC
- 特征1-5从融合mask计算，确保与表格数据一致
- 特征6-17保持原有计算逻辑

使用方法：
python visualize_features_merged.py
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 配置路径
# ============================================================================

# 腺鳞癌分割结果文件夹路径
XLA_PATH = r"D:\JMC\xla_total"

# 胰腺癌分割结果文件夹路径
YXA_PATH = r"D:\JMC\yxa_total"

# 输出根目录
OUTPUT_ROOT = "feature_visualizations_merged"

# ============================================================================
# 合并特征可视化类
# ============================================================================

class MergedFeatureVisualizer:
    def __init__(self, xla_path, yxa_path, output_root):
        self.xla_path = xla_path
        self.yxa_path = yxa_path
        self.output_root = output_root
        
        # 创建17个特征文件夹（包含特征11-12）
        self.feature_folders = {
            'f01_adc_ratio': '01_腺癌比例',
            'f02_scc_ratio': '02_鳞癌比例',
            'f03_scc_adc_ratio': '03_鳞腺比例',
            'f04_interface_ratio': '04_SCC与ADC接触边界比例',
            'f05_interface_ratio_symmetric': '05_SCC与ADC接触边界比例_对称',
            'f06_dcr': '06_SCC最大连通域占比DCR',
            'f07_fragmentation': '07_SCC碎片化指数',
            'f08_shape_factor': '08_SCC形状因子',
            'f09_perimeter_area_ratio': '09_SCC周长面积比',
            'f10_num_components': '10_SCC连通域数量',
            'f11_largest_area': '11_SCC最大连通域面积',
            'f12_total_area': '12_SCC总面积',
            'f13_near_front': '13_SCC靠近浸润前沿比例',
            'f14_distance_to_front': '14_SCC到前沿平均距离',
            'f15_multifocality': '15_SCC多灶性',
            'f16_num_foci': '16_SCC显著病灶数量',
            'f17_second_largest': '17_SCC第二大连通域占比'
        }
        
        for folder_name in self.feature_folders.values():
            os.makedirs(os.path.join(output_root, folder_name), exist_ok=True)
    
    def load_masks_with_fusion(self, sample_name):
        """
        加载原始mask文件并创建融合mask（与表格数据一致）
        
        Returns:
            tuple: (scc_mask, adc_mask, fusion_mask) 或 (None, None, None)
        """
        xla_file = os.path.join(self.xla_path, sample_name + '.jpg')
        yxa_file = os.path.join(self.yxa_path, sample_name + '.jpg')
        
        if not os.path.exists(xla_file) or not os.path.exists(yxa_file):
            return None, None, None
        
        # 读取图像
        yxa_img = cv2.imread(yxa_file, cv2.IMREAD_GRAYSCALE)
        xla_img = cv2.imread(xla_file, cv2.IMREAD_GRAYSCALE)
        
        if yxa_img is None or xla_img is None:
            return None, None, None
        
        # 转换为二值数据（0-255范围）
        yxa_data = (yxa_img > 0).astype(np.uint8) * 255
        xla_data = (xla_img > 0).astype(np.uint8) * 255
        
        # 创建融合mask（SCC覆盖ADC的重叠区域）
        fusion_mask = np.zeros_like(yxa_data, dtype=np.uint8)
        
        # 胰腺癌区域标记为1
        yxa_binary_mask = (yxa_data > 0).astype(np.uint8)
        fusion_mask[yxa_binary_mask == 1] = 1
        
        # 腺鳞癌区域标记为2（会覆盖胰腺癌）
        xla_binary_mask = (xla_data > 0).astype(np.uint8)
        fusion_mask[xla_binary_mask == 1] = 2
        
        # 从融合mask中提取SCC和ADC
        scc_mask = (fusion_mask == 2).astype(np.uint8)  # SCC区域
        adc_mask = (fusion_mask == 1).astype(np.uint8)  # ADC区域（不包括被SCC覆盖的部分）
        
        return scc_mask, adc_mask, fusion_mask
    
    def calculate_all_features(self, scc_mask, adc_mask, fusion_mask, dilation_radius=1):
        """
        计算所有特征值
        
        特征1-5: 从融合mask计算（与表格数据一致）
        特征6-17: 从SCC mask计算
        """
        features = {}
        
        total_pixels = fusion_mask.size
        
        # ====================================================================
        # 特征1-5: 从融合mask计算（与visualize_5_features.py一致）
        # ====================================================================
        
        # 从融合mask中统计像素数
        pancreas_pixels = np.sum(fusion_mask == 1)  # ADC像素（不包括被SCC覆盖的）
        squamous_pixels = np.sum(fusion_mask == 2)  # SCC像素
        
        # 特征1: 腺癌比例
        features['adc_ratio'] = pancreas_pixels / total_pixels if total_pixels > 0 else 0
        
        # 特征2: 鳞癌比例
        features['scc_ratio'] = squamous_pixels / total_pixels if total_pixels > 0 else 0
        
        # 特征3: 鳞腺比例
        features['scc_adc_ratio'] = features['scc_ratio'] / features['adc_ratio'] if features['adc_ratio'] > 0 else 0
        
        # 特征4和5: 接触边界比例
        if squamous_pixels == 0:
            features['interface_ratio'] = 0.0
            features['interface_ratio_symmetric'] = 0.0
        else:
            # 提取SCC边界
            scc_boundary = cv2.Canny(scc_mask * 255, 50, 150)
            scc_boundary = (scc_boundary > 0).astype(np.uint8)
            
            # 对ADC mask进行膨胀
            if dilation_radius > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                  (2*dilation_radius+1, 2*dilation_radius+1))
                adc_dilated = cv2.dilate(adc_mask, kernel, iterations=1)
            else:
                adc_dilated = adc_mask
            
            # 特征4: SCC与ADC接触边界比例
            scc_boundary_pixels = np.sum(scc_boundary > 0)
            if scc_boundary_pixels > 0:
                interface_pixels = np.sum((scc_boundary > 0) & (adc_dilated > 0))
                features['interface_ratio'] = interface_pixels / scc_boundary_pixels
            else:
                features['interface_ratio'] = 0.0
            
            # 特征5: 对称定义的接触边界比例
            adc_boundary = cv2.Canny(adc_mask * 255, 50, 150)
            adc_boundary = (adc_boundary > 0).astype(np.uint8)
            
            if dilation_radius > 0:
                scc_dilated = cv2.dilate(scc_mask, kernel, iterations=1)
            else:
                scc_dilated = scc_mask
            
            adc_boundary_pixels = np.sum(adc_boundary > 0)
            total_boundary_pixels = scc_boundary_pixels + adc_boundary_pixels
            
            if total_boundary_pixels > 0:
                scc_interface = np.sum((scc_boundary > 0) & (adc_dilated > 0))
                adc_interface = np.sum((adc_boundary > 0) & (scc_dilated > 0))
                features['interface_ratio_symmetric'] = (scc_interface + adc_interface) / total_boundary_pixels
            else:
                features['interface_ratio_symmetric'] = 0.0
        
        # ====================================================================
        # 特征6-17: 从SCC mask计算（与visualize_features_individual_no1112_nobili.py一致）
        # ====================================================================
        
        scc_pixels = np.sum(scc_mask)
        
        if scc_pixels == 0:
            features.update({
                'dcr': 0, 'fragmentation': 0, 'shape_factor': 0, 'perimeter_area_ratio': 0,
                'num_components': 0, 'largest_area': 0, 'total_area': 0,
                'near_front': 0, 'distance_to_front': 0, 'multifocality': 'unifocal', 'num_foci': 0, 'second_largest': 0
            })
            return features
        
        # 形态特征（特征6-12）
        labeled_mask, num_components = ndimage.label(scc_mask, structure=np.ones((3, 3)))
        component_areas = [np.sum(labeled_mask == i) for i in range(1, num_components + 1) if np.sum(labeled_mask == i) >= 10]
        
        features['total_area'] = scc_pixels
        features['num_components'] = len(component_areas)
        
        if len(component_areas) > 0:
            features['largest_area'] = max(component_areas)
            features['dcr'] = features['largest_area'] / features['total_area']
            features['fragmentation'] = 1.0 - features['dcr']
            
            component_areas_sorted = sorted(component_areas, reverse=True)
            features['second_largest'] = component_areas_sorted[1] / features['total_area'] if len(component_areas_sorted) >= 2 else 0
        else:
            features['largest_area'] = 0
            features['dcr'] = 0
            features['fragmentation'] = 0
            features['second_largest'] = 0
        
        # 形状特征
        kernel = np.ones((3, 3), np.uint8)
        smoothed_mask = cv2.morphologyEx(scc_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if contours:
            total_perimeter = sum([cv2.arcLength(contour, True) for contour in contours])
            features['shape_factor'] = (total_perimeter ** 2) / (4 * np.pi * features['total_area']) if features['total_area'] > 0 else 0
            features['perimeter_area_ratio'] = total_perimeter / np.sqrt(features['total_area']) if features['total_area'] > 0 else 0
        else:
            features['shape_factor'] = 0
            features['perimeter_area_ratio'] = 0
        
        # 浸润前沿特征（特征13-14）
        tumor_mask = ((scc_mask > 0) | (adc_mask > 0)).astype(np.uint8)
        if np.sum(tumor_mask) > 0:
            distance_map = distance_transform_edt(tumor_mask)
            front_mask = (distance_map > 0) & (distance_map <= 500)
            
            scc_in_front = np.sum((scc_mask > 0) & front_mask)
            features['near_front'] = scc_in_front / scc_pixels if scc_pixels > 0 else 0
            
            scc_distances = distance_map[scc_mask > 0]
            features['distance_to_front'] = np.mean(scc_distances) if len(scc_distances) > 0 else 0
        else:
            features['near_front'] = 0
            features['distance_to_front'] = 0
        
        # 多灶性特征（特征15-17）
        significant_foci = [area for area in component_areas if area >= 0.05 * features['total_area']]
        features['num_foci'] = len(significant_foci)
        features['multifocality'] = 'multifocal' if features['num_foci'] >= 2 or features['second_largest'] >= 0.05 else 'unifocal'
        
        return features
    
    # ========================================================================
    # 特征1: 腺癌比例
    # ========================================================================
    def visualize_f01_adc_ratio(self, sample_name, adc_mask, features):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - 腺癌(ADC)比例', fontsize=14, fontweight='bold')
        
        axes[0].imshow(adc_mask, cmap='Reds')
        axes[0].set_title(f'腺癌区域分布', fontsize=12)
        axes[0].axis('off')
        
        axes[1].bar(['腺癌比例'], [features['adc_ratio']], color='red', alpha=0.7, width=0.5)
        axes[1].set_ylim(0, max(0.1, features['adc_ratio'] * 1.2))
        axes[1].set_ylabel('比例', fontsize=11)
        axes[1].set_title(f"比例值: {features['adc_ratio']:.6f}", fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f01_adc_ratio'], f'{sample_name}_ADC{features["adc_ratio"]:.6f}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        return output_file
    
    # ========================================================================
    # 特征2: 鳞癌比例
    # ========================================================================
    def visualize_f02_scc_ratio(self, sample_name, scc_mask, features):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - 鳞癌(SCC)比例', fontsize=14, fontweight='bold')
        
        axes[0].imshow(scc_mask, cmap='Blues')
        axes[0].set_title(f'鳞癌区域分布', fontsize=12)
        axes[0].axis('off')
        
        axes[1].bar(['鳞癌比例'], [features['scc_ratio']], color='blue', alpha=0.7, width=0.5)
        axes[1].set_ylim(0, max(0.1, features['scc_ratio'] * 1.2))
        axes[1].set_ylabel('比例', fontsize=11)
        axes[1].set_title(f"比例值: {features['scc_ratio']:.6f}", fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f02_scc_ratio'], f'{sample_name}_SCC{features["scc_ratio"]:.6f}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        return output_file
    
    # ========================================================================
    # 特征3: 鳞腺比例
    # ========================================================================
    def visualize_f03_scc_adc_ratio(self, sample_name, scc_mask, adc_mask, features):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - 鳞腺比例', fontsize=14, fontweight='bold')
        
        overlay = np.zeros((*scc_mask.shape, 3), dtype=np.uint8)
        overlay[adc_mask > 0] = [255, 0, 0]
        overlay[scc_mask > 0] = [0, 0, 255]
        axes[0].imshow(overlay)
        axes[0].set_title('SCC(蓝) / ADC(红)', fontsize=12)
        axes[0].axis('off')
        
        axes[1].bar(['鳞腺比例'], [features['scc_adc_ratio']], color='purple', alpha=0.7, width=0.5)
        axes[1].set_ylabel('比例', fontsize=11)
        axes[1].set_title(f"比例值: {features['scc_adc_ratio']:.6f}", fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f03_scc_adc_ratio'], f'{sample_name}_SCCADC{features["scc_adc_ratio"]:.6f}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        return output_file
    
    # ========================================================================
    # 特征4: SCC与ADC接触边界比例
    # ========================================================================
    def visualize_f04_interface_ratio(self, sample_name, scc_mask, adc_mask, features, dilation_radius=1):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{sample_name} - SCC与ADC接触边界比例', fontsize=14, fontweight='bold')
        
        # 提取边界
        scc_boundary = cv2.Canny(scc_mask * 255, 50, 150)
        scc_boundary = (scc_boundary > 0).astype(np.uint8)
        
        # 膨胀ADC
        if dilation_radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (2*dilation_radius+1, 2*dilation_radius+1))
            adc_dilated = cv2.dilate(adc_mask, kernel, iterations=1)
        else:
            adc_dilated = adc_mask
        
        # 计算接触区域
        interface_mask = (scc_boundary > 0) & (adc_dilated > 0)
        
        # 左图: SCC边界
        axes[0].imshow(scc_boundary, cmap='Blues')
        axes[0].set_title('SCC边界', fontsize=12)
        axes[0].axis('off')
        
        # 中图: 接触区域
        overlay = np.zeros((*scc_mask.shape, 3), dtype=np.uint8)
        overlay[adc_mask > 0] = [255, 100, 100]  # ADC: 浅红色
        overlay[scc_boundary > 0] = [100, 100, 255]  # SCC边界: 浅蓝色
        overlay[interface_mask] = [0, 255, 0]  # 接触区域: 绿色
        axes[1].imshow(overlay)
        axes[1].set_title('接触区域(绿色)', fontsize=12)
        axes[1].axis('off')
        
        # 右图: 比例柱状图
        axes[2].bar(['接触边界比例'], [features['interface_ratio']], color='green', alpha=0.7, width=0.5)
        axes[2].set_ylim(0, 1.0)
        axes[2].set_ylabel('比例', fontsize=11)
        axes[2].set_title(f"比例值: {features['interface_ratio']:.6f}", fontsize=12)
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f04_interface_ratio'], 
                                   f'{sample_name}_接触边界比例_{features["interface_ratio"]:.6f}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    # ========================================================================
    # 特征5: SCC与ADC接触边界比例(对称)
    # ========================================================================
    def visualize_f05_interface_ratio_symmetric(self, sample_name, scc_mask, adc_mask, features, dilation_radius=1):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'{sample_name} - SCC与ADC接触边界比例(对称)', fontsize=14, fontweight='bold')
        
        # 提取边界
        scc_boundary = cv2.Canny(scc_mask * 255, 50, 150)
        scc_boundary = (scc_boundary > 0).astype(np.uint8)
        adc_boundary = cv2.Canny(adc_mask * 255, 50, 150)
        adc_boundary = (adc_boundary > 0).astype(np.uint8)
        
        # 膨胀
        if dilation_radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                              (2*dilation_radius+1, 2*dilation_radius+1))
            adc_dilated = cv2.dilate(adc_mask, kernel, iterations=1)
            scc_dilated = cv2.dilate(scc_mask, kernel, iterations=1)
        else:
            adc_dilated = adc_mask
            scc_dilated = scc_mask
        
        # 计算接触区域
        scc_interface = (scc_boundary > 0) & (adc_dilated > 0)
        adc_interface = (adc_boundary > 0) & (scc_dilated > 0)
        
        # 左上: SCC边界与ADC接触
        overlay1 = np.zeros((*scc_mask.shape, 3), dtype=np.uint8)
        overlay1[adc_mask > 0] = [255, 100, 100]
        overlay1[scc_boundary > 0] = [100, 100, 255]
        overlay1[scc_interface] = [0, 255, 0]
        axes[0, 0].imshow(overlay1)
        axes[0, 0].set_title('SCC边界与ADC接触(绿色)', fontsize=11)
        axes[0, 0].axis('off')
        
        # 右上: ADC边界与SCC接触
        overlay2 = np.zeros((*scc_mask.shape, 3), dtype=np.uint8)
        overlay2[scc_mask > 0] = [100, 100, 255]
        overlay2[adc_boundary > 0] = [255, 100, 100]
        overlay2[adc_interface] = [255, 255, 0]
        axes[0, 1].imshow(overlay2)
        axes[0, 1].set_title('ADC边界与SCC接触(黄色)', fontsize=11)
        axes[0, 1].axis('off')
        
        # 左下: 双向接触叠加
        overlay3 = np.zeros((*scc_mask.shape, 3), dtype=np.uint8)
        overlay3[adc_mask > 0] = [255, 100, 100]
        overlay3[scc_mask > 0] = [100, 100, 255]
        overlay3[scc_interface | adc_interface] = [255, 0, 255]  # 紫色
        axes[1, 0].imshow(overlay3)
        axes[1, 0].set_title('双向接触区域(紫色)', fontsize=11)
        axes[1, 0].axis('off')
        
        # 右下: 比例柱状图
        axes[1, 1].bar(['对称接触边界比例'], [features['interface_ratio_symmetric']], 
                       color='purple', alpha=0.7, width=0.5)
        axes[1, 1].set_ylim(0, 1.0)
        axes[1, 1].set_ylabel('比例', fontsize=11)
        axes[1, 1].set_title(f"比例值: {features['interface_ratio_symmetric']:.6f}", fontsize=12)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f05_interface_ratio_symmetric'], 
                                   f'{sample_name}_对称接触边界比例_{features["interface_ratio_symmetric"]:.6f}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    # ========================================================================
    # 特征6: SCC最大连通域占比(DCR)
    # ========================================================================
    def visualize_f06_dcr(self, sample_name, scc_mask, features):
        if features['dcr'] == 0:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - SCC最大连通域占比(DCR)', fontsize=14, fontweight='bold')
        
        labeled_mask, num_components = ndimage.label(scc_mask, structure=np.ones((3, 3)))
        largest_idx = np.argmax([np.sum(labeled_mask == i) for i in range(1, num_components + 1)]) + 1
        largest_component = (labeled_mask == largest_idx).astype(np.uint8)
        
        axes[0].imshow(largest_component, cmap='Reds')
        axes[0].set_title('最大连通域', fontsize=12)
        axes[0].axis('off')
        
        axes[1].barh(['最大连通域', '其他区域'], [features['dcr'], 1-features['dcr']], 
                     color=['red', 'lightgray'])
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel('占比', fontsize=11)
        axes[1].set_title(f"DCR值: {features['dcr']:.10f}", fontsize=12)
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f06_dcr'], f'{sample_name}_DCR{features["dcr"]:.10f}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        return output_file
    
    # ========================================================================
    # 特征7: SCC碎片化指数
    # ========================================================================
    def visualize_f07_fragmentation(self, sample_name, scc_mask, features):
        if features['fragmentation'] == 0:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - SCC碎片化指数', fontsize=14, fontweight='bold')
        
        labeled_mask, num_components = ndimage.label(scc_mask, structure=np.ones((3, 3)))
        colored_labels = np.zeros((*labeled_mask.shape, 3), dtype=np.uint8)
        for i in range(1, num_components + 1):
            if np.sum(labeled_mask == i) >= 10:
                color = np.random.randint(50, 255, 3)
                colored_labels[labeled_mask == i] = color
        
        axes[0].imshow(colored_labels)
        axes[0].set_title(f'连通域分布 (共{features["num_components"]}个)', fontsize=12)
        axes[0].axis('off')
        
        axes[1].barh(['碎片化', '集中度'], [features['fragmentation'], features['dcr']], 
                     color=['orange', 'green'])
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel('指数', fontsize=11)
        axes[1].set_title(f"碎片化指数: {features['fragmentation']:.6f}", fontsize=12)
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f07_fragmentation'], f'{sample_name}_FRAG{features["fragmentation"]:.6f}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        return output_file
    
    # ========================================================================
    # 特征8: 形状因子
    # ========================================================================
    def visualize_f08_shape_factor(self, sample_name, scc_mask, features):
        if features['shape_factor'] == 0:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - SCC形状因子', fontsize=14, fontweight='bold')
        
        kernel = np.ones((3, 3), np.uint8)
        smoothed = cv2.morphologyEx(scc_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        contour_img = np.zeros((*scc_mask.shape, 3), dtype=np.uint8)
        contour_img[scc_mask > 0] = [100, 100, 255]
        cv2.drawContours(contour_img, contours, -1, (255, 255, 0), 2)
        
        axes[0].imshow(contour_img)
        axes[0].set_title('SCC边界轮廓', fontsize=12)
        axes[0].axis('off')
        
        axes[1].bar(['形状因子'], [features['shape_factor']], color='purple', alpha=0.7, width=0.5)
        axes[1].axhline(y=1, color='red', linestyle='--', linewidth=2, label='圆形=1')
        axes[1].set_ylabel('形状因子', fontsize=11)
        axes[1].set_title(f"形状因子: {features['shape_factor']:.6f}\n(圆形=1, 越大越不规则)", fontsize=11)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f08_shape_factor'], f'{sample_name}_SHAPE{features["shape_factor"]:.6f}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        return output_file
    
    # ========================================================================
    # 特征9: 周长面积比
    # ========================================================================
    def visualize_f09_perimeter_area_ratio(self, sample_name, scc_mask, features):
        if features['perimeter_area_ratio'] == 0:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - SCC周长面积比', fontsize=14, fontweight='bold')
        
        axes[0].imshow(scc_mask, cmap='Blues')
        axes[0].set_title('SCC区域', fontsize=12)
        axes[0].axis('off')
        
        axes[1].bar(['周长面积比'], [features['perimeter_area_ratio']], color='teal', alpha=0.7, width=0.5)
        axes[1].set_ylabel('比值', fontsize=11)
        axes[1].set_title(f"周长面积比: {features['perimeter_area_ratio']:.6f}", fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f09_perimeter_area_ratio'], f'{sample_name}_PERIM{features["perimeter_area_ratio"]:.6f}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        return output_file
    
    # ========================================================================
    # 特征10: 连通域数量
    # ========================================================================
    def visualize_f10_num_components(self, sample_name, scc_mask, features):
        if features['num_components'] == 0:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - SCC连通域数量', fontsize=14, fontweight='bold')
        
        labeled_mask, num_components = ndimage.label(scc_mask, structure=np.ones((3, 3)))
        colored_labels = np.zeros((*labeled_mask.shape, 3), dtype=np.uint8)
        np.random.seed(42)
        for i in range(1, num_components + 1):
            if np.sum(labeled_mask == i) >= 10:
                color = np.random.randint(50, 255, 3)
                colored_labels[labeled_mask == i] = color
        
        axes[0].imshow(colored_labels)
        axes[0].set_title('连通域分布', fontsize=12)
        axes[0].axis('off')
        
        axes[1].bar(['连通域数量'], [features['num_components']], color='steelblue', alpha=0.7, width=0.5)
        axes[1].set_ylabel('数量', fontsize=11)
        axes[1].set_title(f"连通域数量: {features['num_components']}", fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f10_num_components'], f'{sample_name}_COMP{features["num_components"]}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        return output_file
    
    # ========================================================================
    # 特征11: 最大连通域面积
    # ========================================================================
    def visualize_f11_largest_area(self, sample_name, scc_mask, features):
        if features['largest_area'] == 0:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - SCC最大连通域面积', fontsize=14, fontweight='bold')
        
        labeled_mask, num_components = ndimage.label(scc_mask, structure=np.ones((3, 3)))
        valid_indices = [i for i in range(1, num_components + 1) if np.sum(labeled_mask == i) >= 10]
        
        highlight = np.zeros((*labeled_mask.shape, 3), dtype=np.uint8)
        highlight[scc_mask > 0] = [100, 100, 100]
        
        if valid_indices:
            largest_idx = max(valid_indices, key=lambda i: np.sum(labeled_mask == i))
            highlight[labeled_mask == largest_idx] = [255, 0, 0]
        
        axes[0].imshow(highlight)
        axes[0].set_title('最大连通域(红色)', fontsize=12)
        axes[0].axis('off')
        
        axes[1].bar(['最大连通域面积'], [features['largest_area']], color='red', alpha=0.7, width=0.5)
        axes[1].set_ylabel('面积(像素)', fontsize=11)
        axes[1].set_title(f"面积: {features['largest_area']} 像素", fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f11_largest_area'], f'{sample_name}_LARGEST{features["largest_area"]}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        return output_file
    
    # ========================================================================
    # 特征12: SCC总面积
    # ========================================================================
    def visualize_f12_total_area(self, sample_name, scc_mask, features):
        if features['total_area'] == 0:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - SCC总面积', fontsize=14, fontweight='bold')
        
        axes[0].imshow(scc_mask, cmap='Blues')
        axes[0].set_title('SCC总区域', fontsize=12)
        axes[0].axis('off')
        
        axes[1].bar(['SCC总面积'], [features['total_area']], color='blue', alpha=0.7, width=0.5)
        axes[1].set_ylabel('面积(像素)', fontsize=11)
        axes[1].set_title(f"总面积: {features['total_area']} 像素", fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f12_total_area'], f'{sample_name}_TOTAL{features["total_area"]}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        return output_file
    
    # ========================================================================
    # 特征13: SCC靠近浸润前沿比例
    # ========================================================================
    def visualize_f13_near_front(self, sample_name, scc_mask, adc_mask, features):
        if features['near_front'] == 0:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - SCC靠近浸润前沿比例', fontsize=14, fontweight='bold')
        
        tumor_mask = ((scc_mask > 0) | (adc_mask > 0)).astype(np.uint8)
        distance_map = distance_transform_edt(tumor_mask)
        front_mask = (distance_map > 0) & (distance_map <= 500)
        
        front_vis = np.zeros((*scc_mask.shape, 3), dtype=np.uint8)
        front_vis[tumor_mask > 0] = [200, 200, 200]
        front_vis[front_mask] = [255, 165, 0]
        front_vis[scc_mask > 0] = [0, 0, 255]
        front_vis[(scc_mask > 0) & front_mask] = [255, 0, 255]
        
        axes[0].imshow(front_vis)
        axes[0].set_title('SCC在前沿(紫色)', fontsize=12)
        axes[0].axis('off')
        
        axes[1].barh(['前沿区', '内部区'], [features['near_front'], 1-features['near_front']], 
                     color=['purple', 'lightblue'])
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel('比例', fontsize=11)
        axes[1].set_title(f"前沿比例: {features['near_front']:.6f}", fontsize=12)
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f13_near_front'], f'{sample_name}_FRONT{features["near_front"]:.6f}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        return output_file
    
    # ========================================================================
    # 特征14: SCC到前沿平均距离
    # ========================================================================
    def visualize_f14_distance_to_front(self, sample_name, scc_mask, adc_mask, features):
        if features['distance_to_front'] == 0:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - SCC到前沿平均距离', fontsize=14, fontweight='bold')
        
        tumor_mask = ((scc_mask > 0) | (adc_mask > 0)).astype(np.uint8)
        distance_map = distance_transform_edt(tumor_mask)
        
        distance_vis = distance_map.copy()
        distance_vis[tumor_mask == 0] = np.nan
        
        im = axes[0].imshow(distance_vis, cmap='hot', interpolation='nearest')
        axes[0].set_title('距离热图', fontsize=12)
        axes[0].axis('off')
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        
        axes[1].bar(['平均距离'], [features['distance_to_front']], color='orange', alpha=0.7, width=0.5)
        axes[1].set_ylabel('距离(像素)', fontsize=11)
        axes[1].set_title(f"平均距离: {features['distance_to_front']:.2f} 像素", fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f14_distance_to_front'], f'{sample_name}_DIST{features["distance_to_front"]:.2f}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        return output_file
    
    # ========================================================================
    # 特征15: SCC多灶性
    # ========================================================================
    def visualize_f15_multifocality(self, sample_name, scc_mask, features):
        if np.sum(scc_mask) == 0:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - SCC多灶性', fontsize=14, fontweight='bold')
        
        labeled_mask, num_components = ndimage.label(scc_mask, structure=np.ones((3, 3)))
        total_area = np.sum(scc_mask)
        
        colored_foci = np.zeros((*labeled_mask.shape, 3), dtype=np.uint8)
        np.random.seed(42)
        for i in range(1, num_components + 1):
            area = np.sum(labeled_mask == i)
            if area >= 0.05 * total_area:
                color = np.random.randint(100, 255, 3)
            else:
                color = [50, 50, 50]
            colored_foci[labeled_mask == i] = color
        
        axes[0].imshow(colored_foci)
        axes[0].set_title(f'病灶分布 (亮色=显著病灶)', fontsize=12)
        axes[0].axis('off')
        
        multifocal_value = 1 if features['multifocality'] == 'multifocal' else 0
        colors = ['red' if multifocal_value == 1 else 'green']
        axes[1].bar(['多灶性'], [1], color=colors, alpha=0.7, width=0.5)
        axes[1].set_ylim(0, 1.2)
        axes[1].set_ylabel('分类', fontsize=11)
        axes[1].set_title(f"分类: {features['multifocality']}\n显著病灶数: {features['num_foci']}", fontsize=12)
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(['unifocal', 'multifocal'])
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f15_multifocality'], f'{sample_name}_MULTI{features["multifocality"]}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        return output_file
    
    # ========================================================================
    # 特征16: SCC显著病灶数量
    # ========================================================================
    def visualize_f16_num_foci(self, sample_name, scc_mask, features):
        if features['num_foci'] == 0:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - SCC显著病灶数量', fontsize=14, fontweight='bold')
        
        labeled_mask, num_components = ndimage.label(scc_mask, structure=np.ones((3, 3)))
        total_area = np.sum(scc_mask)
        
        significant_vis = np.zeros((*labeled_mask.shape, 3), dtype=np.uint8)
        np.random.seed(42)
        for i in range(1, num_components + 1):
            area = np.sum(labeled_mask == i)
            if area >= 0.05 * total_area:
                color = np.random.randint(100, 255, 3)
                significant_vis[labeled_mask == i] = color
        
        axes[0].imshow(significant_vis)
        axes[0].set_title(f'显著病灶 (≥5%总面积)', fontsize=12)
        axes[0].axis('off')
        
        axes[1].bar(['显著病灶数'], [features['num_foci']], color='darkgreen', alpha=0.7, width=0.5)
        axes[1].set_ylabel('数量', fontsize=11)
        axes[1].set_title(f"显著病灶数: {features['num_foci']}", fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f16_num_foci'], f'{sample_name}_FOCI{features["num_foci"]}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        return output_file
    
    # ========================================================================
    # 特征17: SCC第二大连通域占比
    # ========================================================================
    def visualize_f17_second_largest(self, sample_name, scc_mask, features):
        if features['second_largest'] == 0:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - SCC第二大连通域占比', fontsize=14, fontweight='bold')
        
        labeled_mask, num_components = ndimage.label(scc_mask, structure=np.ones((3, 3)))
        component_areas = [(i, np.sum(labeled_mask == i)) for i in range(1, num_components + 1)]
        component_areas_sorted = sorted(component_areas, key=lambda x: x[1], reverse=True)
        
        top2_vis = np.zeros((*labeled_mask.shape, 3), dtype=np.uint8)
        if len(component_areas_sorted) >= 1:
            top2_vis[labeled_mask == component_areas_sorted[0][0]] = [255, 0, 0]
        if len(component_areas_sorted) >= 2:
            top2_vis[labeled_mask == component_areas_sorted[1][0]] = [0, 0, 255]
        
        axes[0].imshow(top2_vis)
        axes[0].set_title('最大(红) vs 第二大(蓝)', fontsize=12)
        axes[0].axis('off')
        
        axes[1].bar(['第二大占比'], [features['second_largest']], color='blue', alpha=0.7, width=0.5)
        axes[1].axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='显著阈值(5%)')
        axes[1].set_ylabel('占比', fontsize=11)
        axes[1].set_title(f"第二大占比: {features['second_largest']:.6f}", fontsize=12)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f17_second_largest'], f'{sample_name}_SECOND{features["second_largest"]:.6f}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        return output_file
    
    def print_features(self, sample_name, features):
        """打印特征值"""
        print(f"\n{'='*60}")
        print(f"样本: {sample_name} - 特征值汇总")
        print(f"{'='*60}")
        print(f"特征1  - 腺癌比例: {features['adc_ratio']:.6f}")
        print(f"特征2  - 鳞癌比例: {features['scc_ratio']:.6f}")
        print(f"特征3  - 鳞腺比例: {features['scc_adc_ratio']:.6f}")
        print(f"特征4  - SCC与ADC接触边界比例: {features['interface_ratio']:.6f}")
        print(f"特征5  - SCC与ADC接触边界比例(对称): {features['interface_ratio_symmetric']:.6f}")
        print(f"特征6  - SCC最大连通域占比DCR: {features['dcr']:.10f}")
        print(f"特征7  - SCC碎片化指数: {features['fragmentation']:.6f}")
        print(f"特征8  - SCC形状因子: {features['shape_factor']:.6f}")
        print(f"特征9  - SCC周长面积比: {features['perimeter_area_ratio']:.6f}")
        print(f"特征10 - SCC连通域数量: {features['num_components']}")
        print(f"特征11 - SCC最大连通域面积: {features['largest_area']}")
        print(f"特征12 - SCC总面积: {features['total_area']}")
        print(f"特征13 - SCC靠近浸润前沿比例: {features['near_front']:.6f}")
        print(f"特征14 - SCC到前沿平均距离: {features['distance_to_front']:.2f}")
        print(f"特征15 - SCC多灶性: {features['multifocality']}")
        print(f"特征16 - SCC显著病灶数量: {features['num_foci']}")
        print(f"特征17 - SCC第二大连通域占比: {features['second_largest']:.6f}")
        print(f"{'='*60}")
    
    def process_all_samples(self, max_samples=None):
        """处理所有样本"""
        if not os.path.exists(self.xla_path) or not os.path.exists(self.yxa_path):
            print(f"错误: 原始mask目录不存在")
            return
        
        xla_files = [f.replace('.jpg', '') for f in os.listdir(self.xla_path) if f.endswith('.jpg')]
        yxa_files = [f.replace('.jpg', '') for f in os.listdir(self.yxa_path) if f.endswith('.jpg')]
        common_samples = list(set(xla_files) & set(yxa_files))
        
        print(f"找到 {len(common_samples)} 个共同样本")
        
        if max_samples:
            common_samples = common_samples[:max_samples]
            print(f"限制处理前 {max_samples} 个样本")
        
        success_count = 0
        fail_count = 0
        
        # 只处理指定样本
        target_samples = ['2202521A6_predictions', '2314986A5_predictions']
        common_samples = [s for s in common_samples if s in target_samples]
        print(f"只处理指定样本: {target_samples}")
        
        for i, sample_name in enumerate(common_samples, 1):
            print(f"\n[{i}/{len(common_samples)}] 处理样本: {sample_name}")
            
            try:
                # 使用融合mask加载
                scc_mask, adc_mask, fusion_mask = self.load_masks_with_fusion(sample_name)
                
                if scc_mask is None:
                    print(f"  [X] 加载失败")
                    fail_count += 1
                    continue
                
                # 计算所有特征
                features = self.calculate_all_features(scc_mask, adc_mask, fusion_mask)
                
                # 打印特征值
                self.print_features(sample_name, features)
                
                print("  生成特征可视化...")
                
                # 特征1-5
                self.visualize_f01_adc_ratio(sample_name, adc_mask, features)
                self.visualize_f02_scc_ratio(sample_name, scc_mask, features)
                self.visualize_f03_scc_adc_ratio(sample_name, scc_mask, adc_mask, features)
                self.visualize_f04_interface_ratio(sample_name, scc_mask, adc_mask, features)
                self.visualize_f05_interface_ratio_symmetric(sample_name, scc_mask, adc_mask, features)
                
                # 特征6-17
                self.visualize_f06_dcr(sample_name, scc_mask, features)
                self.visualize_f07_fragmentation(sample_name, scc_mask, features)
                self.visualize_f08_shape_factor(sample_name, scc_mask, features)
                self.visualize_f09_perimeter_area_ratio(sample_name, scc_mask, features)
                self.visualize_f10_num_components(sample_name, scc_mask, features)
                self.visualize_f11_largest_area(sample_name, scc_mask, features)
                self.visualize_f12_total_area(sample_name, scc_mask, features)
                self.visualize_f13_near_front(sample_name, scc_mask, adc_mask, features)
                self.visualize_f14_distance_to_front(sample_name, scc_mask, adc_mask, features)
                self.visualize_f15_multifocality(sample_name, scc_mask, features)
                self.visualize_f16_num_foci(sample_name, scc_mask, features)
                self.visualize_f17_second_largest(sample_name, scc_mask, features)
                
                success_count += 1
                print(f"  [OK] 样本处理完成")
                
            except Exception as e:
                print(f"  [X] 处理失败: {str(e)}")
                import traceback
                traceback.print_exc()
                fail_count += 1
        
        print(f"\n{'='*60}")
        print(f"处理完成!")
        print(f"  成功: {success_count} 个样本")
        print(f"  失败: {fail_count} 个样本")
        print(f"  输出目录: {self.output_root}")
        print(f"{'='*60}")


def main():
    """主函数"""
    print("="*60)
    print("癌症特征合并可视化工具")
    print("合并了特征1-5（融合mask）和特征6-17")
    print("="*60)
    
    visualizer = MergedFeatureVisualizer(
        xla_path=XLA_PATH,
        yxa_path=YXA_PATH,
        output_root=OUTPUT_ROOT
    )
    
    visualizer.process_all_samples(max_samples=1000)


if __name__ == "__main__":
    main()
