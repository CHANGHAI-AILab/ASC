#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5个特征独立可视化工具

从cancer_mask_analyzer_complete.py提取并可视化5个特征：
1. 腺癌比例
2. 鳞癌比例
3. 鳞腺比例
4. SCC与ADC接触边界比例
5. SCC与ADC接触边界比例(对称)

每个特征保存在独立文件夹，格式为JPG

关键逻辑说明：
- 与cancer_mask_analyzer_complete.py保持完全一致
- 创建融合mask时，SCC会覆盖ADC的重叠区域
- fusion_mask: 0=背景, 1=ADC(不含被SCC覆盖的), 2=SCC
- 腺癌比例 = fusion_mask中值为1的像素数 / 总像素数
- 鳞癌比例 = fusion_mask中值为2的像素数 / 总像素数
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
OUTPUT_ROOT = "5_features_visualization"

# ============================================================================
# 5特征可视化类
# ============================================================================

class FiveFeatureVisualizer:
    def __init__(self, xla_path, yxa_path, output_root):
        self.xla_path = xla_path
        self.yxa_path = yxa_path
        self.output_root = output_root
        
        # 创建5个特征文件夹
        self.feature_folders = {
            'f1_adc_ratio': '01_腺癌比例',
            'f2_scc_ratio': '02_鳞癌比例',
            'f3_scc_adc_ratio': '03_鳞腺比例',
            'f4_interface_ratio': '04_SCC与ADC接触边界比例',
            'f5_interface_ratio_symmetric': '05_SCC与ADC接触边界比例_对称'
        }
        
        for folder_name in self.feature_folders.values():
            os.makedirs(os.path.join(output_root, folder_name), exist_ok=True)
    
    def load_masks(self, sample_name):
        """
        加载原始mask文件并创建融合mask（与cancer_mask_analyzer_complete.py一致）
        
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
        
        # 创建融合mask（与原始代码完全一致）
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
    
    def calculate_features(self, scc_mask, adc_mask, fusion_mask, dilation_radius=1):
        """
        计算5个特征值（与cancer_mask_analyzer_complete.py完全一致）
        
        Args:
            scc_mask: 鳞癌二值mask（从fusion_mask提取，值为2的区域）
            adc_mask: 腺癌二值mask（从fusion_mask提取，值为1的区域，不包括被SCC覆盖的）
            fusion_mask: 融合mask（1=ADC, 2=SCC）
            dilation_radius: 膨胀半径，用于判断接触
        """
        features = {}
        
        total_pixels = fusion_mask.size
        
        # 从融合mask中统计像素数（与原始代码一致）
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
        
        return features
    
    # ========================================================================
    # 特征1: 腺癌比例
    # ========================================================================
    def visualize_f1_adc_ratio(self, sample_name, adc_mask, features):
        """可视化腺癌比例"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - 腺癌(ADC)比例', fontsize=14, fontweight='bold')
        
        # 左图: 腺癌区域分布
        axes[0].imshow(adc_mask, cmap='Reds')
        axes[0].set_title('腺癌区域分布', fontsize=12)
        axes[0].axis('off')
        
        # 右图: 比例柱状图
        axes[1].bar(['腺癌比例'], [features['adc_ratio']], color='red', alpha=0.7, width=0.5)
        axes[1].set_ylim(0, max(0.1, features['adc_ratio'] * 1.2))
        axes[1].set_ylabel('比例', fontsize=11)
        axes[1].set_title(f"比例值: {features['adc_ratio']:.6f}", fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f1_adc_ratio'], 
                                   f'{sample_name}_腺癌比例_{features["adc_ratio"]:.6f}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  特征1 - 腺癌比例: {features['adc_ratio']:.6f}")
        return output_file
    
    # ========================================================================
    # 特征2: 鳞癌比例
    # ========================================================================
    def visualize_f2_scc_ratio(self, sample_name, scc_mask, features):
        """可视化鳞癌比例"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - 鳞癌(SCC)比例', fontsize=14, fontweight='bold')
        
        # 左图: 鳞癌区域分布
        axes[0].imshow(scc_mask, cmap='Blues')
        axes[0].set_title('鳞癌区域分布', fontsize=12)
        axes[0].axis('off')
        
        # 右图: 比例柱状图
        axes[1].bar(['鳞癌比例'], [features['scc_ratio']], color='blue', alpha=0.7, width=0.5)
        axes[1].set_ylim(0, max(0.1, features['scc_ratio'] * 1.2))
        axes[1].set_ylabel('比例', fontsize=11)
        axes[1].set_title(f"比例值: {features['scc_ratio']:.6f}", fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f2_scc_ratio'], 
                                   f'{sample_name}_鳞癌比例_{features["scc_ratio"]:.6f}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  特征2 - 鳞癌比例: {features['scc_ratio']:.6f}")
        return output_file
    
    # ========================================================================
    # 特征3: 鳞腺比例
    # ========================================================================
    def visualize_f3_scc_adc_ratio(self, sample_name, scc_mask, adc_mask, features):
        """可视化鳞腺比例"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{sample_name} - 鳞腺比例', fontsize=14, fontweight='bold')
        
        # 左图: SCC和ADC叠加显示
        overlay = np.zeros((*scc_mask.shape, 3), dtype=np.uint8)
        overlay[adc_mask > 0] = [255, 0, 0]  # 腺癌: 红色
        overlay[scc_mask > 0] = [0, 0, 255]  # 鳞癌: 蓝色
        axes[0].imshow(overlay)
        axes[0].set_title('SCC(蓝) / ADC(红)', fontsize=12)
        axes[0].axis('off')
        
        # 右图: 比例柱状图
        axes[1].bar(['鳞腺比例'], [features['scc_adc_ratio']], color='purple', alpha=0.7, width=0.5)
        axes[1].set_ylabel('比例', fontsize=11)
        axes[1].set_title(f"比例值: {features['scc_adc_ratio']:.6f}", fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_root, self.feature_folders['f3_scc_adc_ratio'], 
                                   f'{sample_name}_鳞腺比例_{features["scc_adc_ratio"]:.6f}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  特征3 - 鳞腺比例: {features['scc_adc_ratio']:.6f}")
        return output_file
    
    # ========================================================================
    # 特征4: SCC与ADC接触边界比例
    # ========================================================================
    def visualize_f4_interface_ratio(self, sample_name, scc_mask, adc_mask, features, dilation_radius=1):
        """可视化SCC与ADC接触边界比例"""
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
        output_file = os.path.join(self.output_root, self.feature_folders['f4_interface_ratio'], 
                                   f'{sample_name}_接触边界比例_{features["interface_ratio"]:.6f}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  特征4 - SCC与ADC接触边界比例: {features['interface_ratio']:.6f}")
        return output_file
    
    # ========================================================================
    # 特征5: SCC与ADC接触边界比例(对称)
    # ========================================================================
    def visualize_f5_interface_ratio_symmetric(self, sample_name, scc_mask, adc_mask, features, dilation_radius=1):
        """可视化SCC与ADC接触边界比例(对称定义)"""
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
        output_file = os.path.join(self.output_root, self.feature_folders['f5_interface_ratio_symmetric'], 
                                   f'{sample_name}_对称接触边界比例_{features["interface_ratio_symmetric"]:.6f}.jpg')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  特征5 - SCC与ADC接触边界比例(对称): {features['interface_ratio_symmetric']:.6f}")
        return output_file
    
    # ========================================================================
    # 处理单个样本
    # ========================================================================
    def process_sample(self, sample_name):
        """处理单个样本，生成所有5个特征的可视化"""
        print(f"\n处理样本: {sample_name}")
        
        # 加载mask（包括融合mask）
        scc_mask, adc_mask, fusion_mask = self.load_masks(sample_name)
        if scc_mask is None or adc_mask is None or fusion_mask is None:
            print(f"  跳过: 无法加载mask文件")
            return False
        
        # 计算特征（传入融合mask）
        features = self.calculate_features(scc_mask, adc_mask, fusion_mask, dilation_radius=1)
        
        # 生成可视化
        try:
            self.visualize_f1_adc_ratio(sample_name, adc_mask, features)
            self.visualize_f2_scc_ratio(sample_name, scc_mask, features)
            self.visualize_f3_scc_adc_ratio(sample_name, scc_mask, adc_mask, features)
            self.visualize_f4_interface_ratio(sample_name, scc_mask, adc_mask, features, dilation_radius=1)
            self.visualize_f5_interface_ratio_symmetric(sample_name, scc_mask, adc_mask, features, dilation_radius=1)
            
            print(f"  ✓ 完成")
            return True
        except Exception as e:
            print(f"  ✗ 错误: {str(e)}")
            return False
    
    # ========================================================================
    # 批量处理
    # ========================================================================
    def process_all_samples(self):
        """处理所有样本"""
        # 获取共同样本
        if not os.path.exists(self.xla_path) or not os.path.exists(self.yxa_path):
            print("错误: 输入路径不存在")
            return
        
        xla_files = set([os.path.splitext(f)[0] for f in os.listdir(self.xla_path) 
                         if f.endswith('.jpg')])
        yxa_files = set([os.path.splitext(f)[0] for f in os.listdir(self.yxa_path) 
                         if f.endswith('.jpg')])
        
        common_samples = list(xla_files.intersection(yxa_files))
        print(f"找到 {len(common_samples)} 个共同样本")
        
        if not common_samples:
            print("没有找到共同样本")
            return
        
        # 处理所有样本
        success_count = 0
        for i, sample_name in enumerate(common_samples, 1):
            print(f"\n[{i}/{len(common_samples)}]", end=" ")
            if self.process_sample(sample_name):
                success_count += 1
        
        print(f"\n\n{'='*60}")
        print(f"处理完成!")
        print(f"  总样本数: {len(common_samples)}")
        print(f"  成功处理: {success_count}")
        print(f"  失败样本: {len(common_samples) - success_count}")
        print(f"  输出目录: {self.output_root}")
        print(f"{'='*60}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("="*60)
    print("5个特征独立可视化工具")
    print("="*60)
    
    # 创建可视化器
    visualizer = FiveFeatureVisualizer(XLA_PATH, YXA_PATH, OUTPUT_ROOT)
    
    # 处理所有样本
    visualizer.process_all_samples()


if __name__ == "__main__":
    main()
