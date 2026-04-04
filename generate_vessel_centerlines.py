"""
血管中心线提取脚本
从包含10根血管的mask（label 1-10）中提取每根血管的中心线
并将中心线添加到mask中（使用label 11-20表示对应血管的中心线）
"""
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize_3d, dilation, ball
import os

print("=" * 80)
print("血管中心线提取")
print("=" * 80)

# ============================================================
# 1. 读取数据
# ============================================================
print("\n[步骤1] 读取血管mask...")

input_file = r'D:\wechatfile\xwechat_files\CCW452402567_6b46\msg\file\2026-03\5070913.nii.gz'
output_file = r'D:\wechatfile\xwechat_files\CCW452402567_6b46\msg\file\2026-03\5070913_with_centerlines.nii.gz'

# 检查文件是否存在
if not os.path.exists(input_file):
    print(f"错误: 文件不存在 - {input_file}")
    exit(1)

# 读取NIfTI文件
nii_img = nib.load(input_file)
mask_data = np.round(nii_img.get_fdata()).astype(np.int32)
affine = nii_img.affine
header = nii_img.header

print(f"文件路径: {input_file}")
print(f"数据形状: {mask_data.shape}")
print(f"数据类型: {mask_data.dtype}")

unique_labels = np.unique(mask_data)
unique_labels = unique_labels[unique_labels != 0]
print(f"唯一标签值（排除背景）: {unique_labels}")

# ============================================================
# 2. 为每根血管提取中心线
# ============================================================
print("\n[步骤2] 提取每根血管的中心线...")

# 动态确定血管标签和中心线标签
vessel_labels = unique_labels
CENTERLINE_LABEL = int(unique_labels.max()) + 1
print(f"检测到的血管标签: {vessel_labels}")
print(f"中心线将使用标签: {CENTERLINE_LABEL}")

def extract_centerline_3d(binary_mask, bold_radius=1):
    """
    提取3D二值mask的中心线
    使用骨架化算法，然后进行加粗处理
    """
    # 确保是二值mask
    binary_mask = binary_mask.astype(bool)
    
    # 使用3D骨架化算法提取中心线
    skeleton = skeletonize_3d(binary_mask)
    
    # 加粗中心线：使用形态学膨胀
    if bold_radius > 0:
        skeleton = dilation(skeleton, ball(bold_radius))
    
    return skeleton.astype(np.uint8)

def extract_centerline_distance_transform(binary_mask, bold_radius=1):
    """
    使用距离变换方法提取中心线
    找到距离边界最远的点作为中心线，然后进行加粗处理
    """
    # 计算距离变换
    distance = ndimage.distance_transform_edt(binary_mask)
    
    # 找到局部最大值作为中心线
    # 使用形态学操作找到脊线
    local_max = ndimage.maximum_filter(distance, size=3)
    centerline = (distance == local_max) & (distance > 0)
    
    # 细化中心线
    centerline = skeletonize_3d(centerline)
    
    # 加粗中心线：使用形态学膨胀
    if bold_radius > 0:
        centerline = dilation(centerline, ball(bold_radius))
    
    return centerline.astype(np.uint8)

# 处理每根血管：先收集所有中心线，最后统一写入
all_centerlines = np.zeros(mask_data.shape, dtype=bool)

# 设置中心线加粗半径（可以调整这个值来控制加粗程度）
BOLD_RADIUS = 3  # 默认为1，可以增加到2或3获得更粗的中心线
print(f"中心线加粗半径: {BOLD_RADIUS}")

for vessel_label in vessel_labels:
    print(f"\n处理血管 {vessel_label}...")
    
    # 从原始mask提取当前血管
    vessel_mask = (mask_data == vessel_label)
    
    if not vessel_mask.any():
        print(f"  警告: 未找到标签 {vessel_label}，跳过")
        continue
    
    vessel_voxels = np.sum(vessel_mask)
    print(f"  血管体素数: {vessel_voxels}")
    
    try:
        centerline = extract_centerline_3d(vessel_mask, BOLD_RADIUS)
        
        if not centerline.any():
            print(f"  骨架化方法未找到中心线，尝试距离变换方法...")
            centerline = extract_centerline_distance_transform(vessel_mask, BOLD_RADIUS)
        
        # 确保中心线只在该血管内部（防止骨架化溢出）
        centerline = centerline.astype(bool) & vessel_mask
        
        centerline_voxels = np.sum(centerline)
        print(f"  中心线体素数: {centerline_voxels}")
        
        if centerline_voxels > 0:
            all_centerlines |= centerline
            print(f"  成功: 中心线已收集（已加粗）")
        else:
            print(f"  警告: 未能提取到中心线")
            
    except Exception as e:
        print(f"  错误: 提取中心线失败 - {e}")
        continue

# 最后统一写入：在原始mask基础上叠加中心线
# 原始血管体素保持不变，只有中心线所在体素被覆盖为 CENTERLINE_LABEL
output_mask = mask_data.copy()
output_mask[all_centerlines] = CENTERLINE_LABEL
print(f"\n所有中心线已写入，使用标签: {CENTERLINE_LABEL}")

# ============================================================
# 3. 保存结果
# ============================================================
print("\n[步骤3] 保存结果...")

# 保存前检查
print(f"保存前 output_mask unique labels: {np.unique(output_mask)}")

output_nii = nib.Nifti1Image(output_mask.astype(np.int32), affine, header)

# 保存文件
nib.save(output_nii, output_file)
print(f"已保存: {output_file}")

# ============================================================
# 4. 统计信息与验证
# ============================================================
print("\n" + "=" * 80)
print("处理完成！")
print("=" * 80)

print("\n【标签统计】")
output_unique = np.unique(output_mask)
print(f"输出mask中的唯一标签: {output_unique}")

print("\n【详细统计】")
print(f"{'标签':<10} {'类型':<22} {'输入体素数':<15} {'输出体素数':<15} {'差值':<10}")
print("-" * 75)

for label in output_unique:
    if label == 0:
        label_type = "背景"
    elif label < CENTERLINE_LABEL:
        label_type = f"血管 {int(label)}"
    elif label == CENTERLINE_LABEL:
        label_type = "中心线（所有血管）"
    else:
        label_type = "未知"
    
    input_count = int(np.sum(mask_data == label))
    output_count = int(np.sum(output_mask == label))
    diff = output_count - input_count
    print(f"{int(label):<10} {label_type:<22} {input_count:<15} {output_count:<15} {diff:<10}")

# ============================================================
# 验证：检查输入的所有血管label是否都保留在输出中
# ============================================================
print("\n【验证：血管label完整性检查】")
input_vessel_labels = set(vessel_labels.tolist())
output_labels = set(output_unique[output_unique != 0].astype(int).tolist())
output_labels.discard(CENTERLINE_LABEL)  # 排除中心线label

missing_labels = input_vessel_labels - output_labels
if missing_labels:
    print(f"  ⚠️  警告: 以下血管label在输出中完全消失: {sorted(missing_labels)}")
    print(f"  原因: 这些血管体素全部被中心线label {CENTERLINE_LABEL} 覆盖")
    print(f"  建议: 检查这些血管是否过细（体素数极少）")
else:
    print(f"  ✅ 所有血管label均保留在输出中")

print(f"\n  输入血管label数: {len(input_vessel_labels)}")
print(f"  输出血管label数: {len(output_labels)}")
print(f"  中心线label: {CENTERLINE_LABEL}")

print("\n【文件信息】")
print(f"输入文件: {input_file}")
print(f"输出文件: {output_file}")
print(f"数据形状: {output_mask.shape}")
print(f"数据类型: {output_mask.dtype}")

print("\n【标签说明】")
print(f"  Label 1-{int(CENTERLINE_LABEL)-1}: 原始血管mask")
print(f"  Label {CENTERLINE_LABEL}:   所有血管的中心线（统一标签）")

print("\n" + "=" * 80)
