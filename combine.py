import os
import glob
import cv2
import numpy as np

for each in glob.glob(r'D:\JMC\stroma_total\stroma_total\*.jpg'):
	print(each)
	eachfilename=each.split('\\')[-1]

	# 读取三个mask图像
	#mask1 = cv2.imread(r'D:\JMC\stroma_total\stroma_total\\'+eachfilename)
	mask2 = cv2.imread(r'D:\JMC\yxa_total\\'+eachfilename)
	mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
	threshold_value = 127
	_, mask2 = cv2.threshold(mask2, threshold_value, 255, cv2.THRESH_BINARY)


	mask1 = cv2.imread(r'D:\JMC\stroma_total\stroma_total\\'+eachfilename)
	mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
	threshold_value = 127
	_, mask1 = cv2.threshold(mask1, threshold_value, 255, cv2.THRESH_BINARY)


	mask3 = cv2.imread(r'D:\JMC\xla_total\\'+eachfilename)
	mask3 = cv2.cvtColor(mask3, cv2.COLOR_BGR2GRAY)
	threshold_value = 127
	_, mask3 = cv2.threshold(mask3, threshold_value, 255, cv2.THRESH_BINARY)

	combined_mask = np.zeros((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)

	combined_mask[mask1 > 0] = [255, 0, 0]  # 红色
	combined_mask[mask2 > 0] = [0, 255, 0]  # 绿色
	combined_mask[mask3 > 0] = [0, 0, 255]  # 蓝色



	# 保存合成后的mask
	cv2.imwrite(r'D:\JMC\\'+eachfilename, combined_mask)