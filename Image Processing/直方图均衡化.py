# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time

train_path = 'C:/Users/Hao/Desktop/train/annGray/'
classes = ['zh_zhe']

for index, name in enumerate(classes):              # 枚举classes中所有项目
    print("Creating TFRecord of " , name)
    class_path = train_path + name + '/'
    num = 0
    for img_name in os.listdir(class_path):
        num += 1
        i=0
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name            # 记录照片路径
        image = cv2.imread(img_path, 0)             # 打开照片
        lut = np.zeros(256, dtype = image.dtype )   # 创建空的查找表  
  
        hist,bins = np.histogram(image.flatten(),256,[0,256])   
        cdf = hist.cumsum()                         # 计算累积直方图  
        cdf_m = np.ma.masked_equal(cdf,0)           # 除去直方图中的0值  
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())#等同于前面介绍的lut[i] = int(255.0 *p[i])公式  
        cdf = np.ma.filled(cdf_m,0).astype('uint8') # 将掩模处理掉的元素补为0  
  
        #计算  
        result2 = cdf[image]  
        result = cv2.LUT(image, cdf)

        path = 'C:/Users/Hao/Desktop/train/annGray/'
        temp_path = path + name + '/' + str(num+i) + '.jpg'
        cv2.imwrite(temp_path, result)
        i += 1
