# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time

train_path = 'C:/Users/Hao/Desktop/train/annGray/'
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
          'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
          'W', 'X', 'Y', 'Z', 'zh_zhe']

for index, name in enumerate(classes):              # 枚举classes中所有项目
    print("Creating TFRecord of " , name)
    class_path = train_path + name + '/'
    num = 0
    for img_name in os.listdir(class_path):
        num += 1
        i=0
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name            # 记录照片路径
        img = cv2.imread(img_path, 0)               # 打开照片
        (x, y) = img.shape
        if x > y:
            img = cv2.copyMakeBorder(img, 0, 0, int((x - y) / 2), int((x - y) / 2), cv2.BORDER_CONSTANT,
                                         value=[0, 0, 0])
        elif x < y:
            img = cv2.copyMakeBorder(img, int((x - y) / 2), int((x - y) / 2), 0, 0, cv2.BORDER_CONSTANT,
                                         value=[0, 0, 0])
        elif x == y:
            img = img

        path = 'C:/Users/Hao/Desktop/train/annGray/'
        temp_path = path + name + '/' + str(num+i) + '.jpg'
        cv2.imwrite(temp_path, img)
        i += 1
