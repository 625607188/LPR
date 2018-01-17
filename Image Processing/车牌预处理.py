import cv2  
import numpy as np
import matplotlib.pyplot as plt
import os   
  

# 直方图均衡化
#image1 = cv2.equalizeHist(image)

path = "C:/Users/Hao/Desktop/train/svm/has/train"

num = 0
for img_name in os.listdir(path):
    num += 1
m = 0
for img_name in os.listdir(path):
    image_path = path + '/' + img_name
    image = cv2.imread(image_path)
    for i in range(len(image)):
        for l in range(len(image[i])):
            image[i][l] = image[i][l] * 0.5
    cv2.imwrite(path + '/' + str(num+m) + '.jpg', image)
    m = m + 1
    for i in range(len(image)):
        for l in range(len(image[i])):
            image[i][l] = image[i][l] * 1.5
    cv2.imwrite(path + '/' + str(num+m) + '.jpg', image)
    m = m + 1
