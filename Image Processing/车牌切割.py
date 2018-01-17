# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time

img = cv2.imread("29.jpg", 0)
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, image = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

(x, y) = image.shape

image = list(image)
line = [0] * x
sum = 0
for i in range(x):
    for l in range(y):
        line[i] += image[i][l]
    sum += line[i]

for i in range(x):
    if line[x-i-1] < (sum/x/x) :
        del image[x-i-1]

plt.subplot(1, 1, 1), plt.imshow(image)
plt.show()

image = np.array(image)
(x, y) = image.shape
line = [0] * y
para = [[]]
section = 0
top = None
bottom = None
for i in range(y):
    line[i] = 0
    for l in range(x):
        line[i] += image[l][i]
    if (line[i] < sum/y/y) and (top != None):
        bottom = i - 1
        for m in range(x):
            para[section].append(image[m][top:bottom])
        para.append([])
        section += 1
        top = None
    elif (line[i] > sum/y/y) and (top == None):
        top = i

l=0
for i in range(section):
    if(len(para[section-1-i][0])<5):
        del para[section-1-i]
        l += 1
section -= l


para.remove([])
for i in range(section):
    para[i] = np.array(para[i]).reshape(x, -1)
    (x, y) = para[i].shape
    if x > y:
        para[i] = cv2.copyMakeBorder(para[i], 0, 0, int((x - y) / 2), int((x - y) / 2), cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif x < y:
        para[i] = cv2.copyMakeBorder(para[i], int((y - x) / 2), int((y - x) / 2), 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif x == y:
        para[i] = para[i]
    para[i] = cv2.resize(para[i], (20, 20), interpolation=cv2.INTER_AREA)

for i in range(section):
    plt.subplot(4, 4, i+1), plt.imshow(para[i])


plt.show()
