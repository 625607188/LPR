import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''class_path = 'D:/final work/FinalWork-Ms.Wu/Project/Train/annGray/zh_zhe/2.jpg'
img = cv2.imread(class_path)
rows,cols,_ = img.shape
pts1 = np.float32([[2,5],[68,18],[5,20]])
pts2 = np.float32([[1,6],[68,18],[7,19]])
M = cv2.getAffineTransform(pts1,pts2)
#第三个参数：变换后的图像大小
res1 = cv2.warpAffine(img,M,(cols,rows))
pts1 = np.float32([[2, 2],[68,18],[132,2]])
pts2 = np.float32([[10, 4],[68,18],[130,4]])
M = cv2.getAffineTransform(pts1,pts2)
res2 = cv2.warpAffine(img,M,(cols,rows))

plt.subplot(3, 1, 1), plt.imshow(img)
plt.subplot(3, 1, 2), plt.imshow(res1)
plt.subplot(3, 1, 3), plt.imshow(res2)
plt.show()'''


class_path = 'D:/final work/FinalWork-Ms.Wu/Project/Train/annGray/zh_zhe'
num = 0
for img_name in os.listdir(class_path):
    num += 1
i = 0
for img_name in os.listdir(class_path):
    path = class_path + '/' + img_name
    img = cv2.imread(path)
    rows,cols,_ = img.shape
    pts1 = np.float32([[2,5],[68,18],[5,20]])
    pts2 = np.float32([[1,6],[68,18],[7,19]])
    M = cv2.getAffineTransform(pts1,pts2)
    #第三个参数：变换后的图像大小
    res1 = cv2.warpAffine(img,M,(cols,rows))
    pts1 = np.float32([[2, 2],[68,18],[132,2]])
    pts2 = np.float32([[10, 4],[68,18],[130,4]])
    M = cv2.getAffineTransform(pts1,pts2)
    res2 = cv2.warpAffine(img,M,(cols,rows))

    cv2.imwrite(class_path + '/' + str(num+i) + '.jpg', res1)
    i += 1
    cv2.imwrite(class_path + '/' + str(num+i) + '.jpg', res2)
    i += 1
