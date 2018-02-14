import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class_path = 'D:/final work/FinalWork-Ms.Wu/Project/Train/annGray'

def saltpepper(img,n):
    m=int((img.shape[0]*img.shape[1])*n)
    m = 10
    for a in range(m):
        i=int(np.random.random()*img.shape[1])
        j=int(np.random.random()*img.shape[0])
        if img.ndim==2:
            img[j,i]=255
        elif img.ndim==3:
            img[j,i,0]=255
            img[j,i,1]=255
            img[j,i,2]=255
    for b in range(m):
        i=int(np.random.random()*img.shape[1])
        j=int(np.random.random()*img.shape[0])
        if img.ndim==2:
            img[j,i]=0
        elif img.ndim==3:
            img[j,i,0]=0
            img[j,i,1]=0
            img[j,i,2]=0
    return img


for dirpath, dirname, filename in os.walk('.'):
    num = 0
    for img_name in os.listdir(dirpath):
        if '.jpg' in img_name:
            num += 1
    i = 0
    for img_name in os.listdir(dirpath):
        if '.jpg' in img_name:
            image_path = dirpath + '/' + img_name
            print(image_path)
            img = cv2.imread(image_path)
            saltImage = saltpepper(img,0.01)
            cv2.imwrite(dirpath + '/' + str(num+i) + '.jpg', saltImage)
            i += 1
'''class_path = 'C:/Users/Hao/Desktop/train/svm/no/train/1.jpg'
img = cv2.imread(class_path)
saltImage = saltpepper(img, 10)
plt.subplot(2, 1, 1), plt.imshow(img)
plt.subplot(2, 1, 2), plt.imshow(saltImage)
plt.show()'''
