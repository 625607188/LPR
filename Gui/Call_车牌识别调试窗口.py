import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
from Ui_车牌识别调试窗口 import Ui_MainWindow

sys.path.append('../Deep Learning/')
from eval import *


class MainWindow(QMainWindow,  Ui_MainWindow):
    def __init__(self,  parent=None):
        super(MainWindow,  self).__init__(parent)
        self.setupUi(self)
    
    def getimage(self):
        image_path,  _ = QFileDialog.getOpenFileName(self,  "打开图片",  "../Train/svm/has/train/",  "Image file (*jpg)")
        if image_path:
            image = QPixmap(image_path)
            self.Photo.setPixmap(image)
            self.getcharacter(image_path)
            self.getresult(image_path)

    def getcharacter(self, image_path):
        section, para = image_to_character(image_path)
        path = ""
        self.Char0.clear()
        self.Char1.clear()
        self.Char2.clear()
        self.Char3.clear()
        self.Char4.clear()
        self.Char5.clear()
        self.Char6.clear()
        if 0 < section :
            path = "C:/Users/Hao/Desktop/temp/0.jpg"
            cv2.imwrite(path,  para[0])
            paragraph = QPixmap(path)
            self.Char0.setPixmap(paragraph)
        if 1 < section :
            path = "C:/Users/Hao/Desktop/temp/0.jpg"
            cv2.imwrite(path,  para[1])
            paragraph = QPixmap(path)
            self.Char1.setPixmap(paragraph)
        if 2 < section :
            path = "C:/Users/Hao/Desktop/temp/0.jpg"
            cv2.imwrite(path,  para[2])
            paragraph = QPixmap(path)
            self.Char2.setPixmap(paragraph)
        if 3 < section :
            path = "C:/Users/Hao/Desktop/temp/0.jpg"
            cv2.imwrite(path,  para[3])
            paragraph = QPixmap(path)
            self.Char3.setPixmap(paragraph)
        if 4 < section :
            path = "C:/Users/Hao/Desktop/temp/0.jpg"
            cv2.imwrite(path,  para[4])
            paragraph = QPixmap(path)
            self.Char4.setPixmap(paragraph)
        if 5 < section :
            path = "C:/Users/Hao/Desktop/temp/0.jpg"
            cv2.imwrite(path,  para[5])
            paragraph = QPixmap(path)
            self.Char5.setPixmap(paragraph)
        if 6 < section :
            path = "C:/Users/Hao/Desktop/temp/0.jpg"
            cv2.imwrite(path,  para[6])
            paragraph = QPixmap(path)
            self.Char6.setPixmap(paragraph)
            
        if path:
            os.remove(path)

    def getresult(self, image_path):
        result = evaluate_one_image(image_path)
        self.result.setText(result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
