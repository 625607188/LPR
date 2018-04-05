import os
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
from Ui_车牌识别调试窗口V2_0 import Ui_MainWindow

sys.path.append('../Deep Learning/')
import eval


class MainWindow(QMainWindow,  Ui_MainWindow):
    def __init__(self,  parent=None):
        super(MainWindow,  self).__init__(parent)
        self.para = []
        self.image1 = []
        self.image2 = []
        self.image1_QImage = QImage()
        self.image2_QImage = QImage()
        self.setupUi(self)
    
    def getimage(self):
        image_path,  _ = QFileDialog.getOpenFileName(self,  "打开图片",  "../Train/svm/has/train/",  "Image file (*jpg)")
        if image_path:
            self.image1 = cv2.imread(image_path, cv2.IMREAD_COLOR)
            self.preprocessimage()
            height, width, _ = self.image1.shape
            self.image1_QImage = QImage(self.image1.data, width, height, width*3, QImage.Format_RGB888)
            self.photo.setPixmap(QPixmap.fromImage(self.image1_QImage))
            self.getlicense()
            self.getcharacter()
            self.getresult()
    
    def preprocessimage(self):
        height, width, _ = self.image1.shape
        if 300/400 > height/width:
            self.image1 = self.image1[0:height, int(width/2-height/2/300*400):int(width/2+height/2/300*400)]
        elif 300/400 < height/width:
            self.image1 = self.image1[int(height/2-width/2/400*300):int(height/2+width/2/400*300), 0:width]
        self.image1 = cv2.resize(self.image1,  (400, 300))
        self.image1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)

    def getlicense(self):
        self.image2 = eval.evaluate_one_photo(cv2.cvtColor(self.image1, cv2.COLOR_RGB2BGR))
        self.image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
        height, width, _ = self.image2.shape
        self.image2_QImage = QImage(self.image2.data, width, height, width * 3, QImage.Format_RGB888)
        self.license.setPixmap(QPixmap.fromImage(self.image2_QImage))

    def getcharacter(self):
        self.para = eval.image_to_character2(self.image2)
        path = ""
        self.char0.clear()
        self.char1.clear()
        self.char2.clear()
        self.char3.clear()
        self.char4.clear()
        self.char5.clear()
        self.char6.clear()
        if 0 < len(self.para):
            path = "temp.jpg"
            cv2.imwrite(path,  self.para[0])
            paragraph = QPixmap(path)
            self.char0.setPixmap(paragraph)
        if 1 < len(self.para):
            path = "temp.jpg"
            cv2.imwrite(path,  self.para[1])
            paragraph = QPixmap(path)
            self.char1.setPixmap(paragraph)
        if 2 < len(self.para):
            path = "temp.jpg"
            cv2.imwrite(path,  self.para[2])
            paragraph = QPixmap(path)
            self.char2.setPixmap(paragraph)
        if 3 < len(self.para):
            path = "temp.jpg"
            cv2.imwrite(path,  self.para[3])
            paragraph = QPixmap(path)
            self.char3.setPixmap(paragraph)
        if 4 < len(self.para):
            path = "temp.jpg"
            cv2.imwrite(path,  self.para[4])
            paragraph = QPixmap(path)
            self.char4.setPixmap(paragraph)
        if 5 < len(self.para):
            path = "temp.jpg"
            cv2.imwrite(path,  self.para[5])
            paragraph = QPixmap(path)
            self.char5.setPixmap(paragraph)
        if 6 < len(self.para):
            path = "temp.jpg"
            cv2.imwrite(path,  self.para[6])
            paragraph = QPixmap(path)
            self.char6.setPixmap(paragraph)
        if path:
            os.remove(path)

    def getresult(self):
        result = eval.evaluate_characters(self.para)
        self.result.setText(result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

