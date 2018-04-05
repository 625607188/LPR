import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
from Ui_车牌识别调试窗口 import Ui_MainWindow

sys.path.append('../Deep Learning/')
import eval


class MainWindow(QMainWindow,  Ui_MainWindow):
    def __init__(self,  parent=None):
        super(MainWindow,  self).__init__(parent)
        self.para = []
        self.setupUi(self)
    
    def getimage(self):
        image_path,  _ = QFileDialog.getOpenFileName(self,  "打开图片",  "C:/Users/Hao/Desktop/test/",  "Image file (*jpg)")
        if image_path:
            self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            height, width, _ = self.image.shape
            self.image_QImage = QImage(self.image.data, width, height, QImage.Format_RGB888)
            self.photo.setPixmap(QPixmap.fromImage(self.image_QImage))
            self.getcharacter()
            self.getresult()

    def getcharacter(self):
        self.para = eval.image_to_character2(self.image)
        width, height = (20,  20)
        self.char0.clear()
        self.char1.clear()
        self.char2.clear()
        self.char3.clear()
        self.char4.clear()
        self.char5.clear()
        self.char6.clear()
        if 0 < len(self.para):
            self.char0_QImage = QImage(self.para[0].data, width, height, QImage.Format_Grayscale8)
            self.char0.setPixmap(QPixmap.fromImage(self.char0_QImage))
        if 1 < len(self.para):
            self.char1_QImage = QImage(self.para[1].data, width, height, QImage.Format_Grayscale8)
            self.char1.setPixmap(QPixmap.fromImage(self.char1_QImage))
        if 2 < len(self.para):
            self.char2_QImage = QImage(self.para[2].data, width, height, QImage.Format_Grayscale8)
            self.char2.setPixmap(QPixmap.fromImage(self.char2_QImage))
        if 3 < len(self.para):
            self.char3_QImage = QImage(self.para[3].data, width, height, QImage.Format_Grayscale8)
            self.char3.setPixmap(QPixmap.fromImage(self.char3_QImage))
        if 4 < len(self.para):
            self.char4_QImage = QImage(self.para[4].data, width, height, QImage.Format_Grayscale8)
            self.char4.setPixmap(QPixmap.fromImage(self.char4_QImage))
        if 5 < len(self.para):
            self.char5_QImage = QImage(self.para[5].data, width, height, QImage.Format_Grayscale8)
            self.char5.setPixmap(QPixmap.fromImage(self.char5_QImage))
        if 6 < len(self.para):
            self.char6_QImage = QImage(self.para[6].data, width, height, QImage.Format_Grayscale8)
            self.char6.setPixmap(QPixmap.fromImage(self.char6_QImage))

    def getresult(self):
        result = eval.evaluate_characters(self.para)
        self.result.setText(result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

