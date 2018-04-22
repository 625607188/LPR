import sys
import datetime
import numpy as np
from PyQt5.QtCore import *
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
        self.result_str = []
        self.coordinate = []
        self.image1_QImage = QImage()
        self.image2_QImage = QImage()
        self.char0_QImage = QImage()
        self.char1_QImage = QImage()
        self.char2_QImage = QImage()
        self.char3_QImage = QImage()
        self.char4_QImage = QImage()
        self.char5_QImage = QImage()
        self.char6_QImage = QImage()
        self.thread_getlicense = GetLicenseThread()
        self.thread_getlicense.signal_licnese.connect(self.getlicense)
        self.thread_getcharacter = GetCharacterThread()
        self.thread_getcharacter.signal_character.connect(self.getcharacter)
        self.thread_getresult = GetResultThread()
        self.thread_getresult.signal_result.connect(self.getresult)
        self.setupUi(self)
    
    def getimage(self):
        image_path,  _ = QFileDialog.getOpenFileName(self, "打开图片", "../Train/svm/has/train/",  "Image file (*.jpg)")
        if image_path:
            self.image1 = cv2.imread(image_path, cv2.IMREAD_COLOR)
            self.preprocessimage()
            height, width, _ = self.image1.shape
            self.image1_QImage = QImage(self.image1.data, width, height, width*3, QImage.Format_RGB888)
            self.photo.setPixmap(QPixmap.fromImage(self.image1_QImage))
            self.thread_getlicense.getimage(self.image1)

    def preprocessimage(self):
        height, width, _ = self.image1.shape
        if 300/400 > height/width:
            self.image1 = self.image1[0:height, int(width/2-height/2/300*400):int(width/2+height/2/300*400)]
        elif 300/400 < height/width:
            self.image1 = self.image1[int(height/2-width/2/400*300):int(height/2+width/2/400*300), 0:width]
        self.image1 = cv2.resize(self.image1,  (400, 300))
        self.image1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)

    def getlicense(self, license):
        [x, y, w, h, self.image2] = license
        if [x, y, w, h] != [0, 0, 0, 0]:
            cv2.rectangle(self.image1, (x, y), (x + w, y + h), (255, 0, 0), 5)
            height, width, _ = self.image1.shape
            self.image1_QImage = QImage(self.image1.data, width, height, width * 3, QImage.Format_RGB888)
            self.photo.setPixmap(QPixmap.fromImage(self.image1_QImage))
            self.image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
            height, width, _ = self.image2.shape
            self.image2_QImage = QImage(self.image2.data, width, height, width * 3, QImage.Format_RGB888)
            self.license.setPixmap(QPixmap.fromImage(self.image2_QImage))

            self.thread_getcharacter.getimage(self.image2)

    def getcharacter(self, character):
        self.para = character
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

        self.thread_getresult.getimage(self.para)

    def getresult(self, result):
        self.result_str.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "   " + result)
        self.result.setText(result)
        self.results.insertPlainText(self.result_str[-1] + "\n")
        
    def printresult(self):
        save_path,  _ = QFileDialog.getSaveFileName(self,  "文件保存",  "C:/Users/Hao/Desktop/",  "Text Files (*.txt)")
        if save_path:
            with open(save_path, "w") as f:
                for index in self.result_str:
                    f.write(index + "\n")


class GetLicenseThread(QThread):
    signal_licnese = pyqtSignal(list)

    def __init__(self, parent=None):
        super(GetLicenseThread, self).__init__(parent)
        self.image = []

    def getimage(self, image):
        self.image = image
        self.start()

    def run(self):
        x, y, w, h, image = eval.evaluate_one_photo(cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        self.signal_licnese.emit([x, y, w, h, image])


class GetCharacterThread(QThread):
    signal_character = pyqtSignal(list)

    def __init__(self, parent=None):
        super(GetCharacterThread, self).__init__(parent)
        self.image = []

    def getimage(self, image):
        self.image = image
        self.start()

    def run(self):
        para = eval.image_to_character2(self.image)
        self.signal_character.emit(para)


class GetResultThread(QThread):
    signal_result = pyqtSignal(str)

    def __init__(self, parent=None):
        super(GetResultThread, self).__init__(parent)
        self.image = []

    def getimage(self, image):
        self.image = image
        self.start()

    def run(self):
        result = eval.evaluate_characters(self.image)
        self.signal_result.emit(result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

