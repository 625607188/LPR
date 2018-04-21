import sys
import datetime
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
        self.image1_QImage = QImage()
        self.image2_QImage = QImage()
        self.char0_QImage = QImage()
        self.char1_QImage = QImage()
        self.char2_QImage = QImage()
        self.char3_QImage = QImage()
        self.char4_QImage = QImage()
        self.char5_QImage = QImage()
        self.char6_QImage = QImage()
        self.setupUi(self)
    
    def getimage(self):
        image_path,  _ = QFileDialog.getOpenFileName(self, "打开图片", "../Train/svm/has/train/",  "Image file (*jpg)")
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
        x, y, w, h, self.image2 = eval.evaluate_one_photo(cv2.cvtColor(self.image1, cv2.COLOR_RGB2BGR))
        cv2.rectangle(self.image1, (x, y), (x + w, y + h), (255, 0, 0), 5)
        height, width, _ = self.image1.shape
        self.image1_QImage = QImage(self.image1.data, width, height, width * 3, QImage.Format_RGB888)
        self.photo.setPixmap(QPixmap.fromImage(self.image1_QImage))
        self.image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
        height, width, _ = self.image2.shape
        self.image2_QImage = QImage(self.image2.data, width, height, width * 3, QImage.Format_RGB888)
        self.license.setPixmap(QPixmap.fromImage(self.image2_QImage))

    def getcharacter(self):
        self.para = eval.image_to_character2(self.image2)
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
        self.result_str.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "   " + eval.evaluate_characters(self.para))
        self.result.insertPlainText(self.result_str[-1] + "\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

