# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\final work\FinalWork-Ms.Wu\Project\Gui\车牌识别调试窗口V2_0.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(741, 422)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralWidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 30, 421, 361))
        self.groupBox.setObjectName("groupBox")
        self.photo = QtWidgets.QLabel(self.groupBox)
        self.photo.setGeometry(QtCore.QRect(9, 20, 400, 300))
        self.photo.setText("")
        self.photo.setScaledContents(True)
        self.photo.setObjectName("photo")
        self.choose_image = QtWidgets.QPushButton(self.groupBox)
        self.choose_image.setGeometry(QtCore.QRect(330, 330, 75, 23))
        self.choose_image.setObjectName("choose_image")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralWidget)
        self.groupBox_2.setGeometry(QtCore.QRect(450, 30, 261, 141))
        self.groupBox_2.setObjectName("groupBox_2")
        self.license = QtWidgets.QLabel(self.groupBox_2)
        self.license.setGeometry(QtCore.QRect(30, 50, 200, 50))
        self.license.setText("")
        self.license.setScaledContents(True)
        self.license.setObjectName("license")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralWidget)
        self.groupBox_3.setGeometry(QtCore.QRect(450, 163, 261, 121))
        self.groupBox_3.setObjectName("groupBox_3")
        self.layoutWidget = QtWidgets.QWidget(self.groupBox_3)
        self.layoutWidget.setGeometry(QtCore.QRect(0, 50, 261, 41))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.char0 = QtWidgets.QLabel(self.layoutWidget)
        self.char0.setText("")
        self.char0.setScaledContents(True)
        self.char0.setObjectName("char0")
        self.horizontalLayout.addWidget(self.char0)
        self.char1 = QtWidgets.QLabel(self.layoutWidget)
        self.char1.setText("")
        self.char1.setScaledContents(True)
        self.char1.setObjectName("char1")
        self.horizontalLayout.addWidget(self.char1)
        self.char2 = QtWidgets.QLabel(self.layoutWidget)
        self.char2.setText("")
        self.char2.setScaledContents(True)
        self.char2.setObjectName("char2")
        self.horizontalLayout.addWidget(self.char2)
        self.char3 = QtWidgets.QLabel(self.layoutWidget)
        self.char3.setText("")
        self.char3.setScaledContents(True)
        self.char3.setObjectName("char3")
        self.horizontalLayout.addWidget(self.char3)
        self.char4 = QtWidgets.QLabel(self.layoutWidget)
        self.char4.setText("")
        self.char4.setScaledContents(True)
        self.char4.setObjectName("char4")
        self.horizontalLayout.addWidget(self.char4)
        self.char5 = QtWidgets.QLabel(self.layoutWidget)
        self.char5.setText("")
        self.char5.setScaledContents(True)
        self.char5.setObjectName("char5")
        self.horizontalLayout.addWidget(self.char5)
        self.char6 = QtWidgets.QLabel(self.layoutWidget)
        self.char6.setText("")
        self.char6.setScaledContents(True)
        self.char6.setObjectName("char6")
        self.horizontalLayout.addWidget(self.char6)
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralWidget)
        self.groupBox_4.setGeometry(QtCore.QRect(450, 276, 261, 115))
        self.groupBox_4.setObjectName("groupBox_4")
        self.result = QtWidgets.QTextEdit(self.groupBox_4)
        self.result.setGeometry(QtCore.QRect(20, 30, 221, 61))
        self.result.setObjectName("result")
        MainWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(MainWindow)
        self.choose_image.clicked.connect(MainWindow.getimage)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "待识别的图片"))
        self.choose_image.setText(_translate("MainWindow", "选择图片"))
        self.groupBox_2.setTitle(_translate("MainWindow", "识别车牌结果"))
        self.groupBox_3.setTitle(_translate("MainWindow", "车牌分割结果"))
        self.groupBox_4.setTitle(_translate("MainWindow", "字符识别结果"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

