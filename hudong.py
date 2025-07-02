#gui页面设计
import random
import sys
import numpy as np
import subprocess
import networkx as nx
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
import matplotlib.pyplot as plt
community = []
class Ui_Form(object):
    def setupUi(self, Form):
        # Form.setStyleSheet("background-color: #BBFFFF;")
        Form.setStyleSheet("QWidget { background-color: white; }")
        Form.setObjectName("Form")
        Form.resize(980, 830)
        self.groupBox = QtWidgets.QGroupBox(Form)
        # 数据集
        self.groupBox.setGeometry(QtCore.QRect(50, 20,411, 300))
        self.groupBox.setObjectName("groupBox")
        self.comboBox = QtWidgets.QComboBox(self.groupBox)
        self.comboBox.setGeometry(QtCore.QRect(110, 20, 281, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.textBrowser = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser.setGeometry(QtCore.QRect(110, 80, 281, 130))
        self.textBrowser.setObjectName("textBrowser")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(30, 20, 72, 15))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(30, 80, 81, 16))
        self.label_2.setObjectName("label_2")
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setGeometry(QtCore.QRect(500, 20, 411, 300))
        self.groupBox_2.setObjectName("groupBox_2")
        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox_2.setGeometry(QtCore.QRect(130, 20, 251, 21))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(20, 20, 101, 20))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(30, 60, 71, 31))
        self.label_4.setObjectName("label_4")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.groupBox_2)
        self.textBrowser_2.setGeometry(QtCore.QRect(130, 60, 256, 120))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.label_9 = QtWidgets.QLabel(self.groupBox_2)
        self.label_9.setGeometry(QtCore.QRect(20, 210, 101, 16))
        self.label_9.setObjectName("label_9")
        self.TextEdit_3 = QtWidgets.QTextEdit(self.groupBox_2)
        self.TextEdit_3.setGeometry(QtCore.QRect(130, 210, 250, 30))
        self.TextEdit_3.setText("")
        self.TextEdit_3.setObjectName("TextEdit_3")
        self.pushButton = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton.setGeometry(QtCore.QRect(128, 250, 91, 21))
        self.pushButton.setObjectName("pushButton")
        self.groupBox_3 = QtWidgets.QGroupBox(Form)
        self.groupBox_3.setGeometry(QtCore.QRect(50, 350, 861, 410))
        self.groupBox_3.setObjectName("groupBox_3")
        self.groupBox_3.setStyleSheet("QGroupBox {"
                                      "  border: 2px dashed #483D8B;"  # 黑色边框
                                      "  border-radius: 10px;"  # 边框圆角
                                      "  margin: 10px;"  # 分组框外边距
                                      "}")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit.setGeometry(QtCore.QRect(160, 70, 340, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.label_5 = QtWidgets.QLabel(self.groupBox_3)
        self.label_5.setGeometry(QtCore.QRect(50, 70, 111, 16))
        self.label_5.setObjectName("label_5")
        self.label_7 = QtWidgets.QLabel(self.groupBox_3)
        self.label_7.setGeometry(QtCore.QRect(60, 370, 71, 21))
        self.label_7.setObjectName("label_7")
        self.label_6 = QtWidgets.QLabel(self.groupBox_3)
        self.label_6.setGeometry(QtCore.QRect(50, 150, 91, 16))
        self.label_6.setObjectName("label_6")
        # self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox_3)
        # self.lineEdit_2.setGeometry(QtCore.QRect(160, 370, 191, 16))
        # self.lineEdit_2.setObjectName("lineEdit_2")
        self.textEdit = QtWidgets.QTextEdit(self.groupBox_3)
        self.textEdit.setGeometry(QtCore.QRect(160, 150, 300, 161))
        # self.textEdit.setMarkdown("")
        self.textEdit.setObjectName("textEdit")
        self.label_8 = QtWidgets.QLabel(self.groupBox_3)
        self.label_8.setGeometry(QtCore.QRect(60, 330, 101, 21))
        self.label_8.setObjectName("label_8")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_2.setGeometry(QtCore.QRect(180, 330, 120, 33))
        self.pushButton_2.setObjectName("pushButton_2")
        # Right side image QLabel inside groupBox_3
        self.imageLabel = QtWidgets.QLabel(self.groupBox_3)
        self.imageLabel.setGeometry(QtCore.QRect(510, 60, 330, 300))  # Adjust size and position as needed
        self.imageLabel.setStyleSheet("border: 1px solid black;")  # Optional border to visualize label size
        self.imageLabel.setObjectName("imageLabel")
        self.imageLabel.setText("")
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "基于多视图一致性学习的联合谱嵌入聚类分析系统"))
        self.groupBox.setTitle(_translate("Form", "数据集选择"))
        self.comboBox.addItem("")
        self.comboBox.setItemText(0, _translate("Form", " "))
        self.comboBox.setItemText(1, _translate("Form", "Caltech101_7.mat"))
        self.comboBox.setItemText(2, _translate("Form", "3Sources.mat"))
        self.comboBox.setItemText(3, _translate("Form", "MSRC_v1.mat"))
        self.comboBox.setItemText(4, _translate("Form", "ORL.mat"))
        self.comboBox.setItemText(5, _translate("Form", "BBCSport.mat"))
        # 启动数据集介绍
        self.comboBox.currentIndexChanged.connect(self.handleSelectionChange1)
        self.label.setText(_translate("Form", "数据集"))
        self.label_2.setText(_translate("Form", "数据集介绍"))
        self.groupBox_2.setTitle(_translate("Form", "算法"))
        self.comboBox_2.setItemText(0, _translate("Form", " "))
        self.comboBox_2.setItemText(1, _translate("Form", "JSEC（基于多视图多样性学习的联合谱嵌入聚类算法）"))
        self.comboBox_2.setItemText(2, _translate("Form", "DUBGHIF（基于动态统一二部图学习与高阶信息融合的多视图聚类算法）"))
        # 启动算法介绍
        self.comboBox_2.currentIndexChanged.connect(self.handleSelectionChange2)
        self.label_3.setText(_translate("Form", "聚类算法"))
        self.label_4.setText(_translate("Form", "算法介绍"))
        self.label_9.setText(_translate("Form", "运行次数"))
        self.pushButton.setText(_translate("Form", "确定"))
        self.pushButton.clicked.connect(self.execAlgorithm)
        self.groupBox_3.setTitle(_translate("Form", "运行结果"))
        self.lineEdit.setText(_translate("Form", ""))
        self.label_5.setText(_translate("Form", "聚类算法"))
        self.label_6.setText(_translate("Form", "评价指标"))
        self.label_7.setText(_translate("Form", ""))
        # self.lineEdit_2.setText(_translate("Form", "0.29369373205346905"))
        self.textEdit.setHtml(_translate("Form", f"""
        <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0//EN" "http://www.w3.org/TR/REC-html40/strict.dtd">
        <html><head>
        <meta name="qrichtext" content="1" />
        <style type="text/css">
        p, li {{ white-space: pre-wrap; font-family: 'Arial'; font-size: 9pt; color: #333333; }}
        </style></head>
        <body>
        </body></html>"""))
        self.label_8.setText(_translate("Form", "收敛曲线"))
        self.pushButton_2.setText(_translate("Form", "点击显示结果"))
        self.pushButton_2.clicked.connect(self.draw)
    def handleSelectionChange1(self):
        # self.textBrowser.clear()
        if (self.comboBox.currentIndex() == 0):
            self.textBrowser.setText(' ')
        if (self.comboBox.currentIndex() == 1):
            self.textBrowser.setText(
                "Caltech101_7数据集在101个类的图像数据集中选择了7个较为常见的类别，包含1474个样本，提取6个常用的图像特征以形成不同的视图特征。")
        if (self.comboBox.currentIndex() == 2):
            self.textBrowser.setText(
                "3Sources数据集包括三个新闻资源：BBC、Reuters、The Guardian，共169篇文章，按类别可划分为六类，分别是：商业、娱乐、政治、健康、体育和科技。")
        if (self.comboBox.currentIndex() == 3):
            self.textBrowser.setText(
                "MSRC-V1是一个图像数据集，由210张图像组成，分为7个类别，包括树木、建筑物、飞机、奶牛、人脸、汽车和自行车，GIST和HOG特征为两个视图。")
        if (self.comboBox.currentIndex() == 4):
            self.textBrowser.setText(
                "ORL数据集包含40个不同的人的400张灰度人脸图像。每个人提供10张不同姿势和表情的图像，在相机中以不同的光照条件捕获。")
        if (self.comboBox.currentIndex() == 5):
            self.textBrowser.setText(
                "BBCSport数据集包含544篇体育文章，从田径、板球、足球、橄榄球和网球5个主题领域收集，每篇文章被分为两个特征。")
    def handleSelectionChange2(self):
        index = self.comboBox_2.currentIndex()
        if index == 1:
            self.textBrowser_2.setText(
                "JSEC 算法通过多视图多样性学习、高阶信息融合和联合谱嵌入实现聚类。首先，利用 KNN 构建相似矩阵，分解为一致特征和多样特征，抑制噪声并保留全局结构。然后，通过混合相似图捕捉高阶关联，约束一致特征与混合图匹配，增强全局建模能力。最后，将各视图一致特征映射到低维空间生成联合嵌入矩阵，通过交替迭代优化确保模块协同，提升聚类精度和鲁棒性。"
            )
        elif index == 2:
            self.textBrowser_2.setText(
                "DUBGHIF算法设计动态滤波器构建自适应相似性矩阵，缓解噪声干扰并捕捉数据分布变化；其次，构造高阶拉普拉斯矩阵，融合一阶与二阶邻接关系，刻画视图间复杂的非线性关联，挖掘数据隐含的高阶结构信息；进一步，引入统一二部图学习框架，结合交替采样锚点策略和动态权重分配机制，实现多视图信息的动态自适应融合。"
            )
        else:
            self.textBrowser_2.setText("")
        self.lineEdit.setText(self.comboBox_2.currentText())
    def execAlgorithm(self):
        try:
            num = int(self.TextEdit_3.toPlainText())
        except ValueError:
            QtWidgets.QMessageBox.warning(None, "输入错误", "请输入有效的聚类个数（整数）")
            return
        dataset = self.comboBox.currentText()
        algorithm = self.comboBox_2.currentText()
        print(f"Selected algorithm: {algorithm}")
        self.textEdit.setText(f"Algorithm: {algorithm}\nDataset: {dataset}\nNumber: {num}")
        QApplication.processEvents()
        if "JSEC" in algorithm:
            jsec_algorithm(dataset, num)
        elif "DUBGHIF" in algorithm:
            dubghif_algorithm(dataset, num)
    def draw(self):
        num = int(self.TextEdit_3.toPlainText())
        dataset = self.comboBox.currentText()
        algorithm = self.comboBox_2.currentText()
        if "JSEC" in algorithm:
            pixmap = QPixmap(r"C:\\...\\JSEC\\imgs\\jsec_image.jpg")
        elif "DUBGHIF" in algorithm:
            pixmap = QPixmap(r"C:\\...\\DUBGHIF\\imgs\\dubghif_image.jpg")
        else:
            pixmap = QPixmap()
        if not pixmap.isNull():
            self.imageLabel.setPixmap(pixmap)
            self.imageLabel.setScaledContents(True)
            QApplication.processEvents()
        else:
            print("加载图片失败，检查文件路径。")
if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()  # 创建主窗口
    ui1 = Ui_Form()
    ui1.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())