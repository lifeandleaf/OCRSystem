import sys, os, json
import threading
import time
from PIL import ImageQt


from base import Ui_OCR
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QListWidgetItem, QListView, QInputDialog, QLineEdit
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QSize

from tools import Recognize, cropImage

class MyWindow(QWidget, Ui_OCR):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)
        self.setupUi(self)
        # init window
        self.stackedWidget.setCurrentIndex(1)
        self.progressBar.reset()
        # 绑定回调函数
        self.pushButton.clicked.connect(self.pushButtonClicked)
        self.pushButton_2.clicked.connect(self.pushButton_2Clicked)
        self.pushButton_3.clicked.connect(self.pushButton_3Clicked)

        self.pushButton_7.clicked.connect(self.pushButton_7Clicked)

        self.pushButton_9.clicked.connect(self.pushButton_9Clicked)
        self.pushButton_10.clicked.connect(self.pushButton_10Clicked)
        self.pushButton_11.clicked.connect(self.pushButton_11Clicked)
        self.pushButton_12.clicked.connect(self.pushButton_12Clicked)
        self.pushButton_13.clicked.connect(self.pushButton_13Clicked)

        self.pushButton_15.clicked.connect(self.pushButton_15Clicked)
        self.listWidget_3.itemDoubleClicked.connect(self.listWidget_3ItemClicked)
        self.listWidget_4.itemDoubleClicked.connect(self.listWidget_4ItemClicked)

        self.show()

    # 图像采集界面
    def pushButtonClicked(self):
        self.stackedWidget.setCurrentIndex(1)

    # 打开摄像头
    def pushButton_9Clicked(self):
        img = QPixmap('./test.jpg')
        img = img.scaled(self.label_3.width()-10, self.label_3.height()-10, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_3.setPixmap(img)

    # 文本识别界面
    def pushButton_2Clicked(self):
        self.stackedWidget.setCurrentIndex(2)

    def pushButton_7Clicked(self):
        dir = QFileDialog.getExistingDirectory(self, '打开文件夹', '.')
        if dir == '':
            return
        fileList = os.listdir(dir)
        fileListFilter = []
        for path in fileList:
            if path.endswith('.jpg') or path.endswith('png'):
                fileListFilter.append(os.path.join(dir, path))
        self.listWidget_3AddItem(fileListFilter)
        if self.listWidget_3.count() > 0:
            self.listWidget_3.setCurrentRow(0)
            self.listWidget_3ItemShow()

    def listWidget_3AddItem(self, items):
        '''
        :param items:a list add to listwidget_3
        :return: void
        '''
        for item in items:
            self.listWidget_3.addItem(item)

    def listWidget_3ItemClicked(self):
        index = self.listWidget_3.selectedIndexes()[0]
        self.listWidget_3.setCurrentIndex(index)
        self.listWidget_3ItemShow()

    def listWidget_3ItemShow(self):
        imgPath = self.listWidget_3.currentItem().text()
        fileExtension = os.path.splitext(imgPath)[-1]
        # 原图获取
        img = QPixmap(imgPath)
        img = img.scaled(self.label_4.width() - 10, self.label_4.height() - 10, Qt.KeepAspectRatio,
                         Qt.SmoothTransformation)
        self.label_4.setPixmap(img)
        # 缩略图获取
        # 首先重置listwidget4
        self.listWidget_4.clear()
        info = {}
        infoPath = imgPath.replace(fileExtension, '.json')
        if os.path.exists(infoPath):
            with open(infoPath) as f:
                info = json.loads(f.read())
        if 'points' in info:
            for point in info['points']:
                cropImg = cropImage(imgPath, point)
                cropImg = QPixmap(ImageQt.toqpixmap(cropImg))
                cropImg = cropImg.scaled(self.listWidget_4.width()-10, self.listWidget_4.height()-10, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                item = QListWidgetItem()
                item.setIcon(QIcon(cropImg))
                self.listWidget_4.setIconSize(QSize(self.listWidget_4.width()-10, self.listWidget_4.height()-10))
                self.listWidget_4.setResizeMode(QListView.Adjust)
                self.listWidget_4.addItem(item)

        # 标签获取
        if 'label' in info:
            self.label_7.setText(info['label'][0])

    def pushButton_15Clicked(self):
        if self.listWidget_3.count() <= 0:
            return
        index = self.listWidget_3.currentRow()
        index += 1
        if index < self.listWidget_3.count():
            self.listWidget_3.setCurrentRow(index)
        self.listWidget_3ItemShow()

    def pushButton_11Clicked(self):
        if self.listWidget_3.count() <= 0:
            return
        index = self.listWidget_3.currentRow()
        index -= 1
        if index >= 0:
            self.listWidget_3.setCurrentRow(index)
        self.listWidget_3ItemShow()

    def pushButton_10Clicked(self):
        if self.listWidget_3.count() <= 0:
            return
        imgPath = self.listWidget_3.currentItem().text()
        imgExtension = os.path.splitext(imgPath)[-1]
        jsonPath = imgPath.replace(imgExtension, '.json')
        # 检查是否已存在检测结果(不存在则进行识别,并写入json)
        if os.path.exists(jsonPath):
            with open(jsonPath, 'r') as f:
                info = json.loads(f.read())
                self.label_7.setText(info['label'][0])
        else:
            info = Recognize(imgPath)
            self.label_7.setText(info['label'][0])
            with open(jsonPath, 'w') as f:
                f.write(json.dumps(info))

    def pushButton_13Clicked(self):
        if self.listWidget_3.count() <= 0:
            return
        imgPathList = [self.listWidget_3.item(i).text() for i in range(self.listWidget_3.count())]
        count = 0
        for imgPath in imgPathList:
            imgExtension = os.path.splitext(imgPath)[-1]
            jsonPath = imgPath.replace(imgExtension, '.json')
            # 检查是否已存在检测结果(不存在则进行识别,并写入json)
            if not os.path.exists(jsonPath):
                info = Recognize(imgPath)
                with open(jsonPath, 'w') as f:
                    f.write(json.dumps(info))
            count += 1
            self.progressBar.setValue(int(100 * count / len(imgPathList)))
        self.progressBar.setValue(100)

    def listWidget_4ItemClicked(self):
        index = self.listWidget_4.currentRow()
        imgPath = self.listWidget_3.currentItem().text()
        imgExtension = os.path.splitext(imgPath)[-1]
        jsonPath = imgPath.replace(imgExtension, '.json')
        with open(jsonPath, 'r') as f:
            info = json.loads(f.read())
        self.label_7.setText(info['label'][index])

    def pushButton_12Clicked(self):
        if self.listWidget_3.count() <= 0 or self.listWidget_4.count() <= 0:
            return
        text, ok = QInputDialog.getText(self, '修改Label', '请输入文本', QLineEdit.Normal, '')
        if not ok:
            return
        imgPath = self.listWidget_3.currentItem().text()
        imgExtension = os.path.splitext(imgPath)[-1]
        jsonPath = imgPath.replace(imgExtension, '.json')
        with open(jsonPath, 'r') as f:
            info = json.loads(f.read())
        index = self.listWidget_4.currentRow()
        info['label'][index] = text
        with open(jsonPath, 'w') as f:
            f.write(json.dumps(info))
        self.label_7.setText(text)

    # 数据生成界面
    def pushButton_3Clicked(self):
        self.stackedWidget.setCurrentIndex(0)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MyWindow = MyWindow()
    sys.exit(app.exec())