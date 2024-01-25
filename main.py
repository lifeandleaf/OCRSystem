import sys, os, json
import threading
import time
from PIL import ImageQt, Image # pillow == 8.1.0
import cv2


from base import Ui_OCR
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QListWidgetItem, QListView, QInputDialog, QLineEdit
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPixmap, QIcon, QImage, QMovie
from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal

from tools import Recognize, cropImage

class getInfo(QThread):
    signal = pyqtSignal(dict)
    def __init__(self, imgPath):
        super().__init__()
        self.imgPath = imgPath

    def run(self):
        fileExtension = os.path.splitext(self.imgPath)[-1]
        if os.path.exists(self.imgPath.replace(fileExtension, '.json')):
            self.signal.emit(self.imgPath)
            return
        info = Recognize(self.imgPath)
        with open(self.imgPath.replace(fileExtension, '.json'), 'w') as f:
            f.write(json.dumps(info))
        self.signal.emit(info)

class MyWindow(QWidget, Ui_OCR):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)
        self.setupUi(self)
        # init window
        self.stackedWidget.setCurrentIndex(1)
        self.progressBar.reset()
        self.timer_camera = QTimer()
        self.videoCap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.radioButton.setEnabled(False)
        self.videoCapDir = ''
        # 绑定回调函数
        self.timer_camera.timeout.connect(self.showVideo)

        self.pushButton.clicked.connect(self.pushButtonClicked)
        self.pushButton_2.clicked.connect(self.pushButton_2Clicked)
        self.pushButton_3.clicked.connect(self.pushButton_3Clicked)

        self.pushButton_7.clicked.connect(self.pushButton_7Clicked)
        self.pushButton_8.clicked.connect(self.pushButton_8Clicked)
        self.pushButton_9.clicked.connect(self.pushButton_9Clicked)
        self.pushButton_10.clicked.connect(self.pushButton_10Clicked)
        self.pushButton_11.clicked.connect(self.pushButton_11Clicked)
        self.pushButton_12.clicked.connect(self.pushButton_12Clicked)
        self.pushButton_13.clicked.connect(self.pushButton_13Clicked)

        self.pushButton_15.clicked.connect(self.pushButton_15Clicked)

        self.listWidget_2.itemDoubleClicked.connect(self.listWidget_2ItemClicked)
        self.listWidget_3.itemDoubleClicked.connect(self.listWidget_3ItemClicked)
        self.listWidget_4.itemDoubleClicked.connect(self.listWidget_4ItemClicked)
        self.show()

    # 图像采集界面
    def pushButtonClicked(self):
        self.stackedWidget.setCurrentIndex(1)

    def listWidget_2ItemClicked(self):
        if self.listWidget_2.count() <= 0:
            return
        imgPath = self.listWidget_2.currentItem().text()
        img = QPixmap(imgPath)
        img = img.scaled(self.label_5.width() - 10, self.label_5.height() - 10, Qt.KeepAspectRatio,
                         Qt.SmoothTransformation)
        self.label_5.setPixmap(img)
        fileExtension = os.path.splitext(imgPath)[-1]
        jsonPath = imgPath.replace(fileExtension, '.json')
        if not os.path.exists(jsonPath):
            self.infoThread = getInfo(imgPath)
            self.infoThread.signal.connect(self.showImageLabel)
            self.infoThread.start()
        else:
            with open(jsonPath, 'r', encoding='utf-8') as f:
                info = json.loads(f.read())
                self.showImageLabel(info)


    # 打开摄像头
    def pushButton_9Clicked(self):
        if self.timer_camera.isActive() == False:
            flag = self.videoCap.open(self.CAM_NUM)
            if flag == False:
                msg = QMessageBox.warning(self, '警告', "请检查相机与电脑是否正确连接", buttons=QMessageBox.Ok)
            else:
                self.timer_camera.start(30)
                self.pushButton_9.setText('关闭摄像头')
                if self.pushButton_8.text() == '拍摄':
                    self.radioButton.setEnabled(True)
        else:
            self.timer_camera.stop()
            self.videoCap.release()
            self.label_3.setText('视频窗口')
            self.pushButton_9.setText('打开摄像头')
            # 关闭自动采样
            self.radioButton.setChecked(False)
            self.radioButton.setEnabled(False)

    # 从视频流采样并显示
    def showVideo(self):
        flag, img = self.videoCap.read()

        show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        showImg = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        showImg = QPixmap.fromImage(showImg).scaled(self.label_3.width() - 10, self.label_3.height() - 10, Qt.KeepAspectRatio,
                         Qt.SmoothTransformation)
        self.label_3.setPixmap(showImg)

    def pushButton_8Clicked(self):
        if self.pushButton_8.text() == '打开文件夹':
            self.videoCapDir = QFileDialog.getExistingDirectory(self, '打开文件夹', '.')
            if self.videoCapDir == '':
                return
            for path in os.listdir(self.videoCapDir):
                fileExtension = os.path.splitext(path)[-1]
                if fileExtension != '.jpg' and fileExtension != '.png':
                    continue
                self.listWidget_2.addItem(os.path.join(self.videoCapDir, path))
            self.pushButton_8.setText('拍摄')
            if self.pushButton_9.text() == '关闭摄像头':
                self.radioButton.setEnabled(True)
        elif self.pushButton_8.text() == '拍摄':
            if self.timer_camera.isActive():
                flag, img = self.videoCap.read()
                filePath = os.path.join(self.videoCapDir, 'sampling_{}.jpg'.format(self.listWidget_2.count()))
                cv2.imwrite(filePath, img)
                self.listWidget_2.addItem(filePath)
                img = QPixmap(filePath)
                img = img.scaled(self.label_5.width() - 10, self.label_5.height() - 10, Qt.KeepAspectRatio,
                                 Qt.SmoothTransformation)
                self.label_5.setPixmap(img)
                self.loading_label(self.label_6)
                self.infoThread = getInfo(filePath)
                self.infoThread.signal.connect(self.showImageLabel)
                self.infoThread.start()
        else:
            return

    def loading_label(self, label):
        # 暂时缓冲
        loading_gif = QMovie('./resources/loading.gif')
        loading_gif.setScaledSize(QSize(min(label.width() - 10, 200), min(label.height() - 10, 200)))
        loading_gif.start()
        label.setMovie(loading_gif)

    def showImageLabel(self, info):
        '''
        :param info:
        :return:
        '''
        str = ''
        for label in info['label']:
            str += label + '\n'
        self.label_6.setText(str)

    # 文本识别界面l
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