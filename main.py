import sys, os, json
import numpy
from PIL import Image, ImageDraw
import cv2


from base import Ui_OCR
from aug import Ui_Form
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QListWidgetItem, QListView, QInputDialog, QLineEdit
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPixmap, QIcon, QImage, QMovie
from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal

from tools import Recognize, cropImage, augImage, Detect

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
        print(self.imgPath)
        info = Recognize(self.imgPath)
        with open(self.imgPath.replace(fileExtension, '.json'), 'w') as f:
            f.write(json.dumps(info))
        self.signal.emit(info)

class AugWindow(QWidget, Ui_Form):
    def __init__(self):
        QWidget.__init__(self)
        self.setupUi(self)
        # init window
        self.label.setAlignment(Qt.AlignCenter)
        self.listWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.listWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.listWidget.setViewMode(QListView.IconMode)
        self.setFixedSize(800, 600)

        # 绑定接口
        self.listWidget.itemDoubleClicked.connect(self.listWidget_ItemClicked)
        self.pushButton.clicked.connect(self.pushButtonClicked)
        self.pushButton_2.clicked.connect(self.pushButton_2Clicked)

    def Image2Pixmap(self, image: Image) -> QPixmap:
        img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
        h, w, d = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QImage(img.data, w, h, w * d, QImage.Format_RGB888)
        return QPixmap.fromImage(img)

    def showInLabel(self, imgPath) -> None:
        # 添加增广图像显示在列表中
        self.listWidget.clear()
        self.augImages = [{'name': '原图', 'image': Image.open(imgPath)}] + augImage(imgPath)
        for dict in self.augImages:
            key = dict['name']
            img = dict['image']
            img = QPixmap(self.Image2Pixmap(img)).scaled(self.listWidget.width() - 10, self.listWidget.height() - 10, Qt.KeepAspectRatio,
                         Qt.SmoothTransformation)
            item = QListWidgetItem()
            item.setIcon(QIcon(img))
            item.setText(key)
            self.listWidget.setIconSize(QSize(min(self.listWidget.width() - 50, 200), self.listWidget.height() - 50))
            self.listWidget.setResizeMode(QListView.Adjust)
            self.listWidget.addItem(item)
        if self.listWidget.count() <= 0:
            return
        self.listWidget.setCurrentRow(0)
        self.listWidget_ItemClicked()

    def listWidget_ItemClicked(self):
        row = self.listWidget.currentRow()
        img = QPixmap(self.Image2Pixmap(self.augImages[row]['image'])).scaled(self.label.width() - 10, self.label.height() - 10, Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
        self.label.setPixmap(img)

    def pushButtonClicked(self):
        row = self.listWidget.currentRow()
        if row <= 0:
            return
        row -= 1
        self.listWidget.setCurrentRow(row)
        self.listWidget_ItemClicked()

    def pushButton_2Clicked(self):
        row = self.listWidget.currentRow()
        if row + 1 >= self.listWidget.count():
            return
        row += 1
        self.listWidget.setCurrentRow(row)
        self.listWidget_ItemClicked()

class MyWindow(QWidget, Ui_OCR):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)
        self.setupUi(self)
        # init window
        self.stackedWidget.setCurrentIndex(1)
        self.progressBar.reset()
        self.timer_camera = QTimer()
        self.videoDetectCount = 0
        self.videoDetectInfo = []
        self.videoCap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.radioButton.setEnabled(False)
        self.videoCapDir = ''
        self.augWindow = AugWindow()
        # 设置窗体无边框
        # self.setWindowFlags(Qt.FramelessWindowHint)
        # 设置背景透明
        # self.setAttribute(Qt.WA_TranslucentBackground)
        # 绑定回调函数
        self.timer_camera.timeout.connect(self.showVideo)

        self.pushButton.clicked.connect(self.pushButtonClicked)
        self.pushButton_2.clicked.connect(self.pushButton_2Clicked)
        self.pushButton_3.clicked.connect(self.pushButton_3Clicked)

        self.pushButton_6.clicked.connect(self.pushButton_6Clicked)
        self.pushButton_7.clicked.connect(self.pushButton_7Clicked)
        self.pushButton_8.clicked.connect(self.pushButton_8Clicked)
        self.pushButton_9.clicked.connect(self.pushButton_9Clicked)
        self.pushButton_10.clicked.connect(self.pushButton_10Clicked)
        self.pushButton_11.clicked.connect(self.pushButton_11Clicked)
        self.pushButton_12.clicked.connect(self.pushButton_12Clicked)
        self.pushButton_13.clicked.connect(self.pushButton_13Clicked)
        self.pushButton_14.clicked.connect(self.pushButton_14Clicked)
        self.pushButton_15.clicked.connect(self.pushButton_15Clicked)

        self.listWidget_2.itemDoubleClicked.connect(self.listWidget_2ItemClicked)
        self.listWidget_3.itemDoubleClicked.connect(self.listWidget_3ItemClicked)
        self.listWidget_4.itemDoubleClicked.connect(self.listWidget_4ItemClicked)
        self.listWidget_5.itemClicked.connect(self.listWidget_5ItemClicked)
        self.listWidget_5.itemDoubleClicked.connect(self.listWidget_5ItemDoubleClicked)

        self.show()

    def Image2Pixmap(self, image: Image) -> QPixmap:
        img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
        h, w, d = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QImage(img.data, w, h, w * d, QImage.Format_RGB888)
        return QPixmap.fromImage(img)

    # 图像采集界面
    def pushButtonClicked(self):
        self.stackedWidget.setCurrentIndex(1)

    def pushButton_6Clicked(self):
        self.label_5.clear()
        self.label_6.clear()
        self.listWidget_2.clear()
        if self.pushButton_9.text() == '关闭摄像头':
            self.pushButton_9Clicked()
        if self.pushButton_8.text() == '拍摄':
            self.pushButton_8.setText('打开文件夹')
        self.radioButton.setEnabled(False)

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
                self.videoDetectCount = 0
                self.timer_camera.start(30)
                self.pushButton_9.setText('关闭摄像头')
                self.radioButton.setEnabled(True)
        else:
            self.timer_camera.stop()
            self.videoCap.release()
            self.videoDetectInfo = []
            self.videoDetectCount = 0
            self.label_3.setText('视频窗口')
            self.pushButton_9.setText('打开摄像头')
            # 关闭自动采样
            self.radioButton.setChecked(False)
            self.radioButton.setEnabled(False)

    # 从视频流采样并显示
    def showVideo(self):
        flag, img = self.videoCap.read()
        self.videoDetectCount = (self.videoDetectCount + 1) % 10
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.radioButton.isChecked() and self.videoDetectCount == 0:
            self.videoDetectInfo = Detect(img)['points']
        draw = ImageDraw.Draw(img)
        if len(self.videoDetectInfo) > 0:
            for i in range(len(self.videoDetectInfo)):
                tmp = []
                for ix in range(len(self.videoDetectInfo[i]) // 2):
                    tmp.append((self.videoDetectInfo[i][ix * 2], self.videoDetectInfo[i][ix * 2 + 1]))
                draw.polygon(tmp, outline=(255, 0, 0))
        img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
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
                self.listWidget_2.addItem(QListWidgetItem(QIcon('./resources/file_icon.png'), os.path.join(self.videoCapDir, path)))
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
        loading_gif = QMovie('./resources/loading_2.gif')
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
            if path.endswith('.jpg') or path.endswith('.png'):
                fileListFilter.append(os.path.join(dir, path))
        self.listWidget_3.clear()
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
            self.listWidget_3.addItem(QListWidgetItem(QIcon('./resources/file_icon.png'), item))

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
                cropImg = QPixmap(self.Image2Pixmap(cropImg))
                cropImg = cropImg.scaled(self.listWidget_4.width()-10, self.listWidget_4.height()-10, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                item = QListWidgetItem()
                item.setIcon(QIcon(cropImg))
                self.listWidget_4.setIconSize(QSize(self.listWidget_4.width()-10, self.listWidget_4.height()-10))
                self.listWidget_4.setResizeMode(QListView.Adjust)
                self.listWidget_4.addItem(item)

        # 标签获取
        if 'label' in info:
            if len(info['label']) > 0:
                self.label_7.setText(info['label'][0])
            else:
                self.label_7.setText("")

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

    def pushButton_14Clicked(self):
        dir = QFileDialog.getExistingDirectory(self, '打开文件夹', '.')
        if dir == '':
            return
        fileList = os.listdir(dir)
        fileListFilter = []
        for path in fileList:
            if path.endswith('.jpg') or path.endswith('.png'):
                fileListFilter.append(os.path.join(dir, path))
        self.listWidget_5.clear()
        items = [QListWidgetItem(QIcon('./resources/file_icon.png'), x) for x in fileListFilter]
        [self.listWidget_5.addItem(item) for item in items]

    def listWidget_5ItemClicked(self):
        row = self.listWidget_5.currentRow()
        filePath = self.listWidget_5.item(row).text()
        img = QPixmap(filePath)
        img = img.scaled(self.label_2.width()-10, self.label_2.height()-10, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_2.setPixmap(img)

    def listWidget_5ItemDoubleClicked(self):
        row = self.listWidget_5.currentRow()
        imgPath = self.listWidget_5.item(row).text()
        self.augWindow.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.augWindow.setWindowModality(Qt.ApplicationModal)
        self.augWindow.show()
        self.augWindow.showInLabel(imgPath)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MyWindow = MyWindow()
    sys.exit(app.exec())