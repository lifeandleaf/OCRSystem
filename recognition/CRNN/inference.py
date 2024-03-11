import numpy
import torch
import numpy as np
from PIL import Image
import os
import sys
import pathlib
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__.parent))

from model.CATT import CATT
from utils import ctc_decode

class Catt:
    def __init__(self, model_path: str, image_height: int, charList: str, gpu_id=None):
        '''
        :param model_path: 模型地址
        :param image_size: 输入图像大小
        '''
        self.gpu_id = gpu_id
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        self.model = CATT(imgH=image_height, nclass=len(charList)+1)
        self.charList = ['blank'] + [x for x in charList] + ['blank']
        self.H = image_height
        # 删除参数名前的module.
        # new_state_dict = {}
        # for k, v in torch.load(model_path, map_location=self.device).items():
        #     new_state_dict[k[7:]] = v

        new_state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

    def imageResize(self, image):
        '''
        :param image: 调整图像大小，高度为self.H，宽度等比缩放并保证是8倍数（卷积层在宽度上进行了三次二倍降采样）
        :return: image pillow.Image
        '''
        w, h = image.size
        ratio = self.H / h
        w_ = int(w * ratio)
        w_ = (w_ // 8) * 8
        image = image.resize((w_, self.H))
        return image

    def predict(self, image):
        '''
        :param image: 输入图像 pillow.Image格式
        :return:
        '''
        # 预处理
        image = self.imageResize(image.convert('L'))
        image = numpy.array(image)
        image = image / 255.0 * 2.0 - 1.0
        image = image[None, None, :, :]
        image = torch.from_numpy(image.astype(np.float32))
        preds = self.model(image)
        preds = preds.permute(1, 0, 2)
        preds = ctc_decode(preds)[0]

        # 解码模型输出->字符串
        res = ''
        for i in preds:
            if self.charList[i] != 'blank':
                res += self.charList[i]
        return res

if __name__ == '__main__':
    charList = '0123456789abcdefghijklmnopqrstuvwxyz'
    model = Catt(model_path='./trained_weights/expr_CATT_epoch50/best.pth', image_height=32, charList=charList)
    image = Image.open('../crnn.png')
    ans = model.predict(image)
    print(ans)