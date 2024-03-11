import torch
from PIL import Image, ImageDraw
import numpy as np
import os
import sys
import pathlib
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__.parent))

from data_loader import get_transforms
from models import build_model
from post_processing import get_post_processing



class DBNet:
    def __init__(self, model_path: str, post_p_thre=0.7, gpu_id=None):
        '''
        :param model_path: 加载模型地址
        :param post_p_thre:
        :param gpu_id: 运行模型的gpu
        '''
        self.gpu_id = gpu_id
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device: ', self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False
        self.model = build_model(config['arch'])
        self.post_process = get_post_processing(config['post_processing'])
        self.post_process.box_thresh = post_p_thre
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = []
        for t in config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = get_transforms(self.transform)
        # 记录图像缩放比例
        self.w_ratio = 1
        self.h_ratio = 1

    def img_resize_32(self, image):
        '''
        :param image: 输入图像，pillow.Image格式
        :return:
        '''
        # 将图像缩放到长款为32的整数倍
        w, h = image.size
        w_ = (w // 32) * 32
        h_ = (h // 32) * 32
        image = image.resize((w_, h_))
        self.w_ratio = float(w_ / w)
        self.h_ratio = float(h_ / h)
        return image, w_, h_

    def predict(self, image, is_output_polygon=False):
        '''
        :param image: 输入图像，pillow.Image格式
        :param is_output_polygon: 是否在输出图像上绘制矩形框
        :param short_size:
        :return:
        '''
        image, w, h = self.img_resize_32(image)
        tensor = self.transform(image)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        batch = {'shape': [(h, w)]}
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            preds = self.model(tensor)
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            box_list, score_list = self.post_process(batch, preds, is_output_polygon=is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
        if box_list.shape[0] > 0:
            box_list = box_list / np.array((self.w_ratio, self.h_ratio))
            box_list = box_list.astype(np.int16)
        return box_list.tolist()

def show_res(image, polygon):
    draw = ImageDraw.Draw(image)
    for i in range(len(polygon)):
        draw.polygon([tuple(x) for x in polygon[i]], outline=(255, 0, 0))
    image.show()

if __name__ == '__main__':
    model = DBNet('./model_best.pth')
    image = Image.open('../0190.jpg')
    res = model.predict(image)
    print(res)
    show_res(image, res)


