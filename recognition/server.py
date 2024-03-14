from flask import Flask, request
import json
from tools import base64ToImage, cropImage

from DBNet.inference import DBNet
from CRNN.inference import Catt

app = Flask(__name__)
dbnet_ = DBNet('./DBNet/model_best.pth')
miniDb = DBNet('./DBNet/model_best.pth', fast=True)
catt_ = Catt(model_path='./CRNN/trained_weights/expr_CATT_epoch50/best.pth', image_height=32,
              charList='0123456789abcdefghijklmnopqrstuvwxyz')

ERRORRET = {'state': False, 'points':[], 'label':[]}
@app.route('/', methods=['POST'])
def recognition():
   '''
   接收post的dict：
   :dict: {'image': ...base64...}
   :return: 识别结果，一个dict:{'state': True or False, 'points': [[x1,y1,x2,y2,x3,y3,x4,y4], [...], ...], 'label':[['123123'], [...], ...]}
   '''
   if request.method == 'POST':
      data = json.loads(request.data)
      if 'image' not in data:
         return ERRORRET
      image = base64ToImage(data['image'])
      # 基于模型的识别算法
      info = {
         'status': True,
         'points': [],
         'label': []
      }
      res_d = dbnet_.predict(image)
      for box in res_d:
         point = []
         for p in box:
            point = point + p
         s_image = cropImage(image, point)
         res_r = catt_.predict(s_image)
         info['points'].append(point)
         info['label'].append(res_r)
      return info
   return ERRORRET

@app.route('/dbnet', methods=['POST'])
def detect():
   if request.method == 'POST':
      data = json.loads(request.data)
      if 'image' not in data:
         return ERRORRET
      image = base64ToImage(data['image'])
      info = {
         'status': True,
         'points': [],
         'label': []
      }
      res_d = miniDb.predict(image)
      for box in res_d:
         point = []
         for p in box:
            point = point + p
         info['points'].append(point)
      return info
   return ERRORRET

if __name__ == '__main__':
   app.run(host='127.0.0.1', port=5000, debug=False)

