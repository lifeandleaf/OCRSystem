from flask import Flask, request
from PIL import Image
import json
from tools import base64ToImage

app = Flask(__name__)

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
      # TODO:基于模型的识别算法
      info = {
         'status': True,
         'points': [
            [0, 0, 0, 30, 30, 30, 30, 0]
         ],
         'label': [
            'sdhasjkdas'
         ]
      }
      return info


if __name__ == '__main__':
   app.run(host='127.0.0.1', port=5000, debug=True)

