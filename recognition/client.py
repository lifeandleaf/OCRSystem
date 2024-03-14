from PIL import Image
import requests
import json
from tools import imageToBase64
import time
from tools import cropImage

def requestForImage(image: Image.Image, fmt='jpeg') -> dict:
    base64str = imageToBase64(image, fmt)
    data = {'image': base64str}
    res = requests.post('http://127.0.0.1:5000/', data=json.dumps(data))
    return json.loads(res.text)

if __name__ == '__main__':
    images = ['0020.jpg', '0050.jpg', '0190.jpg']
    # images = ['../Dir/sampling_9.jpg']
    for i in images:
        image = Image.open(i)
        start = time.time()
        info = requestForImage(image)
        end = time.time()
        print(end - start)
        print(info)
        # s_image = cropImage(image, info['points'][0])
        # s_image.show()


'''
[[[202, 255], [377, 260], [376, 305], [201, 300]], [[228, 154], [358, 157], [357, 205], [227, 202]]]
'''