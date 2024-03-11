from PIL import Image
import requests
import json
from tools import imageToBase64

from tools import cropImage

def requestForImage(image: Image.Image, fmt='jpeg') -> dict:
    base64str = imageToBase64(image, fmt)
    data = {'image': base64str}
    res = requests.post('http://127.0.0.1:5000/', data=json.dumps(data))
    return json.loads(res.text)

if __name__ == '__main__':
    image = Image.open('./test3.jpg')
    info = requestForImage(image)
    print(info)
    s_image = cropImage(image, info['points'][0])
    s_image.show()
