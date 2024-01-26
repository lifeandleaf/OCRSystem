from io import BytesIO
import base64
from PIL import Image

def base64ToImage(base64str: str) -> Image.Image:
    '''
    :param base64str: The base64 data of image.
    :return: A pillow image
    '''
    byte_data = base64.b64decode(base64str)
    image_data = BytesIO(byte_data)
    image = Image.open(image_data)
    return image

def imageToBase64(image: Image.Image, fmt='jpeg') -> str:
    '''
    :param image:pillow image
    :param fmt:文件格式，png or jpeg or gif ...
    :return:base64str of this image
    '''
    output_buffer = BytesIO()
    image.save(output_buffer, format=fmt)
    byte_data = output_buffer.getvalue()
    base64str = base64.b64encode(byte_data).decode('utf-8')
    return base64str

if __name__ == '__main__':
    img = Image.open('./test.jpg')
    base64str = imageToBase64(img)
    print(base64str)
    img = base64ToImage(base64str)
    img.show()