from io import BytesIO
import base64
from PIL import Image
import cv2
import numpy as np

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

def cropImage(img_pil, points):
    '''
    :param imgPath:原始图像路径
    :param points:矩形四个坐标点[x1, y1, x2, y2, x3, y3, x4, y4]
    :return: 返回裁剪好的图像 PIL格式
    '''
    # 首先读入img
    # img_pil = Image.open(imgPath)
    img = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    points = [(points[2*i], points[2*i+1]) for i in range(len(points) // 2)]
    # 找面积最小的矩形
    rect = cv2.minAreaRect(np.array(points))
    # 得到最小矩形的坐标
    box = cv2.boxPoints(rect)
    dest_w, dest_h = int(rect[1][0]), int(rect[1][1])
    # 标准化坐标到整数
    box = np.intp(box)
    # 获取最小矩形内对应坐标
    x, y, w, h = cv2.boundingRect(np.array(points))
    roi_img = img[y:y + h, x:x + w]
    sourcepoints = []
    for i in range(len(points)):
        sourcepoints.append((points[i][0] - x, points[i][1] - y))
    # 实例化tps
    tps = cv2.createThinPlateSplineShapeTransformer()
    # 源点集合，处理为合适的格式
    sourceshape = np.array(sourcepoints, np.int32)
    sourceshape = sourceshape.reshape(1, -1, 2)
    # opencv中匹配函数
    matches = []
    N = len(points)
    for i in range(0, N):
        matches.append(cv2.DMatch(i, i, 0))
    # 开始变动，获取目标点
    newpoints = []
    N = N // 2
    dx = int(w / (N - 1))
    for i in range(0, N):
        newpoints.append((dx * i, 2))
    for i in range(N - 1, -1, -1):
        newpoints.append((dx * i, h - 2))
    targetshape = np.array(newpoints, np.int32)
    targetshape = targetshape.reshape(1, -1, 2)
    # 估计插值矩阵，并进行tps插值获取插值后的图像
    tps.estimateTransformation(targetshape, sourceshape, matches)
    roi_img_ = tps.warpImage(roi_img)

    roi_img_ = cv2.resize(roi_img_, (max(dest_w, dest_h), min(dest_w, dest_h)), interpolation=cv2.INTER_NEAREST)
    ret = Image.fromarray(cv2.cvtColor(roi_img_, cv2.COLOR_BGR2RGB))
    return ret

if __name__ == '__main__':
    img = Image.open('./test.jpg')
    base64str = imageToBase64(img)
    # print(base64str)
    img = base64ToImage(base64str)
    img.show()