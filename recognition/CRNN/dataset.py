import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from skimage import io, transform, color
import numpy as np

# 中文数据集构建
class TextDataset(Dataset):
    def __init__(self, txt_file, root_dir, max_label_length, transform=None):
        self.txt_frame = read_from_txt(txt_file)
        self.root_dir = root_dir
        self.transform = transform
        self.max_label_length = max_label_length

    def __len__(self):
        return len(self.txt_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.txt_frame[idx][0])
        img = io.imread(img_name)
        label = self.txt_frame[idx][1]

        label_length = len(label)
        [label.append(-1) for _ in range(len(label), self.max_label_length)]

        sample = {'image': img, 'label': label, 'label_length': label_length}
        if self.transform:
            sample = self.transform(sample)
        return sample


class CharClasses():
    def __init__(self, txt_file):
        def read_txt(txt_file):
            with open(txt_file, 'r') as f:
                chars = [char.strip('\n') for char in f.readlines()]
                return chars
        self.chars = read_txt(txt_file)

# 读取键值对（中文数据集）
def read_from_txt(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        frame = [(line.strip('\n').split(' ')[0], [int(l) for l in line.strip('\n').split(' ')[1:]]) for line in lines]
        return frame


class ToTensor(object):
    def __call__(self, sample):
        image, label, label_length = sample['image'], sample['label'], sample['label_length']
        if len(image.shape) < 3:
            image = image[:, :, np.newaxis]
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image.astype(np.float32)),
                'label': torch.from_numpy(np.array(label, dtype=np.float32)),
                'label_length': torch.from_numpy(np.array(label_length, dtype=np.float32))}

# 归一化（-1~1）
class ZeroMean(object):
    def __call__(self, sample):
        image, label, label_length = sample['image'], sample['label'], sample['label_length']
        image = image / 255.0 * 2.0 - 1.0

        return {'image': image, 'label': label, 'label_length': label_length}

# 图像缩放
class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, label_length = sample['image'], sample['label'], sample['label_length']
        new_h, new_w = self.output_size
        image = transform.resize(image=image, output_shape=(new_h, new_w))
        return {'image': image, 'label': label, 'label_length': label_length}

# 灰度化
class Gray(object):
    def __call__(self, sample):
        image, label, label_length = sample['image'], sample['label'], sample['label_length']
        image = color.rgb2gray(image) * 255.0

        return {'image': image, 'label': label, 'label_length': label_length}

# 随机反色
class RandomConvert(object):
    '''
    随机变成反色, 因为按道理来说无论是黑底白字,还是白底黑字都应该认识
    '''
    def __call__(self, sample):
        image, label, label_length = sample['image'], sample['label'], sample['label_length']
        if np.random.randint(0, 10) < 5:
            image = 255.0 - image

        return {'image': image, 'label': label, 'label_length': label_length}

# 英文数据集构建
class AlpDataset(Dataset):
    def __init__(self, txt_file, root_dir, max_label_length, char_list, transform=None):
        self.txt_frame = read_from_txt_alp(txt_file)
        self.root_dir = root_dir
        self.transform = transform
        self.max_label_length = max_label_length
        self.char_dict = {}
        for i in range(len(char_list)):
            self.char_dict[char_list[i]] = i

    def __len__(self):
        return len(self.txt_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.txt_frame[idx][0])
        img = io.imread(img_name)
        labeltext = self.txt_frame[idx][1]

        label = [self.char_dict[x] for x in labeltext.lower()]
        label_length = len(label)
        [label.append(-1) for _ in range(len(label), self.max_label_length)]

        sample = {'image': img, 'label': label, 'label_length': label_length}
        if self.transform:
            sample = self.transform(sample)
        return sample

# 读取键值对（英文数据集）
def read_from_txt_alp(txt_file):
    '''
        格式：
            path label
            path label
            ...
    '''
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        frame = [(line.strip('\n').split(' ')[0], line.strip('\n').split(' ')[1]) for line in lines]
        return frame

if __name__ == '__main__':
    chars = CharClasses('./IIIT5k/charlist.txt')
    dataset = AlpDataset(txt_file='./IIIT5k/traindata.txt',
                          root_dir='./IIIT5k/',
                          max_label_length=30,
                          char_list=chars.chars,
                          transform=transforms.Compose([Gray(), ToTensor()]))
    data = dataset[0]
    image = data['image']
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    label = data['label']
    print(label)
    text = [chars.chars[num] for num in label.numpy().astype(np.int32)]
    print(text)
    data_t = ZeroMean()(data)
