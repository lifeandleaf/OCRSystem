import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, hidden_size=128, num_heads=8, drop_out=0.2):
        '''
        :param hidden_size: 中间层参数量
        :param num_heads: 注意力头个数
        :param drop_out: dropout参数
        '''
        super(SelfAttention, self).__init__()
        if hidden_size % num_heads is not 0:
            raise ValueError('Hidden size is not the multiple of num_heads, hidden_size: {} num_heads: {}'.format(hidden_size, num_heads))
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_hidden_size = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, self.hidden_size)
        self.key   = nn.Linear(hidden_size, self.hidden_size)
        self.value = nn.Linear(hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(drop_out)

    def transposeToScores(self, x):
        # x: [b, n, hidden_size]
        b, n, c = x.size()
        x = x.view(b, n, self.num_heads, c // self.num_heads)
        return x.permute(0, 2, 1, 3) # [b, num_heads, n, c']

    def forward(self, input, mask=None):
        if mask is not None:
            attention_mask = mask.unsqueeze(1).unsqueeze(2) # [b, seqlen] -> [b, 1, 1, seqlen]
            attention_mask = (1.0 - attention_mask) * -10000.0

        # 获取q, k, v
        q = self.transposeToScores(self.query(input))
        k = self.transposeToScores(self.key(input))
        v = self.transposeToScores(self.value(input))

        # [b, num_heads, n, c] * [b, num_heads, c, n] = [b, num_heads, n, n]
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_hidden_size)
        if mask is not None:
            attention_scores = attention_scores + attention_mask # 广播机制

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # [b, num_heads, n, n] * [b, num_heads, n, c] = [b, num_heads, n, c]
        output = torch.matmul(attention_probs, v)
        output = output.permute(0, 2, 1, 3).contiguous() # [b, n, num_heads, c]
        b, n, nh, c = output.size()
        output = output.view(b, n, -1)
        return  output


class CATT(nn.Module):

    def __init__(self, imgH=32, nclass=64, nc=1, nh=256, leakyRelu=False):
        '''
            nc: 1
            nh: rnn hidden, 256
        '''
        super(CATT, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x32
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x32
        convRelu(6, True)  # 512x1x33

        self.cnn = cnn
        self.attention = nn.Sequential(
            SelfAttention(hidden_size=512, num_heads=16),
            nn.Linear(512, nclass)
        )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # [b, c, w]
        conv = conv.permute(0, 2, 1)  # [b, w, c]

        # attention features
        output = self.attention(conv)
        return output