import torch
import torch.nn
import re

model = torch.load('./crnn.pth')
del_list = []
for key in model:
    if re.match(r'rnn.*', key) is not None:
        del_list.append(key)

for key in del_list:
    model.pop(key)

for key in model:
    print(key)

torch.save(model, 'crnn_without_rnn.pth')