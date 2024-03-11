import torch
import itertools
import numpy as np


def ctc_decode(pred, blank_index=0):  # T * N * C
    arg_max = pred.argmax(dim=-1)  # T * N
    arg_max = arg_max.t()  # N * T
    arg_max = arg_max.to(device='cpu').numpy()
    pred_labels = []
    for line in arg_max:
        label = [k for k, g in itertools.groupby(line)]
        while blank_index in label:
            label.remove(blank_index)
        pred_labels.append(label)
    return pred_labels  # type: list

if __name__ == '__main__':
    pred = torch.tensor([[1, 2, 3], [1, 1, 1]], dtype=torch.float)
    target = torch.tensor([[1, 2, 3], [2, 1, 2]], dtype=torch.float)
