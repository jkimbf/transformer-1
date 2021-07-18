from data import *
from train import evaluate

import torch
import torch.nn as nn


def compute_metrics():
    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
    model = torch.load('./saved/model-saved.pt', map_location=torch.device('cpu'))
    _, bleu, rouge = evaluate(model, valid_iter, criterion)

    print(f'\tBLEU Score: {bleu:.3f}')
    print(f'\tRouge Recall: {rouge:.3f}')


if __name__ == '__main__':
    compute_metrics()
