from data import *

import torch
import torch.nn as nn

from train import evaluate
from models.model.transformer import Transformer


def compute_metrics():
    model = Transformer(src_pad_idx=src_pad_idx,
                        trg_pad_idx=trg_pad_idx,
                        trg_sos_idx=trg_sos_idx,
                        d_model=d_model,
                        enc_voc_size=enc_voc_size,
                        dec_voc_size=dec_voc_size,
                        max_len=max_len,
                        ffn_hidden=ffn_hidden,
                        n_head=n_heads,
                        n_layers=n_layers,
                        drop_prob=0.00,
                        device=device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)
    model.load_state_dict(torch.load("./saved/model-saved.pt", map_location=torch.device('cpu')))
    _, bleu, rouge = evaluate(model, valid_iter, criterion)

    print(f'\tBLEU Score: {bleu:.3f}')
    print(f'\tRouge Recall: {rouge:.3f}')


if __name__ == '__main__':
    compute_metrics()
