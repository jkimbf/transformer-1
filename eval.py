from data import *
from rouge import Rouge

import torch
import torch.nn as nn

from util.bleu import idx_to_word, get_bleu



def eval(model, iterator):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    batch_rouge = []
    rouge = Rouge()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            total_bleu = []
            total_rouge = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)

                    rouge_score = rouge.get_scores(output_words, trg_words)
                    total_rouge.append(rouge_score[0]['rouge-l']['r'])  # getting recall value
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

            total_rouge = sum(total_rouge) / len(total_rouge)
            batch_rouge.append(total_rouge)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    batch_rouge = sum(batch_rouge) / len(batch_rouge)

    return epoch_loss / len(iterator), batch_bleu, batch_rouge


def compute_metrics():
    model = torch.load('./saved/model-saved.pt', map_location=torch.device('cpu'))
    _, bleu, rouge = eval(model, valid_iter)

    print(f'\tBLEU Score: {bleu:.3f}')
    print(f'\tRouge Recall: {rouge:.3f}')


if __name__ == '__main__':
    compute_metrics()
