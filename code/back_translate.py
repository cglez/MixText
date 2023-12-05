import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm_notebook as tqdm
import os
import re
import pickle

from code.read_data import train_val_split

# split and get our unlabeled training data
"""
def train_val_split(labels, n_labeled_per_class, n_labels, seed=0):
    np.random.seed(seed)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(n_labels):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class: n_labeled_per_class + 10000])
        val_idxs.extend(idxs[-3000:])

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)
    return train_labeled_idxs, train_unlabeled_idxs, val_idxs
"""


# back translate using Russian as middle language
"""
def translate_ru(start, end, file_name):
    trans_result = {}
    for id in tqdm(range(start, end)):
        trans_result[idxs[id]] = ru2en.translate(
            en2ru.translate(train_text[idxs[id]], sampling=True, temperature=0.9), sampling=True, temperature=0.9
        )
        if id % 500 == 0:
            with open(file_name, 'wb') as f:
                pickle.dump(trans_result, f)
    with open(file_name, 'wb') as f:
        pickle.dump(trans_result, f)
"""


# back translate using German as middle language
def translate_de(file_name, train_text, idxs, device):
    if not hasattr(translate_de, 'forward_model') or hasattr(translate_de, 'back_model'):
        en2de = torch.hub.load(
            'pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe'
        )
        de2en = torch.hub.load(
            'pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe'
        )
        en2de.to(device)
        de2en.to(device)
        translate_de.forward_model = en2de
        translate_de.back_model = de2en

    trans_result = {}
    for id in tqdm(range(len(train_text))):
        trans_result[idxs[id]] = translate_de.back_model.translate(
            translate_de.forward_model.translate(train_text[idxs[id]], sampling=True, temperature=0.9),
            sampling=True, temperature=0.9
        )
        if id % 500 == 0:
            with open(file_name, 'wb') as f:
                pickle.dump(trans_result, f)
    with open(file_name, 'wb') as f:
        pickle.dump(trans_result, f)


def main(dataset: str, n_labeled=200, n_unlabeled=20_000):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("gpu num: ", n_gpu)

    # Load translation model
    #en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
    #ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')
    #en2ru.to(device)
    #ru2en.to(device)

    path = f'./data/{dataset}'
    train_df = pd.read_csv(path + '/train.csv', header=None)
    train_labels = [v - 1 for v in train_df[0]]
    train_text = [v for v in train_df[2]]
    print(train_df.head())
    print(train_text[0])
    print(dataset, len(train_text))
    print('num labels:', max(train_labels))

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(train_labels, n_labeled, n_unlabeled)
    print('unlabeled', len(train_unlabeled_idxs))
    print(train_unlabeled_idxs[0])

    translate_de(f'{path}/de_1.pkl', train_text, idxs=train_unlabeled_idxs, device=device)


if __name__ == "__main__":
    main(dataset='ohsumed', n_labeled=200)
