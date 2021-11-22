# This file aims to clean the text data

import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords
from collections import Counter
from tqdm import tqdm
import re
import joblib
import numpy as np

dataset = "mr"

# param
stop_words = set(stopwords.words('english'))
least_freq = 100
if dataset == "mr" or "SST" in dataset:
    # stop_words = set()
    least_freq = 0


# func load texts & labels
def load_dataset(dataset):
    with open(f"corpus/{dataset}.texts.txt", "r", encoding="latin1") as f:
        texts = f.read().strip().split("\n")
    with open(f"corpus/{dataset}.labels.txt", "r") as f:
        labels = f.read().strip().split("\n")
    return texts, labels


def filter_text(text: str):
    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", text)
    text = text.replace("'ll ", " will ")
    text = text.replace("'d ", " would ")
    text = text.replace("'m ", " am ")
    text = text.replace("'s ", " is ")
    text = text.replace("'re ", " are ")
    text = text.replace("'ve ", " have ")
    text = text.replace(" can't ", " can not ")
    text = text.replace(" ain't ", " are not ")
    text = text.replace("n't ", " not ")
    text = text.replace(",", " , ")
    text = text.replace("!", " ! ")
    text = text.replace("(", " ( ")
    text = text.replace(")", " ) ")
    text = text.replace("?", " ? ")
    text = re.sub(r"\s{2,}", " ", text)
    return " ".join(text.strip().split())


if __name__ == '__main__':
    texts, labels = load_dataset(dataset)

    # handle texts
    texts_clean = [filter_text(t) for t in texts]

    word2count = Counter([w for t in texts_clean for w in t.split()])
    word_count = [[w, c] for w, c in word2count.items() if c >= least_freq and w not in stop_words]
    word2index = {w: i for i, (w, c) in enumerate(word_count)}

    words_list = [[w for w in t.split() if w in word2index] for t in texts_clean]

    texts_remove = [" ".join(ws) for ws in words_list]

    # labels 2 targets
    label2index = {l: i for i, l in enumerate(set(labels))}
    targets = [label2index[l] for l in labels]

    # save
    with open(f"temp/{dataset}.texts.clean.txt", "w") as f:
        f.write("\n".join(texts_clean))

    with open(f"temp/{dataset}.texts.remove.txt", "w") as f:
        f.write("\n".join(texts_remove))

    np.save(f"temp/{dataset}.targets.npy", targets)
    joblib.dump(word2index, f"temp/{dataset}.word2index.pkl")
########################
    word2index = joblib.load(f"temp/{dataset}.word2index.pkl")
    with open(f"temp/{dataset}.texts.remove.txt", "r") as f:
        texts = f.read().strip().split("\n")

    BoW = np.zeros((len(texts), len(word2index)))
    for i, text in tqdm(enumerate(texts)):
        words = [word2index[w] for w in text.split()]
        for j in range(len(words)):
            BoW[i][words[j]] += 1

    np.save(f"temp/{dataset}.BoW.npy", BoW)