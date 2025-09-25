#!/usr/bin/env python3
"""
train_hmm_pos.py
Builds a POS Hidden Markov Model (HMM) from the first 10k tagged sentences of the Brown corpus
with the Universal tagset. Uses add-1 (Laplace) smoothing everywhere and includes an UNK token for
emissions. Outputs:
- hmm_model.npz: transition (A), emission (B), initial (pi) matrices/vectors
- mappings.json: word2idx, idx2word, tag2idx, idx2tag
Requires: numpy, nltk (for data loading only)
"""

import argparse
import json
import numpy as np
from collections import Counter, defaultdict
import nltk

def load_brown_universal(n_train=10000):
    sents = nltk.corpus.brown.tagged_sents(tagset='universal')
    train = sents[:n_train]
    return train

def build_vocab_and_tags(train_sents, min_count=1):
    word_counts = Counter()
    tag_set = set()
    for sent in train_sents:
        for w, t in sent:
            word_counts[w.lower()] += 1
            tag_set.add(t)
    # Include UNK in vocabulary
    vocab = [w for w,c in word_counts.items() if c >= min_count]
    vocab = sorted(vocab)
    vocab = ['<UNK>'] + vocab
    tags = sorted(tag_set)
    word2idx = {w:i for i,w in enumerate(vocab)}
    idx2word = {i:w for w,i in word2idx.items()}
    tag2idx = {t:i for i,t in enumerate(tags)}
    idx2tag = {i:t for t,i in tag2idx.items()}
    return word2idx, idx2word, tag2idx, idx2tag

def sent_words_tags(sent):
    words = [w.lower() for (w,t) in sent]
    tags  = [t for (w,t) in sent]
    return words, tags

def train_hmm(train_sents, word2idx, tag2idx, add_k=1.0):
    N = len(tag2idx)      # number of states (tags)
    M = len(word2idx)     # number of observations (words incl. UNK)

    # Count matrices
    init_counts = np.zeros(N, dtype=np.float64)
    trans_counts = np.zeros((N, N), dtype=np.float64)
    emit_counts = np.zeros((N, M), dtype=np.float64)

    for sent in train_sents:
        words, tags = sent_words_tags(sent)
        # initial
        init_counts[tag2idx[tags[0]]] += 1
        # emissions
        for w, t in zip(words, tags):
            wid = word2idx.get(w, word2idx['<UNK>'])
            tid = tag2idx[t]
            emit_counts[tid, wid] += 1
        # transitions
        for t_prev, t_next in zip(tags[:-1], tags[1:]):
            i = tag2idx[t_prev]
            j = tag2idx[t_next]
            trans_counts[i, j] += 1

    # Add-1 smoothing everywhere (Laplace)
    # Initial distribution pi
    pi = (init_counts + add_k) / (init_counts.sum() + add_k * N)

    # Transition matrix A
    A = (trans_counts + add_k) / ((trans_counts.sum(axis=1, keepdims=True)) + add_k * N)

    # Emission matrix B
    B = (emit_counts + add_k) / ((emit_counts.sum(axis=1, keepdims=True)) + add_k * M)

    return pi, A, B

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=10000, help="Number of Brown sentences for training")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Where to write model files")
    args = parser.parse_args()

    nltk.download("brown", quiet=True)
    nltk.download("universal_tagset", quiet=True)

    train = load_brown_universal(args.n_train)
    word2idx, idx2word, tag2idx, idx2tag = build_vocab_and_tags(train, min_count=1)

    pi, A, B = train_hmm(train, word2idx, tag2idx, add_k=1.0)

    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(f"{args.out_dir}/hmm_model.npz", pi=pi, A=A, B=B)

    mappings = {
        "word2idx": word2idx,
        "idx2word": {int(k):v for k,v in idx2word.items()},
        "tag2idx": tag2idx,
        "idx2tag": {int(k):v for k,v in idx2tag.items()},
    }
    with open(f"{args.out_dir}/mappings.json", "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)

    print("Saved HMM model to", f"{args.out_dir}/hmm_model.npz")
    print("Saved mappings to", f"{args.out_dir}/mappings.json")
    print("Shapes: pi", pi.shape, "A", A.shape, "B", B.shape)

if __name__ == "__main__":
    import os
    main()
