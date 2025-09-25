#!/usr/bin/env python3
"""
tag_with_viterbi.py
Loads HMM (pi, A, B) and mappings, then runs Viterbi on Brown sentences 10150-10152
with the Universal tagset. Reports token-level predictions vs. gold and accuracy.
Compatible with numpy + standard library; uses nltk only for gold/reference data.
"""

import argparse
import json
import numpy as np
import nltk

def load_model(out_dir):
    data = np.load(f"{out_dir}/hmm_model.npz")
    with open(f"{out_dir}/mappings.json","r",encoding="utf-8") as f:
        mappings = json.load(f)
    pi = data["pi"]
    A = data["A"]
    B = data["B"]
    return pi, A, B, mappings

def words_to_ids(words, word2idx):
    unk = word2idx["<UNK>"]
    return [word2idx.get(w.lower(), unk) for w in words]

from typing import Sequence
import numpy as np


def viterbi(
    obs: Sequence[int],
    pi: np.ndarray[tuple[int], np.dtype[np.float64]],
    A: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    B: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> tuple[list[int], float]:
    """Infer most likely state sequence using the Viterbi algorithm.

    Q is the number of possible states.
    V is the number of possible observations.
    N is the length of the observation sequence.


    Args:
        obs: A length-N sequence of ints representing observations.
        pi: A length-Q numpy array of floats representing initial state probabilities.
        A: A Q-by-Q numpy array of floats representing state transition probabilities.
        B: A Q-by-V numpy array of floats representing emission probabilities.

    Returns:
        A tuple of:
        * A 1D numpy array of ints representing the most likely state sequence.
        * A float representing the probability of the most likely state sequence.
    """
    N = len(obs)
    Q, V = B.shape  # num_states, num_observations

    # d_{ti} = max prob of being in state i at step t
    #   AKA viterbi
    # \psi_{ti} = most likely state preceeding state i at step t
    #   AKA backpointer

    # initialization
    log_d = [np.log(pi) + np.log(B[:, obs[0]])]
    log_psi = [np.zeros((Q,))]

    # recursion
    for z in obs[1:]:
        log_da = np.expand_dims(log_d[-1], axis=1) + np.log(A)
        log_d.append(np.max(log_da, axis=0) + np.log(B[:, z]))
        log_psi.append(np.argmax(log_da, axis=0))

    # termination
    log_ps = np.max(log_d[-1])
    qs = [-1] * N
    qs[-1] = int(np.argmax(log_d[-1]))
    for i in range(N - 2, -1, -1):
        qs[i] = log_psi[i + 1][qs[i + 1]]

    return qs, np.exp(log_ps)

def evaluate_on_brown(pi, A, B, mappings):

    nltk.download("brown", quiet=True)
    nltk.download("universal_tagset", quiet=True)
    sents = nltk.corpus.brown.tagged_sents(tagset='universal')
    eval_slice = sents[10150:10153]

    tag2idx = mappings["tag2idx"]
    idx2tag = [None] * len(tag2idx)
    for t, i in tag2idx.items():
        idx2tag[i] = t

    word2idx = mappings["word2idx"]

    total, correct = 0, 0
    detailed = []

    for sid, sent in enumerate(eval_slice, start=10150):
        words = [w for (w, t) in sent]
        gold_tags = [t for (w, t) in sent]
        obs = words_to_ids(words, word2idx)


        pred_states, path_prob = viterbi(obs, pi, A, B)


        pred_tags = []
        for s in pred_states:
            if isinstance(s, (int, np.integer)) and s < len(idx2tag):
                pred_tags.append(idx2tag[s])
            else:
                pred_tags.append("<UNK_TAG>")
                print(f"[Warning] State {s} out of range for idx2tag (len={len(idx2tag)})")


        for w, g, p in zip(words, gold_tags, pred_tags):
            total += 1
            correct += int(g == p)

 
        detailed.append({
            "sentence_index": sid,
            "tokens": words,
            "gold": gold_tags,
            "pred": pred_tags,
            "path_prob": path_prob
        })

    acc = correct / total if total else 0.0
    return acc, detailed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./outputs")
    parser.add_argument("--save_report", type=str, default="./viterbi_report.json")
    args = parser.parse_args()

    pi, A, B, mappings = load_model(args.model_dir)
    acc, detailed = evaluate_on_brown(pi, A, B, mappings)

    out = {
        "accuracy": acc,
        "details": detailed
    }
    with open(args.save_report, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Token accuracy on Brown[10150:10153]: {acc:.4f}")
    print(f"Saved detailed report to {args.save_report}")

if __name__ == "__main__":
    main()
