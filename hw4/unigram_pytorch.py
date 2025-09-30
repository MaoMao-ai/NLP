"""Pytorch."""

import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional
from torch import nn
import matplotlib.pyplot as plt

FloatArray = NDArray[np.float64]


def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding


def loss_fn(logp: float) -> float:
    """Compute loss to maximize probability."""
    return -logp


class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()

        # construct uniform initial s
        s0 = np.ones((V, 1))
        self.s = nn.Parameter(torch.tensor(s0.astype(float)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # convert s to proper distribution p
        logp = torch.nn.LogSoftmax(0)(self.s)

        # compute log probability of input
        return torch.sum(input, 1, keepdim=True).T @ logp


def gradient_descent_example():
    """Demonstrate gradient descent."""
    # vocabulary: 26 letters + space + OOV
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

    # load training text
    try:
        text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()
    except LookupError:
        nltk.download("gutenberg")
        text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    tokens = [char for char in text]
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])
    x = torch.tensor(encodings.astype(float))

    # model
    model = Unigram(len(vocabulary))

    # ====== hyperparameters ======
    num_iterations = 400
    learning_rate = 0.05

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for _ in range(num_iterations):
        logp_pred = model(x)
        loss = loss_fn(logp_pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    # ====== compute optimal empirical distribution ======
    counts = encodings.sum(axis=1).reshape(-1)
    T = counts.sum()
    p_emp = (counts / T).astype(float)
    eps = 1e-12
    min_loss = float(-T * np.sum(p_emp * np.log(p_emp + eps)))

    # ====== learned distribution ======
    with torch.no_grad():
        logp = torch.nn.LogSoftmax(0)(model.s).detach().cpu().numpy().reshape(-1)
        p_learned = np.exp(logp)

    # ====== plot 1: probabilities ======
    idxs = np.arange(len(vocabulary))
    labels = [c if c is not None else "<OOV>" for c in vocabulary]

    plt.figure(figsize=(14, 4))
    width = 0.4
    plt.bar(idxs - width/2, p_emp, width=width, label="Optimal (empirical)")
    plt.bar(idxs + width/2, p_learned, width=width, label="Learned (final)")
    plt.xticks(idxs, labels, rotation=90)
    plt.ylabel("Probability")
    plt.title("Final token probabilities: Learned vs. Empirical")
    plt.legend()
    plt.tight_layout()
    plt.savefig("probabilities_comparison.png", dpi=160)

    # ====== plot 2: loss curve ======
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(losses)), losses, linewidth=2)
    plt.axhline(min_loss, linestyle="--", linewidth=2, label=f"Min possible loss = {min_loss:.0f}")
    plt.xlabel("Iteration")
    plt.ylabel("Negative log-likelihood")
    plt.title("Training loss vs. iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=160)

    print("Saved: probabilities_comparison.png, loss_curve.png")


if __name__ == "__main__":
    gradient_descent_example()
