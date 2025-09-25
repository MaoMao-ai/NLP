"""Viterbi Algorithm for inferring the most likely sequence of states from an HMM.

Patrick Wang, 2025
"""

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
