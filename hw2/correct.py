import math
from lm import UnigramLM
from weighted_edit import WeightedEdit, LOG_MIN

class SpellingCorrector:
    """Spelling corrector using noisy-channel model (unigram LM + weighted edit model)."""

    def __init__(self, lm: UnigramLM, we: WeightedEdit, vocab=None,
                 log_no_error=math.log(0.9)):
        self.lm = lm
        self.we = we
        self.vocab = set(vocab) if vocab else set(lm.freq.keys())
        self.log_no_error = log_no_error

    def correct(self, observed: str) -> str:
        """Return most likely true word given observed word."""
        observed = observed.lower()
        best_w, best_score = observed, LOG_MIN

        # Case 1: No error (word is correct)
        if observed in self.vocab:
            score = self.lm.logP(observed) + self.log_no_error
            best_w, best_score = observed, score

        # Case 2: Candidates from one edit
        cand_scores = {}
        for w, logp_edit in self.we.edits1_with_scores(observed):
            if w not in self.vocab:
                continue
            score = self.lm.logP(w) + logp_edit
            if score > cand_scores.get(w, LOG_MIN):
                cand_scores[w] = score

        # Select best candidate
        for w, s in cand_scores.items():
            if s > best_score:
                best_w, best_score = w, s

        return best_w
