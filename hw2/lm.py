import math

class UnigramLM:
    """Unigram language model with add-one smoothing."""

    def __init__(self, freq_dict, alpha=1.0):
        self.N = sum(freq_dict.values())
        self.alpha = alpha
        self.V = len(freq_dict)
        self.freq = freq_dict

    def logP(self, word):
        """Return log P(word) with add-one smoothing."""
        f = self.freq.get(word, 0)
        return math.log(f + self.alpha) - math.log(self.N + self.alpha * self.V)
