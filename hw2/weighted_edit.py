import math

LOG_MIN = -30.0  # log probability floor

def _safe_log(p):
    return math.log(p) if p > 0 else LOG_MIN

class WeightedEdit:
    """
    Weighted edit distance model with probabilities derived from confusion tables.
    Supports single edit: substitution, insertion, deletion, transpose.
    """

    def __init__(self, subs, subs_base, adds, adds_base, dels, dels_base,
                 bigram, bigram_total, add_smooth=0.1, del_smooth=0.1,
                 sub_smooth=0.1, trans_p=1e-5):
        self.subs = subs
        self.subs_base = subs_base
        self.adds = adds
        self.adds_base = adds_base
        self.dels = dels
        self.dels_base = dels_base
        self.bigram = bigram
        self.bigram_total = bigram_total
        self.add_smooth = add_smooth
        self.del_smooth = del_smooth
        self.sub_smooth = sub_smooth
        self.trans_p = trans_p

    def edits1_with_scores(self, observed):
        """Generate candidate true words within one edit distance, with log probabilities."""
        L = len(observed)
        results = []
        alphabet = self._alphabet_from_tables()

        # Substitutions
        for i in range(L):
            x_ch = observed[i]
            for w_ch in alphabet:
                if w_ch == x_ch:
                    continue
                w = observed[:i] + w_ch + observed[i+1:]
                logp = self._logP_sub(w_ch, x_ch)
                results.append((w, logp))

        # Insertions (observed has one extra char)
        for i in range(L):
            prev = observed[i-1] if i > 0 else '^'
            ins = observed[i]
            w = observed[:i] + observed[i+1:]
            logp = self._logP_ins(prev, ins)
            results.append((w, logp))
        if L > 0:
            prev = observed[L-1]
            ins = observed[-1]
            w = observed[:-1]
            logp = self._logP_ins(prev, ins)
            results.append((w, logp))

        # Deletions (observed missing one char)
        for i in range(L+1):
            prev = observed[i-1] if i > 0 else '^'
            for ch in alphabet:
                w = observed[:i] + ch + observed[i:]
                logp = self._logP_del(prev, ch)
                results.append((w, logp))

        # Transpositions
        for i in range(L-1):
            if observed[i] != observed[i+1]:
                w = observed[:i] + observed[i+1] + observed[i] + observed[i+2:]
                results.append((w, math.log(self.trans_p)))

        return results

    def _alphabet_from_tables(self):
        chars = set()
        for (a, b) in list(self.adds.keys()) + list(self.dels.keys()):
            chars.add(a)
            chars.add(b)
        for (f, t) in self.subs.keys():
            chars.add(f)
            chars.add(t)
        if '^' in chars:
            chars.remove('^')
        return chars if chars else set("abcdefghijklmnopqrstuvwxyz")

    def _logP_sub(self, gold, obs):
        num = self.subs.get((gold, obs), 0) + self.sub_smooth
        den = self.subs_base.get(gold, 0) + self.sub_smooth * max(1, len(self._alphabet_from_tables()))
        return _safe_log(num / den)

    def _logP_ins(self, prev, ins):
        num = self.adds.get((prev, ins), 0) + self.add_smooth
        base = max(self.adds_base.get(prev, 0), sum(c for (a, _), c in self.adds.items() if a == prev))
        den = base + self.add_smooth * max(1, len(self._alphabet_from_tables()))
        return _safe_log(num / den)

    def _logP_del(self, prev, deleted):
        num = self.dels.get((prev, deleted), 0) + self.del_smooth
        base = max(self.dels_base.get(prev, 0), sum(c for (a, _), c in self.dels.items() if a == prev))
        den = base + self.del_smooth * max(1, len(self._alphabet_from_tables()))
        return _safe_log(num / den)
