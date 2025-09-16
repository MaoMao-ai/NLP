
from typing import List, Tuple, Sequence, Dict, Iterable, Any
import random
import collections
import math

Alpha = 0.4  # stupid backoff factor

Token = str

def _ngram_counts(tokens: Sequence[Token], n: int):
    """
    Build n-gram and (n-1)-gram counts for all 1..n orders.
    Returns:
        order_counts: list where order_counts[k] is Counter of k-grams (as tuple), for k=1..n
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    order_counts = [None]  # 1-indexed for convenience
    for k in range(1, n+1):
        counter = collections.Counter()
        if len(tokens) >= k:
            for i in range(len(tokens)-k+1):
                gram = tuple(tokens[i:i+k])
                counter[gram] += 1
        order_counts.append(counter)
    return order_counts  # index 1..n

def _score_candidates(history: Tuple[Token, ...],
                      vocab: Iterable[Token],
                      order_counts: List[collections.Counter],
                      n: int) -> Dict[Token, float]:
    """
    Compute stupid-backoff scores P_bo(w | history) for all vocab tokens.
    history: tuple of up to n-1 tokens (we'll use the last k tokens, k<=n-1)
    order_counts: 1..n gram counters
    Returns dict token->score (non-normalized). If all zero (shouldn't happen), fall back to uniform.
    """
    # Limit history to length n-1
    h = history[-(n-1):] if n > 1 else tuple()
    scores: Dict[Token, float] = {}
    # Pre-calc total tokens for unigram
    unigram_total = sum(order_counts[1].values()) if order_counts[1] else 0
    for w in vocab:
        score = 0.0
        # Try highest possible order first
        max_hist_len = min(len(h), n-1)
        found_any = False
        # Iterate k = max_hist_len down to 1
        for k in range(max_hist_len, 0, -1):
            hist_k = tuple(h[-k:])
            hist_count = order_counts[k].get(hist_k, 0)
            if hist_count == 0:
                # No such history at this order, continue backing off
                continue
            # Check if (hist_k + w) exists
            full = hist_k + (w,)
            full_count = order_counts[k+1].get(full, 0)
            if full_count > 0:
                # Found a matching n-gram at order k+1 with nonzero count
                backoff_factor = (Alpha ** (max_hist_len - k))
                score = backoff_factor * (full_count / hist_count)
                found_any = True
                break
            else:
                # Hist exists but continuation not seen; keep backing off
                continue
        if not found_any:
            # Back off all the way to unigram
            # Use alpha^(max_hist_len) * P_MLE(w)  (if unigram_total==0, score remains 0)
            if unigram_total > 0:
                score = (Alpha ** max_hist_len) * (order_counts[1].get((w,), 0) / unigram_total)
        scores[w] = score
    # If all zero (e.g., empty corpus), make uniform tiny scores to avoid degenerate behavior
    if all(s == 0.0 for s in scores.values()):
        n_vocab = len(list(vocab))
        if n_vocab > 0:
            uniform = 1.0 / n_vocab
            return {w: uniform for w in vocab}
    return scores

def _choose_next(scores: Dict[Token, float], randomize: bool) -> Token:
    if not scores:
        raise ValueError("No candidate scores provided")
    if randomize:
        # Sample proportionally to scores; if scores sum to 0 (shouldn't), fall back to uniform
        total = sum(scores.values())
        items = list(scores.items())
        if total <= 0:
            # uniform
            r = random.random()
            cum = 0.0
            for i, (w, s) in enumerate(items):
                cum += 1.0 / len(items)
                if r <= cum or i == len(items)-1:
                    return w
        # Normalize and sample
        r = random.random()
        cum = 0.0
        for i, (w, s) in enumerate(items):
            p = s / total if total > 0 else 0.0
            cum += p
            if r <= cum or i == len(items)-1:
                return w
        return items[-1][0]
    else:
        # Deterministic: argmax score; tie-break alphabetically
        max_score = max(scores.values())
        # collect tokens with score == max_score (use exact float compare; counts-based so OK)
        top_tokens = [w for w, s in scores.items() if s == max_score]
        return sorted(top_tokens)[0]

def finish_sentence(sentence: Sequence[Token],
                    n: int,
                    corpus: Sequence[Token],
                    randomize: bool = False) -> List[Token]:
    """
    Extend the sentence using an n-gram stupid-backoff generator until we hit ., ?, ! or total tokens==10.
    - If randomize=False: deterministic argmax with alphabetical tie-break
    - If randomize=True: sample by backoff-score distribution (normalized)
    """
    if isinstance(sentence, tuple):
        sent = list(sentence)
    else:
        sent = list(sentence)  # copy
    tokens = list(corpus)
    if n < 1:
        raise ValueError("n must be >= 1")
    if len(tokens) == 0:
        return sent[:10]  # nothing to learn from; return truncated or same
    
    # Build counts up to order n
    order_counts = _ngram_counts(tokens, n)
    vocab = set(tokens)
    terminal_set = {'.', '?', '!'}
    
    # Generate until stop
    while len(sent) < 10:
        # Stop if last token is terminal
        if len(sent) > 0 and sent[-1] in terminal_set:
            break
        # Determine history
        history = tuple(sent[-(n-1):]) if n > 1 else tuple()
        scores = _score_candidates(history, vocab, order_counts, n)
        next_tok = _choose_next(scores, randomize)
        sent.append(next_tok)
        # If we just added a terminal, loop will end next iteration
    return sent[:10]
