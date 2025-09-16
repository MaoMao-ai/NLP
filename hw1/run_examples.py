
"""
run_examples.py
Generates deterministic & stochastic example applications for the Markov text generator
using NLTK's Austen "Sense and Sensibility" corpus, and writes a Markdown report
(example_applications.md) you can submit for the red item.

Usage:
  python run_examples.py
"""
import os
import random
import datetime
from typing import List, Tuple
try:
    import nltk
except Exception as e:
    raise SystemExit("Please install nltk first: pip install nltk") from e

# Local module
try:
    from mtg import finish_sentence
except Exception as e:
    raise SystemExit("Could not import mtg.py. Make sure run_examples.py is in the same folder as mtg.py.") from e

SEEDS_AND_N = [
    (["she", "was", "not"], 3),
    (["she", "was", "not"], 2),
    (["i", "would", "ask", "her"], 4),
    (["i", "would", "ask", "her"], 3),
    (["they", "were", "sorry"], 5),
    (["they", "were", "sorry"], 4),
    (["marianne", "was"], 2),
    (["elinor", "could"], 3),
]

def ensure_nltk_resources():
    # Quiet downloads; if already present, no-op
    try:
        nltk.data.find('corpora/gutenberg')
    except LookupError:
        nltk.download('gutenberg', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

def load_corpus_tokens():
    ensure_nltk_resources()
    text = nltk.corpus.gutenberg.raw('austen-sense.txt').lower()
    tokens = nltk.word_tokenize(text)
    return tokens

def run_one(seed: List[str], n: int, corpus: List[str], stochastic: bool, seed_value: int = 42) -> List[str]:
    if stochastic:
        random.seed(seed_value)
    return finish_sentence(seed, n, corpus, randomize=stochastic)

def main():
    corpus = load_corpus_tokens()
    lines_md = []
    lines_md.append("# Markov Text Generation — Example Applications\n")
    lines_md.append(f"_Generated on {datetime.datetime.now().isoformat(timespec='seconds')}_\n")
    lines_md.append("\n## Deterministic vs. Stochastic Modes\n")
    lines_md.append("- **Deterministic (`randomize=False`)**: argmax selection with alphabetical tie-break (reproducible).\n")
    lines_md.append("- **Stochastic (`randomize=True`)**: sample from backoff distribution (diverse outputs; set `random.seed(42)` for reproducibility in this script).\n")
    lines_md.append("\n---\n\n## Examples\n")

    for seed_tokens, n in SEEDS_AND_N:
        # deterministic
        det = run_one(seed_tokens, n, corpus, stochastic=False)
        # stochastic (seeded)
        sto = run_one(seed_tokens, n, corpus, stochastic=True, seed_value=42)

        header = f"### Seed: `{ ' '.join(seed_tokens) }`, n = {n}"
        lines_md.append(header + "\n")
        lines_md.append("**Deterministic**\n")
        lines_md.append("```\n" + " ".join(det) + "\n```\n\n")
        lines_md.append("**Stochastic (seed=42)**\n")
        lines_md.append("```\n" + " ".join(sto) + "\n```\n\n---\n\n")

    # Observations
    lines_md.append("## Observations\n")
    lines_md.append("- Small n (2–3): more flexible/diverse but sometimes less coherent.\n")
    lines_md.append("- Larger n (4–5): more locally fluent but can become repetitive/memorized.\n")
    lines_md.append("- Stochastic mode adds variability; deterministic mode is good for testing.\n")

    out_path = os.path.join(os.getcwd(), "example_applications.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_md))

    print(f"✅ Wrote {out_path}")
    print("Done. You can submit example_applications.md with mtg.py, test_mtg.py, and test_examples.csv.")

if __name__ == "__main__":
    main()
