import os
import math
from loader import (
    load_char_bigram,
    load_additions,
    load_deletions,
    load_substitutions,
    load_unigrams_words_csv,
    load_unigrams_words_from_txt,
)
from lm import UnigramLM
from weighted_edit import WeightedEdit
from correct import SpellingCorrector


BIG = "hw2/bigrams.csv"         
ADD = "hw2/additions.csv"       
DEL = "hw2/deletions.csv"       
SUB = "hw2/substitutions.csv"   
WORD_CSV = "hw2/word_unigrams.csv"  

NORVIG_TXT = "count_1w.txt"     


def load_word_unigrams_auto():
    """
    Try to load a word-level unigram frequency table.
    Order of attempts:
      1) word_unigrams.csv
      2) count_1w.txt (Norvig)
      3) fallback tiny toy vocab (so the demo still runs)
    Returns: (freq_dict, total)
    """
    if os.path.exists(WORD_CSV):
        print(f"[info] Loading word unigrams from {WORD_CSV}")
        return load_unigrams_words_csv(WORD_CSV)

    if os.path.exists(NORVIG_TXT):
        print(f"[info] Loading word unigrams from {NORVIG_TXT}")
        return load_unigrams_words_from_txt(NORVIG_TXT)

    # Fallback: tiny toy vocabulary
    print("[warn] No word unigram file found (word_unigrams.csv or count_1w.txt).")
    print("[warn] Falling back to a tiny toy vocabulary for demo purposes only.")
    toy = {
        "the": 50000,
        "of": 30000,
        "and": 28000,
        "to": 26000,
        "in": 24000,
        "correct": 1000,
        "spelling": 900,
        "receive": 500,
        "their": 500,
        "there": 500,
        "speech": 400,
        "accommodate": 150,
    }
    return toy, sum(toy.values())


def build_models():
    # Load error-model tables
    subs, subs_base = load_substitutions(SUB)
    adds, adds_base = load_additions(ADD)
    dels, dels_base = load_deletions(DEL)
    big, big_total = load_char_bigram(BIG)

    # Load word-level unigrams (language model)
    word_unigrams, _ = load_word_unigrams_auto()

    # Instantiate models
    lm = UnigramLM(word_unigrams, alpha=1.0)
    we = WeightedEdit(
        subs, subs_base,
        adds, adds_base,
        dels, dels_base,
        big, big_total,
        add_smooth=0.1,
        del_smooth=0.1,
        sub_smooth=0.1,
        trans_p=1e-5,
    )
    vocab = set(word_unigrams.keys())
    corrector = SpellingCorrector(lm, we, vocab=vocab, log_no_error=math.log(0.5))

    return corrector


def run_smoke_tests(corrector):
    tests = [
        "speling",     # -> spelling
        "korrect",     # -> correct
        "teh",         # -> the
        "recieve",     # -> receive
        "speach",      # -> speech
        "thier",       # -> their
        "accomodate",  # -> accommodate
        "and",         # already correct (should often stay)
    ]
    print("\n[demo] Smoke tests:")
    for x in tests:
        y = corrector.correct(x)
        print(f"  {x:12s} -> {y}")


def repl(corrector):
    print("\n[type a word to correct; empty line to exit]")
    try:
        while True:
            x = input("> ").strip()
            if not x:
                break
            print(corrector.correct(x))
    except (EOFError, KeyboardInterrupt):
        pass
    print("\n[bye]")


if __name__ == "__main__":
    corrector = build_models()
    run_smoke_tests(corrector)
    repl(corrector)
