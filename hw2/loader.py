import csv
from collections import Counter


# ---------- UNIGRAMS ----------

def load_unigrams_chars(path):
    """
    Load CHARACTER unigrams from CSV with header:
        unigram,count
    Example: a,3607774
    Returns: (dict{char: count}, total_count)
    """
    freqs = {}
    total = 0
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if not row or len(row) < 2:
                continue
            if first and row[0].strip().lower() == "unigram":
                first = False
                continue
            first = False
            ch = row[0].strip()
            c_str = row[1].strip()
            if not c_str.lstrip("-").isdigit():
                continue
            cnt = int(c_str)
            freqs[ch] = freqs.get(ch, 0) + cnt
            total += cnt
    return freqs, total


def load_unigrams_words_csv(path):
    """
    Load WORD unigrams from CSV with header:
        word,count
    Returns: (dict{word: count}, total_count)
    """
    freqs = {}
    total = 0
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if not row or len(row) < 2:
                continue
            if first and row[0].strip().lower() in ("word", "unigram"):
                first = False
                continue
            first = False
            w = row[0].strip().lower()
            c_str = row[1].strip()
            if not c_str.lstrip("-").isdigit():
                continue
            cnt = int(c_str)
            freqs[w] = freqs.get(w, 0) + cnt
            total += cnt
    return freqs, total


def load_unigrams_words_from_txt(path):
    """
    Load WORD unigrams from Norvig's count_1w.txt format:
        word <tab> count
    Returns: (dict{word: count}, total_count)
    """
    freqs = {}
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            w, c_str = parts[0].lower(), parts[-1]
            if not c_str.lstrip("-").isdigit():
                continue
            cnt = int(c_str)
            freqs[w] = freqs.get(w, 0) + cnt
            total += cnt
    return freqs, total


# ---------- BIGRAMS ----------

def load_char_bigram(path):
    """
    Load character bigram counts from CSV with header:
        bigram,count
    Example: th,123
    Returns: (Counter{(a,b): count}, total_count)
    """
    big = Counter()
    total = 0
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if not row or len(row) < 2:
                continue
            if first and row[0].strip().lower() == "bigram":
                first = False
                continue
            first = False
            bg = row[0].strip().strip('"').strip("'")
            c_str = row[1].strip()
            if len(bg) == 2 and c_str.isdigit():
                cnt = int(c_str)
                big[(bg[0], bg[1])] += cnt
                total += cnt
    return big, total


# ---------- ADDITIONS ----------

def load_additions(path):
    """
    Load insertion errors from CSV with header:
        prefix,added,count
    '#' is mapped to '^' as boundary.
    Returns: (Counter{(prev, inserted): count}, Counter{prev: total_from_prev})
    """
    table = Counter()
    base_prev = Counter()

    def norm(ch): return '^' if ch.strip() == '#' else ch.strip()

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if not row or len(row) < 3:
                continue
            if first and row[0].strip().lower() == "prefix":
                first = False
                continue
            first = False
            prev = norm(row[0])
            ins = row[1].strip()
            c_str = row[2].strip()
            if not c_str.lstrip("-").isdigit():
                continue
            cnt = int(c_str)
            table[(prev, ins)] += cnt
            base_prev[prev] += cnt
    return table, base_prev


# ---------- DELETIONS ----------

def load_deletions(path):
    """
    Load deletion errors from CSV with header:
        prefix,deleted,count
    '#' is mapped to '^' as boundary.
    Returns: (Counter{(prev, deleted): count}, Counter{prev: total_from_prev})
    """
    table = Counter()
    base_prev = Counter()

    def norm(ch): return '^' if ch.strip() == '#' else ch.strip()

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if not row or len(row) < 3:
                continue
            if first and row[0].strip().lower() == "prefix":
                first = False
                continue
            first = False
            prev = norm(row[0])
            deleted = row[1].strip()
            c_str = row[2].strip()
            if not c_str.lstrip("-").isdigit():
                continue
            cnt = int(c_str)
            table[(prev, deleted)] += cnt
            base_prev[prev] += cnt
    return table, base_prev


# ---------- SUBSTITUTIONS ----------

def load_substitutions(path):
    """
    Load substitution errors from CSV with header:
        original,substituted,count
    Example: a,c,7   (intended 'a' mistyped as 'c')
    Returns: (Counter{(orig,sub): count}, Counter{orig: total_from_orig})
    """
    table = Counter()
    base_from = Counter()

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if not row or len(row) < 3:
                continue
            if first and row[0].strip().lower() == "original":
                first = False
                continue
            first = False
            orig = row[0].strip()
            sub = row[1].strip()
            c_str = row[2].strip()
            if not c_str.lstrip("-").isdigit():
                continue
            cnt = int(c_str)
            table[(orig, sub)] += cnt
            base_from[orig] += cnt
    return table, base_from
