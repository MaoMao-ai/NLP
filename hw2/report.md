# Spelling Correction with Noisy Channel Model

## Introduction
This project implements a spelling corrector based on the noisy channel model.  
The model combines:
- A **unigram language model** to estimate word probabilities.
- A **weighted edit distance error model** to capture the likelihood of spelling errors.

The goal is to correct corrupted words under the assumption that each word has at most one edit error.

---

## Modeling Assumptions
- **Single edit assumption**: Each observed word differs from the intended word by at most one insertion, deletion, substitution, or adjacent transposition.
- **Noisy channel formulation**:  
  \[
  \hat{w} = \arg\max_w P(w) \cdot P(\text{observed} \mid w)
  \]
- **Unigram LM**: Word probabilities are estimated from unigram counts only. No sentence context is used.
- **Error model**:  
  - Substitution errors estimated from `substitutions.csv`.  
  - Insertions and deletions from `additions.csv` / `deletions.csv`.  
  - Transpositions assigned a fixed small probability.  
- **Boundary handling**: The symbol `#` in the CSV data is mapped to `'^'` to represent start-of-word.
- **No-error prior**: A fixed probability is assigned to the “no edit” case to prevent over-correction.
- **Vocabulary restriction**: Only words in the unigram vocabulary (e.g., Norvig’s `count_1w.txt`) can be produced as corrections.

---

## Scenarios and Examples

### Works Well

| Input      | Output     | Comment |
|------------|------------|---------|
| speling    | spelling   | Corrected a missing "l". |
| teh        | the        | Fixed a common transposition error. |
| recieve    | receive    | Corrected frequent "ie/ei" substitution. |

### Could Do Better

| Input        | Output      | Comment |
|--------------|-------------|---------|
| accomodetion | accomodetion| True word "accommodation" requires two edits, outside the model’s assumption. |
| thier        | their       | Correct here, but cannot distinguish "their" vs "there" without context. |
| chatgpt      | (unchanged) | Out-of-vocabulary word, so no correction is possible. |

---

## Discussion
- **Strengths**: The system handles common single-character mistakes effectively when the true word is in the vocabulary and supported by the error model.
- **Limitations**:  
  - Cannot resolve words that differ by multiple edits.  
  - Cannot disambiguate homophones or context-dependent words.  
  - Cannot handle out-of-vocabulary (OOV) terms such as proper nouns or neologisms.  

---

## Possible Improvements
1. **Contextual modeling**: Use bigram or neural language models (e.g., BERT) to incorporate sentence context.
2. **Multi-edit handling**: Allow two or more edits using beam search or pruning strategies.
3. **Larger vocabularies**: Expand the dictionary to include proper nouns, technical terms, and subword units (BPE).
4. **Keyboard-aware error model**: Adjust edit probabilities by physical key distances for more realistic corrections.

---

## Conclusion
The noisy channel spelling corrector demonstrates strong performance on simple, single-edit spelling errors.  
It struggles with multi-edit cases, homophones, and out-of-vocabulary words.  
Future improvements with contextual language models and richer error modeling could significantly increase its accuracy and robustness.