# HW3: POS Tagging with Hidden Markov Models

## Evaluation on Brown Sentences 10150–10152 (Red Item)

We evaluated on sentences **10150–10152** of the Brown corpus.  
- **Token-level Accuracy:** 93.6%

### Example Predictions and Errors

#### Sentence 10150
`Those coming from other denominations will welcome the opportunity to become informed .`  
- **Gold:** DET VERB ADP ADJ NOUN VERB VERB DET NOUN PRT VERB VERB .  
- **Pred:** DET NOUN ADP ADJ NOUN VERB VERB DET NOUN PRT VERB VERB .  
- **Error:** *coming* → Gold = VERB, Pred = NOUN  

#### Sentence 10151
`The preparatory class is an introductory face-to-face group in which new members become acquainted with one another .`  
- **Gold:** DET ADJ NOUN VERB DET ADJ ADJ NOUN ADP DET ADJ NOUN VERB VERB ADP NUM DET .  
- **Pred:** DET ADJ NOUN VERB DET ADJ NOUN NOUN ADP DET ADJ NOUN VERB VERB ADP NUM NOUN .  
- **Errors:**  
  - *face-to-face* → Gold = ADJ, Pred = NOUN  
  - *another* → Gold = DET, Pred = NOUN  

#### Sentence 10152
`It provides a natural transition into the life of the local church and its organizations .`  
- **Gold:** PRON VERB DET ADJ NOUN ADP DET NOUN ADP DET ADJ NOUN CONJ DET NOUN .  
- **Pred:** PRON VERB DET ADJ NOUN ADP DET NOUN ADP DET ADJ NOUN CONJ DET NOUN .  
-  Correct

---

##  Error Analysis (Red Item)

### Typical Confusions
- **NOUN ↔ VERB** (e.g., *coming*)  
- **ADJ ↔ NOUN** (e.g., *face-to-face*)  
- **DET ↔ NOUN** (e.g., *another*)  

### Why Errors Occur
1. **Data Sparsity:** Some word–tag pairs are rare even in 10k sentences.  
2. **Lexical Ambiguity:** Words like *coming* can be both verbs and nouns.  
3. **Sentence-Initial Bias:** The π distribution can favor common sentence starts.  
4. **Universal Tagset Coarseness:** Some fine distinctions (e.g., participle vs. adjective) are collapsed.  

**Final Result:** The HMM POS tagger achieves **93.6% accuracy** on Brown sentences 10150–10152, with most errors due to NOUN/VERB and ADJ/NOUN ambiguities.

