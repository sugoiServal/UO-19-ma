## Part One
- we used SpaCy to tokenize each line of `microblog2011.txt`. Each url was treated as unique toke.
- Foreign language were filtered. Each token was bracket by [] in each line
- We used nltk library to conduct statistics of tokens 

## Part Two
- we evaluate 8 different word embedding methods over 7 similarity task datasets
and 4 analogy questions task datasets. 

- word embeddings include: pre-trained CBOW, Skip-grams, GloVe,
PDC, HDC, LexVec, ConceptNet Numberbatch, FastText. 

- Similarity test dataset are: MTurk, MEN, WS353, Rubenstein and Goodenough, Rare Words, SimLex999, TR9856. 
- Analogy questions task datasets are: MSR WordRep, Google analogy, MSR, SEMEVAL 2012 Task2.

#### Results are presented in `Assignment1_report.pdf`