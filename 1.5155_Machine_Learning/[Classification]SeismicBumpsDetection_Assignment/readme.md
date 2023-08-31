## Task
- We compared four classification models in the unbalanced dataset [seismic-bump](https://archive.ics.uci.edu/ml/datasets/seismic‐bumps), the task is to predict seismic activity, given some historic measurements.  

- the four prediction models are:
  -  a decision tree, 
  -  a rule‐based classifiers, 
  -  a Naïve Bayesian classifiers  
  -  a k‐nearest neighbor classifier

### Sampling Method
In task2, we applied different sampling method to augment the data, and compare these with results of vanilla model. 
#### the sampling methods includes:
  - Oversampling
  - Under-sampling
  - Balanced sampling, i.e. combining oversampling and under-sampling

### Metrics
#### Classification quality are evaluated with
- accuracy
- confusion matrix
- Recall, Precision
- ROC Area