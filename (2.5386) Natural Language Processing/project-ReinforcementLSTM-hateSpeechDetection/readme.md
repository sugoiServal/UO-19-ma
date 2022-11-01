## Abstract
Using reinforcement learning based structure
is a novel idea in natural language processing.
Tianyang Zhang and Minlie Huang proposed Information Distilled LSTM (ID-LSTM) and Hierarchical Structured LSTM (HS-LSTM) which
learn policies to simplify information and learn
structured representation to sentences. In this
paper, we trained the two models in Offensive
Language Identification task and compare their
performance with other baselines. We also add
BiLSTM, attention and BERT components to the
architecture in ID-LSTM and HS-LSTM. The result shows that although the original ID-LSTM
and HS-LSTM only slightly extend the baselines,
their modified version does significantly better.
The result indicates that reinforcement learning
structures have more potential in natural language
processing.

## Results
The experiment result is given in Table 1. The overall performance of the ID-LSTM and HS-LSTM without modification
is identical or slightly better than the baseline models CNN
and BiLSTM, an observation that aligns with the result from
the original paper(Zhang et al., 2018b). However, the modified version of ID-LSTM and HS-LSTM get even better performance from the original model. Comparing to BiLSTM,
modified ID-BiLSTM+attention and HS-BiLSTM+attention
increase by about 1% in accuracy and about 30% in the recall rate of offensive tweets.
BERT pre-trained models get the highest classification accuracy and are generally better than the same structure that
uses GloVe and Emoji2Vec embedding. Of all implemented
models, BERT+ID-BiLSTM gets the best result.
![](https://imgur.com/4AHv5JV.jpg)