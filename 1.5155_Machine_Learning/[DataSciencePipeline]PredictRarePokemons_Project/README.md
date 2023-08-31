# Predict Rare Pokemons with Predict’em All Dateset

## Abstract 
Pokemon Go is an AR mobile game where
players are interested in catching rare Pokemons in
the game. Predict’em All consists of 292,061 historical
Pokemons’ appearing data including some of the most
rare types. Based on the relevant information in the
data set, we conducted feature engineering and trained
machine learning models that predict rare pokemons’
appear. The best result is about 0.5 recall rate and 0.1
precison on the rare classes. We compared the pros and
cons of different classification models, different sampling
methods, learning models in the highly imbalanced data set.
We are also interested in that whether the number of classes (which is huge as types of Pokemons)
can impact models ability to predict minority classes.

## Methods
#### We did three experiments.
### Section I
we trained 6 models, a kernalized SVM, a
decision tree, a 4-nearest neighbor, a rule learner,
a random forest and a AdaBoost classifier on the
data set. We used 10-fold cross validation to estimate the average score of the 6 models. The scores
we are interested in is recall, precision, f1 score, the confusion matrices
and classification accuracy.

### Section II
In the second part, we merged data points in rarity
class ’1’, ’2’, ’3’ into class ’0’, and merged data
points in rarity class ’4’, ’5’ into class ’1’, and
used the balanced sampling method to retrain the
6 models, to see if reducing the problem from 5-
classes classification to binary classification can
make the prediction of minor class easier.

### Section III
Finally, in the binarized data set, we tried an unsupervised anomaly detection algorithm one-class
SVM to see if it performs differently to algorithms
in the second part.

## Feature Engineering
The data set has 296,021 data points, 207
features and 151 classes. We categorize Pokemons into different rarity classes based on their number of appearance. Detail of the feature engineering can be found in `project report` and `/src`

# Requirement
- sklearn
- imblearn
- matplotlib
- numpy
- pandas
