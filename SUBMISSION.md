# Job Application Submission Summary

### Summary

#### Data transformations

The data transformations can be divided in two distinct
groups: categorical and non-categorical (or numerical).
This distinction appeared naturally in my case because
I encountered some additional challenges with the
categorical data (namely RAM issues) which made so
that I decided to focus first on the numerical data,
and then on the categorical.

__Numerical features:__ 
These are created in the function `create_customer_features` in
the file `returning/ml.py`.

The general approach was to extract a maximum of information
from any column.
When aggregating the data per each customer, I calculated
min, max, average, standard deviation, and number of
unique elements whenever possible.

For time-based features, I took the number of days since both
first and last order, and standard deviation as a poor man's
proxy of how wide the distribution of order times is.

__Categorical features:__
These are created in the function `create_dummy_features` in
the file `returning/ml.py`.

The straightforward approach would have been to create
a one-hot representation for all categories, hour and day
included.
Given some hardware constrains, I was forced to use only
a subset of categories.

#### ML models

Similarly to the data  transformation, this section is also
split in two.
I kept the actual best-model evaluation in the code using
grid search for demonstration purposes.
In a real scenario one would pick the best model and
re-build it separately.
In general I tried to use simple ML models that give
decent result right off the bat without need for much
tuning, due to the time constraint.

__Numerical data:__ Here I tested a random forest and a
logistic regression, on the premise that I wanted to
have a probability prediction (E.g. an SVM would not
have done the job in this case).

The main difference between this two algorithms is speed.
Random forest can be progressively improved by adding trees,
but it becomes more and more slower.
Logistic regression on the other hand is very quick,
which is often a big plus in production.

Eventually a logistic regression was the better one, with
a AUC for the ROC curve of 0.81.

__Categorical data:__ The premise with categorical data was
to have algorithms that would work well with sparse matrices,
so I chose multinomial naive Bayes and logistic regression.

In this case the logistic regression was the better model, with
a AUC for the ROC curve of 0.76.

#### Ensemble model

The natural progression of this would be to combine the two
model into one using a voting classifier, which there was not
enough time to do, but the resulting model should
provide an overall better performance.

#### Possible improvements

(Almost everything in this list is related to lack of time):

- Evaluation and optimization of an ensemble model using
  the two models developed
  (AKA *voting classifier*);

- One-hot representation for restaurants and cities
  (memory issues so I gave up after a while);

- Proper hyperparameter optimization
  (only tried a few simple values and not all parameters);
- Usage of more data instead of removing part of
  the majority class for balancing (using techniques
  for imbalanced learning);

- Analysis of feature importance;

- Proper use of holidays
  (if actual country was known);

- Creation of proper data transformers independent of
  training data (so that they could be reused at test time)
  and inclusion of transformers with models (e.g. using
  sklearn pipelines);

- Fix random seed everywhere for reproducibility;

- Tests;

- Better naming of variables/objects;

- Better analysis and scaling of feature distribution
  (The scaler used here is a generic good compromise
  but wasn't really optimized);


### Quickstart

Requirements: docker.

```
bash launch-analysis.sh
```
will build a docker image and run the analysis in one go.
However, the images and data produced will remain in the
container and therefore won't be accessible.
To get better access to the analysis results, see below.

### Installation

Python 3.8 is required.

```
pip install -r requirements-dev.txt
```

### Notebooks

To run the notebooks:

```
jupyter-notebook
```

### Analysis

To run the code from start to finish:
```
bash run.sh
```
#### Performance graphs

In `data/` one can find a collection of PNG images:
```
customer_features_pdist_prc.png
customer_features_roc.png
dummy_features_pdist_prc.png
dummy_features_roc.png
```
which display the performance of the two models.

#### Cleanup

To reset to the original repo state after a run, execute

```
bash cleanup.sh
```
