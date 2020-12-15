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
first and last order.


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

### Possible improvements

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
