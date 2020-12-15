# Job Application Submission Summary

### Quickstart

Requirements: docker.

```
bash launch-analysis.sh
```
will build a docker image and run the analysis in one go.
However, the images and data produced will remain in the
container and therefore won't be accessible.
To get better access to the analysis results, see below.

### Repository structure

```
├── SUBMISSION.md        # Submission report
├── Notes.md
├── launch-analysis.sh   # Quick run of the analysis, no question asked
├── data                 # Folder for input and output data
├── models               # Folder for trained models
├── notebooks            # Folder for notebooks (follow the order)
│   └── archive          # Extra explorative notebook (drafts)
├── Dockerfile
├── requirements.txt     # Python requirements for the codebase
├── requirements-dev.txt # Python requirements for developers
├── main.py              # Main python script
├── returning            # Utility module for the main script
│   ├── eval.py          # Evaluation-related code
│   └── ml.py            # Data and ML related code
├── run.sh               # Run main analysis
└── cleanup.sh           # Remove analysis output
```

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

- Tests
