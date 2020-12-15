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

### Installation

Python 3.8 is required.

```
pip install -r requirements.txt
```

### Analysis

```
bash run.sh
```
#### Performance graphs

In `data/` one can find a collection of PNG images:
```

```

#### Cleanup

To reset to the original repo state after a run, execute

```
bash cleanup.sh
```