# DialysisML
A Machine Learning application to predict dialysis patients survival metrics.

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C.svg?logo=pytorch&logoColor=white)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E.svg?logo=scikitlearn&logoColor=white)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4-337AB7.svg)]()
[![Hydra](https://img.shields.io/badge/config-Hydra-89b8cd.svg)]()
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2.svg?logo=mlflow&logoColor=white)]()
[![SHAP](https://img.shields.io/badge/explainability-SHAP-6E4B9E.svg)]()
[![uv](https://img.shields.io/badge/uv-managed-DE5FE9.svg?logo=uv&logoColor=white)]()
[![License](https://img.shields.io/badge/license-GPL--3.0-green.svg)](LICENSE)

## Description
- The project's main purpose is to build and analyze multiple ML models in a
  survival analysis context for dialysis patients.
- The project's main domain is to predict the TTE (Time-To-Event) of a
  hypothetical adverse event to which a patient may be subject.

Models' performance is tested along two axes:
- **Pipeline formulation**: using Hydra, we can build multiple pipelines to
  produce dataset variations over which models are trained.
- **Models' configuration**: Hydra allows us to easily run sweeps over
  models' hyperparameters.

## Getting started

Requirements: Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <https://github.com/Vincenzo0404/DialysisML>
cd DialysisML
uv sync
```

`uv sync` creates the virtual environment and installs the required dependencies.

### Data

`data/` is empty since it contains dialysis
patients' records. A run reads two parquet files
that are not shipped here:

```
data/df_sessions.parquet    # one row per dialysis session
data/df_events.parquet      # one row per patient's adverse event
```

### Running
You can run the full ML pipeline with the following command:

```bash
uv run python -m dialysisml.main formulation=first_event_cap365/ffnn
```
`formulation` represents the specific formulation of the data pipeline which produces a specific dataset (in this case ``first_event_cap365`` builds patients' time series ending on their first adverse event), while ``ffnn`` trains multiple FeedForward-Neural-Networks over the said dataset.
After the swept models are trained, the results are logged to MLFLow.

### Inspecting the results
Runs are stored in `mlflow.db`.

You can start MLflow dashboard to visualize training results by running:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

MLFlow server runs locally on http://127.0.0.1:5000. In the UI the 'Experiments' match the pipeline formulations: a
run of `first_event_cap365/ffnn` lands in the `first_event_cap365` experiment,
named after the model that produced it.

### SHAP Analysis
The Python module `dialysisml.explainability.utils` provides `get_shap_values` function which takes as a required argument the MLFlow `run_id` of the run which produced the model you want to analyze.
You can quickly run a full SHAP analysis by running the notebook `notebooks/shap.ipynb` which will plot different charts about SHAP values of the given run.
 Careful: you must still provide the `run_id` of the MLFlow run you want to analyze.

## Project structure
```
.
├── data                                    # contains dataset as parquet files
├── notebooks                               # notebooks for testing or visualizing SHAP plots
└── src                                 
    └── dialysisml                          # package root
        ├── adapters                        # python module containing wrappers for ML models
        ├── conf                            # Hydra's conf folder
        │   ├── adapter                     
        │   ├── formulation                 # Pipeline definition
        │   │   └── first_event_cap365      # Default pipeline definition
        │   ├── postsplit                   
        │   ├── split
        │   └── window_transformation
        ├── explainability                  # Python module for ML models explainability (SHAP)
        ├── models                          # Python module containing models definition
        └── pipeline                        # Python module containing pipeline steps definition
            └── steps
```
