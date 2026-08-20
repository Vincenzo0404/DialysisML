import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from dialysisml.features import *
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / "service" / ".env")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT")

BASE_DIR = Path(__file__).resolve()
DATA_PATH = BASE_DIR.parent.parent.parent / "data"
DATA_PATH.mkdir(parents=True, exist_ok=True)

RAW_DF_IN_PATH = DATA_PATH / "df_in.parquet"
RAW_DF_OUT_PATH = DATA_PATH / "df_out.parquet"

ANALYSIS_PATH = BASE_DIR.parent.parent.parent / "analysis"
ANALYSIS_PATH.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = BASE_DIR.parent.parent.parent / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Senza URI esplicito MLflow usa ./mlruns relativo alla working directory, che per
# un notebook e' notebooks/. Ancorandolo qui la cartella e' sempre quella top-level,
# da qualunque punto si lanci il codice.
MLRUNS_PATH = BASE_DIR.parent.parent.parent / "mlruns"
MLRUNS_PATH.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = MLRUNS_PATH.as_uri()


def mape_loss_fn(preds, targets):
    return torch.mean(torch.abs((targets - preds) / (targets))) * 100.0
