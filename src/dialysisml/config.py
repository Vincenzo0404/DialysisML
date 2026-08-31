import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv

from dialysisml.features import *

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

load_dotenv(PROJECT_ROOT / ".env")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT")

BASE_DIR = Path(__file__).resolve()
DATA_PATH = BASE_DIR.parent.parent.parent / "data"
DATA_PATH.mkdir(parents=True, exist_ok=True)

RAW_SESSIONS_PATH = DATA_PATH / "df_sessions.parquet"
RAW_EVENTS_PATH = DATA_PATH / "df_events.parquet"

# Il file store ./mlruns e' in maintenance mode da MLflow 3: i metadati (run,
# metriche, parametri) stanno ora in mlflow.db, migrati con `mlflow migrate-filestore`.
# La cartella mlruns/ resta perche' continua a contenere gli artifact, referenziati
# per path dalle righe del database.
MLRUNS_PATH = BASE_DIR.parent.parent.parent / "mlruns"
MLRUNS_PATH.mkdir(parents=True, exist_ok=True)

# Ancorato al top-level del progetto: senza URI esplicito MLflow userebbe un path
# relativo alla working directory, che per un notebook e' notebooks/.
MLFLOW_DB_PATH = BASE_DIR.parent.parent.parent / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
