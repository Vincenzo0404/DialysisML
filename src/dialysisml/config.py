import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

load_dotenv(PROJECT_ROOT / ".env")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT")

PACKAGE_DIR = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data"
DATA_PATH.mkdir(parents=True, exist_ok=True)

SAVED_ADAPTERS = PROJECT_ROOT / "saved_adapters"
SAVED_ADAPTERS.mkdir(parents=True, exist_ok=True)

# absolute: Hydra's `initialize` takes a path relative to the calling file, so
# composing from anywhere but `main.py` needs `initialize_config_dir` and this
HYDRA_CONF_DIR = PACKAGE_DIR / "conf"

RAW_SESSIONS_PATH = DATA_PATH / "df_sessions.parquet"
RAW_EVENTS_PATH = DATA_PATH / "df_events.parquet"

MLRUNS_PATH = PROJECT_ROOT / "mlruns"
MLRUNS_PATH.mkdir(parents=True, exist_ok=True)

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
