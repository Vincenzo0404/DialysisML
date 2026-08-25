from contextlib import contextmanager
from typing import Generator

import pandas as pd
from sqlalchemy import Engine, create_engine

from dialysisml import config

# -- Setup raw dataframe


@contextmanager
def get_db_engine() -> Generator[Engine]:
    engine = create_engine(
        f"postgresql+psycopg2://{config.DB_USER}:{config.DB_PASS}@127.0.0.1:{config.DB_PORT}/{config.DB_NAME}"
    )
    try:
        yield engine
    finally:
        engine.dispose()


def read_raw_data(fromdb: bool = False) -> pd.DataFrame:
    """Sessions joined to the first adverse event that follows each one.

    Reads from parquet unless asked otherwise, falling back to the database
    when a file is missing and caching what it reads.
    """
    if not (config.RAW_SESSIONS_PATH.exists() and config.RAW_EVENTS_PATH.exists()):
        fromdb = True

    if fromdb:
        with get_db_engine() as engine:
            # read data from db
            df_sessions = pd.read_sql_query(
                "SELECT * FROM patient.dialysis_sessions", engine
            )
            df_events = pd.read_sql_query(
                "SELECT * FROM patient.patient_events", engine
            )
        # update parquet files
        df_sessions.to_parquet(config.RAW_SESSIONS_PATH)
        df_events.to_parquet(config.RAW_EVENTS_PATH)

    else:
        # read data from parquet files
        df_sessions = pd.read_parquet(config.RAW_SESSIONS_PATH)
        df_events = pd.read_parquet(config.RAW_EVENTS_PATH)

    # make sure df_sessions and df_events share the same patients
    patients_in = set(df_sessions["patient"].dropna().unique())
    df_events = df_events[df_events["patient"].isin(patients_in)]

    # cast dates
    df_sessions["t_session"] = pd.to_datetime(df_sessions["t_session"])
    df_events["t_event"] = pd.to_datetime(df_events["t_event"])

    # merge df_sessions and df_events by patient and nearest timestamp.
    # Ties on t_event are broken by event id and the sort is stable, so two
    # events sharing a date always resolve the same way.
    df_sessions = df_sessions.sort_values("t_session", kind="stable")
    df_events = df_events.sort_values(["t_event", "event"], kind="stable")

    df_merged = pd.merge_asof(
        left=df_sessions,
        right=df_events,
        left_on="t_session",
        right_on="t_event",
        by="patient",
        direction="forward",
    )

    # sessions with no later event are right-censored: they get no target
    df_merged = df_merged.dropna(subset=["t_event", "event"])

    return df_merged
