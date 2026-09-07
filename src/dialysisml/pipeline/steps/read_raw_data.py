from contextlib import contextmanager
from typing import Generator

import pandas as pd
from sqlalchemy import Engine, create_engine

from dialysisml import config


@contextmanager
def get_db_engine() -> Generator[Engine]:
    engine = create_engine(
        f"postgresql+psycopg2://{config.DB_USER}:{config.DB_PASS}@127.0.0.1:{config.DB_PORT}/{config.DB_NAME}"
    )
    try:
        yield engine
    finally:
        engine.dispose()


def read_raw_data(fromdb: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two source frames as they come out of the database, dates cast.

    `fromdb` refreshes the parquet cache; nothing else here is a modelling
    choice, so nothing else is configurable.
    """
    if not (config.RAW_SESSIONS_PATH.exists() and config.RAW_EVENTS_PATH.exists()):
        fromdb = True

    if fromdb:
        with get_db_engine() as engine:
            df_sessions = pd.read_sql_query(
                "SELECT * FROM patient.dialysis_sessions", engine
            )
            df_events = pd.read_sql_query("SELECT * FROM patient.patient_events", engine)
        df_sessions.to_parquet(config.RAW_SESSIONS_PATH)
        df_events.to_parquet(config.RAW_EVENTS_PATH)
    else:
        df_sessions = pd.read_parquet(config.RAW_SESSIONS_PATH)
        df_events = pd.read_parquet(config.RAW_EVENTS_PATH)

    df_sessions["t_session"] = pd.to_datetime(df_sessions["t_session"])
    df_events["t_event"] = pd.to_datetime(df_events["t_event"])

    # an event of a patient with no sessions is unusable whatever comes next
    patients = set(df_sessions["patient"].dropna().unique())
    df_events = df_events[df_events["patient"].isin(patients)]

    return df_sessions, df_events
