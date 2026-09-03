from contextlib import contextmanager
from typing import Generator, Sequence

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


def read_raw_data(
    fromdb: bool = False,
    adv_event_types: Sequence[str] = [
        "Decesso",
        "Ricovero per accesso vascolare",
        "Ricovero per altro",
    ],
    first_event_only: bool = True,
) -> pd.DataFrame:
    """
    Sessions joined to the first adverse event that follows each one.

    Parameters:
    `fromdb`: read data again from db.
    `adv_event_types`: which event types to consider as adverse event.
    `first_event_only`: removes patient series tail after merging his sessions with only his first event if any is found.
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

    # select event types
    df_events = df_events[df_events["event_type"].isin(adv_event_types)]

    # cast dates
    df_sessions["t_session"] = pd.to_datetime(df_sessions["t_session"])
    df_events["t_event"] = pd.to_datetime(df_events["t_event"])

    # remove multiple events
    df_events = df_events.sort_values(["patient", "t_event", "event"], kind="stable")

    if first_event_only:
        # the first event on record is not the first observable one: for ~1500
        # patients it predates their first session, and picking it would send
        # their whole series into the post-event tail dropped below
        first_session = df_sessions.groupby("patient")["t_session"].min()
        df_events = df_events[
            df_events["t_event"] >= df_events["patient"].map(first_session)
        ]

        df_events = df_events.drop_duplicates(subset="patient", keep="first")
        assert df_events["patient"].is_unique, "Multiple events for same patient."

    # merge df_sessions and df_events by patient and nearest timestamp.
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
    # A patient leaves the risk set at his first event: the sessions after it match
    # nothing forward, and keeping them would call censored a stretch where a fifth
    # of the rows has a further event in the data. A null `t_event` for a patient
    # with no event at all is the real thing, and stays.
    if first_event_only:
        had_event = df_merged["patient"].isin(set(df_events["patient"]))
        df_merged = df_merged[df_merged["t_event"].notna() | ~had_event]

    # handle censored series
    df_merged["has_event"] = df_merged["t_event"].notna()

    last_session = df_merged.groupby("patient")["t_session"].transform("max")
    df_merged["t_event"] = df_merged["t_event"].fillna(last_session)

    assert (
        df_merged["event"].isna() == ~df_merged["has_event"]
    ).all(), "Event, and Has_event columns are not coherent."

    return df_merged
