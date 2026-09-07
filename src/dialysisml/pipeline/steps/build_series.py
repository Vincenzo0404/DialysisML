"""The starting frame: sessions joined to the event that follows them."""

from typing import Sequence

import pandas as pd
import pandera.pandas as pa

from dialysisml.schema import Frame, Kind, Role, columns

ADVERSE_EVENTS = ["Decesso", "Ricovero per accesso vascolare", "Ricovero per altro"]

# What comes out of build_series has a fixed schema to have a unique starting point for the pipeline
SOURCE_SCHEMA = pa.DataFrameSchema(
    {
        **columns(str, Role.META, Kind.IDENTIFIER, ["patient"]),
        **columns("datetime64[ns]", Role.META, Kind.DATETIME, ["t_session"]),
        # bookkeeping of the outcome: known only after the fact
        **columns(str, Role.META, Kind.IDENTIFIER, ["event"], nullable=True),
        **columns(str, Role.META, Kind.CATEGORICAL, ["event_type"], nullable=True),
        **columns("datetime64[ns]", Role.META, Kind.DATETIME, ["t_event"]),
        **columns(
            "timedelta64[ns]", Role.META, Kind.TIMEDELTA, ["duration"], nullable=True
        ),
        **columns(bool, Role.LABEL, Kind.BINARY, ["censored"]),
        # observed days: to the event when there is one, to the last session
        # when censored
        **columns(int, Role.LABEL, Kind.NUMERIC, ["tte"], checks=pa.Check.ge(0)),
        **columns(
            str,
            Role.FEATURE,
            Kind.CATEGORICAL,
            ["sesso", "tipo_accesso_vascolare"],
            nullable=True,
        ),
        **columns(
            int,
            Role.FEATURE,
            Kind.NUMERIC,
            # constant within a series while the series stops at the first
            # event, so its slope and rmse over a window are always zero
            ["eta_paziente_anni", "prior_events_count"],
            checks=pa.Check.ge(0),
        ),
        **columns(
            float,
            Role.FEATURE,
            Kind.NUMERIC,
            [
                "bmi",
                "hb",
                "bcm_post",
                "ffm_post",
                "fm_post",
                "ferritina",
                "tsat",
                "fosforemia",
                "azotemia_pre",
                "albuminemia",
                "transferrina",
                "calcemia",
                "pth",
                "azotemia_post",
                "alpha_eri",
                "dosaggio_epoetina_alpha_mensile",
                "vitamina_d",
                "sideremia",
                "fosfatasi_alcalina",
                "fosfalcindex",
                "alpha_epodose_weight",
                "colesterolemia",
                # demografiche e storiche
                "eta_accesso_giorni",
                "mesi_fav",
                # macchina, singola seduta
                "minuti_dialisi",
                "peso_post",
                "peso_pre",
                "qb",
                "uf",
                "arter",
                "vena",
                "a_v",
                "pa_qb",
                "pv_qb",
                # punteggi FAV/CVC
                "score_fav",
                "score_cvc",
            ],
            nullable=True,
        ),
    },
    strict=True,
    checks=pa.Check(
        lambda df: df["event"].isna() == df["censored"],
        name="event_matches_censored",
        description="`censored` is true iff `event` is null.",
    ),
)


def select_event_types(
    events: pd.DataFrame, *, types: Sequence[str] = ADVERSE_EVENTS
) -> pd.DataFrame:
    """Which event types count as adverse: part of what the target means."""
    return events[events["event_type"].isin(list(types))]


def count_prior_events(events: pd.DataFrame) -> pd.DataFrame:
    """How many events that patient has had by each one, itself included."""
    events = events.sort_values(["patient", "t_event", "event"], kind="stable").copy()
    events["prior_events_count"] = events.groupby("patient").cumcount() + 1
    return events


def build_series(
    sessions: pd.DataFrame, events: pd.DataFrame, *, first_event_only: bool = True
) -> Frame:
    """One series per patient: the sessions up to the event that follows them."""
    all_events = events
    events = events.sort_values(["patient", "t_event", "event"], kind="stable")

    if first_event_only:
        first_session = sessions.groupby("patient")["t_session"].min()
        # remove events preceding first patient's session
        events = events[events["t_event"] >= events["patient"].map(first_session)]
        # remove events after first
        events = events.drop_duplicates(subset="patient", keep="first")

    merged = pd.merge_asof(
        left=sessions.sort_values("t_session", kind="stable"),
        # the count comes from the backward join below: taken from here it
        # would be the one at the *next* event, not the one at the session
        right=events.drop(columns="prior_events_count").sort_values(
            ["t_event", "event"], kind="stable"
        ),
        left_on="t_session",
        right_on="t_event",
        by="patient",
        direction="forward",
    )

    # which patient had at least one event
    had_event = merged["patient"].isin(set(events["patient"]))
    # remove sessions subsequent to the event associated with the series
    # i.e. keeps sessions for which (t_event is null) => (patient NEVER had an event) is false
    merged = merged[merged["t_event"].notna() | ~had_event]

    merged["censored"] = merged["t_event"].isna()
    last_session = merged.groupby("patient")["t_session"].transform("max")
    merged["t_event"] = merged["t_event"].fillna(last_session)
    merged["tte"] = (merged["t_event"] - merged["t_session"]).dt.days

    # a second join, backwards: the one above looks forward, so a censored
    # patient — no event ahead of him — would come out of it missing
    prior = all_events[["patient", "t_event", "prior_events_count"]].rename(
        columns={"t_event": "t_prior"}
    )
    merged = pd.merge_asof(
        left=merged.sort_values("t_session", kind="stable"),
        right=prior.sort_values("t_prior", kind="stable"),
        left_on="t_session",
        right_on="t_prior",
        by="patient",
        direction="backward",
        allow_exact_matches=False,
    ).drop(columns="t_prior")
    merged["prior_events_count"] = merged["prior_events_count"].fillna(0).astype(int)

    return Frame(merged, SOURCE_SCHEMA).validate()
