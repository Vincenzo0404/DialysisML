import argparse
import os
from typing import Sequence

import dtale
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sqlalchemy import create_engine

from dialysisml import config
from dialysisml.window import Split, WindowConfig, WindowStore


def get_db_connection():
    return create_engine(
        f"postgresql+psycopg2://{config.DB_USER}:{config.DB_PASS}@127.0.0.1:{config.DB_PORT}/{config.DB_NAME}"
    )


def show(df: pd.DataFrame) -> None:
    """show dataframe in dtale"""

    d = dtale.show(df, host="127.0.0.1", open_browser=True)
    print(f"D-Tale aperto: {d.main_url()}")


def winsorize_df(
    df: pd.DataFrame, cols: list[str], lower_quantile=0.01, upper_quantile=0.99
) -> pd.DataFrame:
    """
    Winsorizes the specified columns of the DataFrame by replacing values below the lower quantile
    with the lower quantile value and values above the upper quantile with the upper quantile value.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    cols (list): List of column names to be winsorized.
    lower_quantile (float): The lower quantile threshold (default is 0.01).
    upper_quantile (float): The upper quantile threshold (default is 0.99).

    Returns:
    pd.DataFrame: The DataFrame with winsorized columns.
    """
    for col in cols:
        lower_bound = df[col].quantile(lower_quantile)
        upper_bound = df[col].quantile(upper_quantile)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df


def read_raw_data(
    fromdb: bool = False, show_df: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(config.RAW_DF_IN_PATH) or not os.path.exists(
        config.RAW_DF_OUT_PATH
    ):
        conn = get_db_connection()

        fromdb = True  # force reading from db if parquet files don't exist
        print(
            "Parquet files not found. Forcing read from database and creating parquet files."
        )
        # read data from db
        df_in = pd.read_sql_query("SELECT * FROM patient.lstm_in", conn)
        df_out = pd.read_sql_query("SELECT * FROM patient.target_data", conn)
        # update parquet files
        df_in.to_parquet(config.RAW_DF_IN_PATH)
        df_out.to_parquet(config.RAW_DF_OUT_PATH)

    else:
        # read data from parquet files
        df_in = pd.read_parquet(config.RAW_DF_IN_PATH)
        df_out = pd.read_parquet(config.RAW_DF_OUT_PATH)

    return df_in, df_out


def prep_series(
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    features: Sequence[str],
    targets: Sequence[str],
    show_df: bool = False,
    conf: WindowConfig | None = None,
) -> pd.DataFrame:
    """Prepares dialysis series data.

    Parameters:
    df_in (pd.DataFrame): Input DataFrame containing features.
    df_out (pd.DataFrame): Output DataFrame containing target data.
    features (Sequence[str]): Sequence of feature column names.
    targets (Sequence[str]): Sequence of target column names.

    Returns:
    pd.DataFrame: dataframe patient dialysis series data.
    """
    if conf is None:
        conf = WindowConfig()

    print(f"Preparing series with conf: {conf}")

    # select only relevant columns
    df_in = df_in[["patient", "t", *features]]

    # make sure df_in and df_out share the same patients
    patients_in = set(df_in["patient"].dropna().unique())
    df_out = df_out[df_out["patient"].isin(patients_in)]

    # cast dates
    df_in["t"] = pd.to_datetime(df_in["t"])
    df_out["t_out"] = pd.to_datetime(df_out["t_out"])

    df_in = df_in.sort_values(["patient", "t"])

    # forward and backward filling, grouped by patient
    cols_to_fill = df_in.columns.difference(["patient", "t"])
    df_in[cols_to_fill] = df_in.groupby("patient")[cols_to_fill].ffill()
    df_in[cols_to_fill] = df_in.groupby("patient")[cols_to_fill].bfill()

    if conf.impute:
        # Keep rows whose numeric features are still missing: they are filled
        # with training-set medians in split_series, once the split exists and
        # a median can be computed without leaking the test set.
        # The categoricals are dropped anyway (only ~1.5% of rows), because the
        # one-hot encoding below happens before the split and cannot wait.
        df_in = df_in.dropna(subset=config.CATEGORICAL_FEATURES)
    else:
        df_in = df_in.dropna()

    df_in = df_in.sort_values("t")
    df_out = df_out.sort_values("t_out")

    # merge df_in and df_out on patient and nearest timestamp
    df_merged = pd.merge_asof(
        left=df_in,
        right=df_out,
        left_on="t",
        right_on="t_out",
        by="patient",
        direction="forward",
    )

    df_merged = df_merged.dropna(
        subset=["t_out", "event"]
    )  # drop rows without a corresponding next event in df_out

    df_merged = df_merged.sort_values(["patient", "t"])

    # compute target attributes
    df_merged["remaining_days"] = (df_merged["t_out"] - df_merged["t"]).dt.days

    # remove exact event day to avoid remaining days = 0
    df_merged = df_merged[df_merged["remaining_days"] > 0]

    assert (df_merged["remaining_days"] >= 0).all()

    # apply one-hot encoding
    ohe = OneHotEncoder(categories=[["M", "F"], ["FAV", "CVC", "CVC-Per", "CVC-Tem"]])
    ohe.fit(df_merged[config.CATEGORICAL_FEATURES])
    new_cols = ohe.get_feature_names_out()

    df_merged[new_cols] = pd.DataFrame(
        ohe.transform(df_merged[config.CATEGORICAL_FEATURES]).toarray(),  # type: ignore
        columns=new_cols,
        index=df_merged.index,
    )

    # remove outliers
    df_merged = winsorize_df(
        df_merged,
        cols=config.NUMERIC_FEATURES,
        lower_quantile=0.01,
        upper_quantile=0.99,
    )

    return df_merged


def split_series(
    df_merged: pd.DataFrame,
    train_size: float = 0.6,
    random_state: int = 42,
    impute: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=42)

    groups = df_merged["patient"].astype(str)

    train_indices, test_indices = next(gss.split(df_merged, groups=groups))

    df_train = df_merged.iloc[train_indices].copy()
    df_test = df_merged.iloc[test_indices].copy()

    if impute:
        # Fit on train only, exactly like the scalers below: the median of a
        # column must not be told anything about the test patients.
        medians = df_train[config.NUMERIC_FEATURES].median()
        df_train[config.NUMERIC_FEATURES] = df_train[config.NUMERIC_FEATURES].fillna(
            medians
        )
        df_test[config.NUMERIC_FEATURES] = df_test[config.NUMERIC_FEATURES].fillna(
            medians
        )

    std_scaled_cols = ["hb", "peso_pre", "peso_post", "albuminemia", "calcemia"]

    robust_scaled_cols = [
        "ferritina",
        "fosforemia",
        "qb",
        "uf",
        "azotemia_pre",
        "eta_accesso_giorni",
        "bmi",
        "transferrina",
        "bcm_post",
        "ffm_post",
        "fm_post",
        "pth",
        "azotemia_post",
        "alpha_eri",
        "dosaggio_epoetina_alpha_mensile",
        "vitamina_d",
        "sideremia",
        "fosfatasi_alcalina",
        "alpha_epodose_weight",
    ]

    minmax_scaled_cols = ["minuti_dialisi", "eta_paziente_anni", "tsat"]

    def transform_df(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        transformer: StandardScaler | MinMaxScaler | RobustScaler,
    ):
        transformer.fit(df_train)  # fit only on train to avoid data leakage
        return transformer.transform(df_train), transformer.transform(df_test)

    df_train[std_scaled_cols], df_test[std_scaled_cols] = transform_df(
        df_train[std_scaled_cols], df_test[std_scaled_cols], StandardScaler()
    )
    df_train[robust_scaled_cols], df_test[robust_scaled_cols] = transform_df(
        df_train[robust_scaled_cols], df_test[robust_scaled_cols], RobustScaler()
    )
    df_train[minmax_scaled_cols], df_test[minmax_scaled_cols] = transform_df(
        df_train[minmax_scaled_cols], df_test[minmax_scaled_cols], MinMaxScaler()
    )

    return df_train, df_test


def extract_windows(
    df: pd.DataFrame,
    features: Sequence[str],
    targets: Sequence[str],
    window_stride: int,
    window_size: int,
) -> Split:
    """Extracts windows of features and corresponding target values from the input DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing features and target.
    features (Sequence[str]): Sequence of feature column names to be used for window extraction.
    targets (Sequence[str]): Sequence of target column names to be used for window extraction.
    window_stride (int): The stride for window extraction.
    window_size (int): The size of the window to be extracted.

    Returns a Split, whose `meta` holds one row per window describing where that
    window came from: see config.META_COLUMNS.
    """
    X_windows = []
    y_windows = []
    meta_records = []

    feature_cols = list(features)
    target_cols = list(targets)

    count = 0

    # compute effective length of the window based on stride and size
    effective_length = (window_size - 1) * window_stride + 1

    # iterate over series
    for (patient, event), group in df.groupby(["patient", "event"]):

        # ensure group is sorted by timestamp 't'
        group = group.sort_values("t")
        group_features = group[feature_cols].to_numpy()
        group_target = group[target_cols].to_numpy()
        times = group["t"].to_numpy()

        # one event per group, hence one event date
        t_out = group["t_out"].iloc[0]
        series_len = len(group)

        # discard short series
        if len(group_features) < effective_length:
            count += 1
            continue

        # create windows from group
        for i in range(len(group) - effective_length + 1):

            # the window's current session, the one it is anchored on
            anchor = i + effective_length - 1

            # Extract X
            X_win = group_features[i : i + effective_length : window_stride]

            # Extract y
            y_win = group_target[anchor]

            X_windows.append(X_win)
            y_windows.append(y_win)
            meta_records.append(
                {
                    "patient": patient,
                    "event": event,
                    "series_idx": i,
                    "series_len": series_len,
                    "t_start": times[i],
                    "t": times[anchor],
                    "t_out": t_out,
                }
            )

    print(
        f"Discarded series (too short to form a {effective_length}-step span): {count}"
    )
    print(
        f"Created {len(X_windows)} windows of size {window_size} with stride {window_stride}"
    )

    # columns= keeps the schema right even if no series was long enough
    meta = pd.DataFrame(meta_records, columns=config.META_COLUMNS)

    return Split(np.array(X_windows), np.array(y_windows), meta)


def make_windows(
    fromdb: bool = False,
    show_df: bool = False,
    window_conf: WindowConfig | None = None,
) -> tuple[Split, Split]:
    """Data pipeline to create windows from raw dialysis data.

    Returns the train and test splits, each carrying its own metadata.
    """
    if window_conf is None:
        window_conf = WindowConfig()

    raw_features = window_conf.raw_features
    final_features = window_conf.features
    targets = ["remaining_days"]

    df_in, df_out = read_raw_data(fromdb=fromdb, show_df=show_df)
    df_merged = prep_series(
        df_in, df_out, raw_features, targets, show_df=show_df, conf=window_conf
    )
    df_train, df_test = split_series(
        df_merged, train_size=0.6, random_state=42, impute=window_conf.impute
    )
    df_train = df_train.sort_values(by=["patient", "t"])
    df_test = df_test.sort_values(by=["patient", "t"])
    train = extract_windows(
        df_train,
        final_features,
        targets,
        window_stride=window_conf.stride,
        window_size=window_conf.size,
    )
    test = extract_windows(
        df_test,
        final_features,
        targets,
        window_stride=window_conf.stride,
        window_size=window_conf.size,
    )
    return train, test


def main():
    parser = argparse.ArgumentParser(
        description="Build LSTM windows from dialysis data."
    )
    parser.add_argument(
        "--fromdb",
        action="store_true",
        help="Read raw data from the database instead of parquet files.",
    )
    parser.add_argument(
        "--show-df",
        action="store_true",
        help="Show intermediate DataFrames with dtale.",
    )
    args = parser.parse_args()

    make_windows(fromdb=args.fromdb, show_df=args.show_df)


if __name__ == "__main__":
    main()
