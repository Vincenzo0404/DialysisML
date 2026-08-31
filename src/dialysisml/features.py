RAW_FEATURES = [
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
    "alpha_epodose_weight",
    # Variabili demografiche e storiche
    "eta_accesso_giorni",
    "eta_paziente_anni",
    # Variabili di macchina (Singola seduta)
    "minuti_dialisi",
    "peso_post",
    "peso_pre",
    "qb",
    "uf",
    # Variabili categoriche
    "sesso",
    "tipo_accesso_vascolare",
]

# Columns of the materialized view that the model should not use. Declaring
# what to leave out rather than what to keep means a new column added to the
# view flows in on its own, without touching this file.
EXCLUDED_FEATURES = [
    # "score_fav",
    # "score_cvc",
    # "colesterolemia",
    "fosfalcindex",
    # "a_v",
    # "pa_qb",
    # "#pv_qb",
    "mesi_fav",
    # "arter",
    # "vena",
    "duration",
]

# Which columns are categorical cannot be read off the dtype: an integer code
# would look numeric, and a sex encoded as 0/1 would be scaled by mistake.
CATEGORICAL_FEATURES = ["sesso", "tipo_accesso_vascolare"]
NUMERIC_FEATURES = [f for f in RAW_FEATURES if f not in CATEGORICAL_FEATURES]

# Which scaler suits which column. Roughly normal values get standardised,
# skewed ones with outliers get the robust treatment, bounded ones min-max.
STD_COLUMNS = ["hb", "peso_pre", "peso_post", "albuminemia", "calcemia"]

ROBUST_COLUMNS = [
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

MINMAX_COLUMNS = ["minuti_dialisi", "eta_paziente_anni", "tsat"]

# What identifies a row, as opposed to what the network is trained on. Kept
# through the whole pipeline whatever features are selected: the grouping key,
# the two timestamps, and what the event is.
# `patient` and `event` together identify a series: `event` alone is NOT
# unique across patients.
META_COLUMNS = [
    "patient",  # patient id
    "t_session",  # date of the dialysis session
    "t_event",  # date of the adverse event that follows it
    "event",  # event id, unique only within a patient
    "event_type",  # death, vascular access admission, other admission
]
