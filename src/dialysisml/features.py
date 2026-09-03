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
    "colesterolemia",
    # Variabili demografiche e storiche
    "eta_accesso_giorni",
    "eta_paziente_anni",
    # Variabili di macchina (Singola seduta)
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
    # Accesso vascolare: nulle per meta' delle sedute, un paziente ha o una FAV
    # o un CVC. Il valore mancante dice quale, non e' un dato perso.
    "score_fav",
    "score_cvc",
    # Variabili categoriche
    "sesso",
    "tipo_accesso_vascolare",
]

# Columns of the materialized view that the model should not use. Declaring
# what to leave out rather than what to keep means a new column added to the
# view flows in on its own, without touching this file.
EXCLUDED_FEATURES = [
    "fosfalcindex",
    "mesi_fav",
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
    "colesterolemia",
    # Pressioni e rapporti di macchina: code lunghe, il rapporto arriva a 5 con
    # una deviazione standard di 0.33
    "arter",
    "vena",
    "a_v",
    "pa_qb",
    "pv_qb",
]

# score_fav e score_cvc sono punteggi ordinali limitati, come le altre qui
MINMAX_COLUMNS = [
    "minuti_dialisi",
    "eta_paziente_anni",
    "tsat",
    "score_fav",
    "score_cvc",
]


META_COLUMNS = [
    "patient",  # patient id
    "t_session",  # date of the dialysis session
    "t_event",  # date of the adverse event or final observasion of the series
    "event",  # event id, unique only within a patient
    "event_type",  # death, vascular access admission, other admission
    "has_event",
]

# What the model is asked to predict. Numeric like any feature, so anything
# that picks columns by dtype has to exclude these by name or it will scale,
# clip or impute the labels themselves.
TARGET_COLUMNS = [
    "tte",  # time to the next adverse event, or the follow-up if censored
    "has_event",  # whether that time is the event or the end of observation
]
