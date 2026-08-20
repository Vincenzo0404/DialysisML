RAW_FEATURES = [
    # Esami ematici e indici sistemici (I "Pesi Massimi" originali e nuovi)
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

CATEGORICAL_FEATURES = ["sesso", "tipo_accesso_vascolare"]
NUMERIC_FEATURES = [f for f in RAW_FEATURES if f not in CATEGORICAL_FEATURES]

OHE_FEATURES = [
    "sesso_M",
    "sesso_F",
    "tipo_accesso_vascolare_FAV",
    "tipo_accesso_vascolare_CVC",
    "tipo_accesso_vascolare_CVC-Per",
    "tipo_accesso_vascolare_CVC-Tem",
]

FINAL_FEATURES = NUMERIC_FEATURES + OHE_FEATURES

# Columns of the per-window metadata table: what identifies a window, as
# opposed to what the network is trained on.
# `patient` and `event` together identify a series: `event` alone is NOT
# unique across patients. `remaining_days` is deliberately absent, being
# exactly (t_out - t).days.
META_COLUMNS = [
    "patient",  # patient id
    "event",  # event id, unique only within a patient
    "series_idx",  # position of the window inside its series
    "series_len",  # number of sessions in the series
    "t_start",  # date of the oldest session of the window (t = 0)
    "t",  # date of the current session of the window (t = W-1)
    "t_out",  # date of the event
]
