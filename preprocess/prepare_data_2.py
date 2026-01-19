import pandas as pd
import numpy as np
from pathlib import Path

# Rutas
ruta_y = "../data/Raw/pv_dataset_full.xlsx"
ruta_x = "../data/Raw/wx_dataset_full.xlsx"
OUT_DIR = Path("../data/Processed")


# =====================================================
# Utils datetime
# =====================================================
def normalizar_dt_df(df, col=None):
    df = df.copy()
    if col is None:
        df.index = pd.to_datetime(df.index, errors="coerce").floor("min")
        df = df[~df.index.isna()]
    else:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.floor("min")
        df = df.dropna(subset=[col]).set_index(col)
    return df


def cargar_excel(ruta: str, hoja: str | int = 0):
    return pd.read_excel(ruta, sheet_name=hoja)


# =====================================================
# Weather encoding (SIN get_dummies)
# =====================================================
def rellenar_weather_description(x_df, valor="desconocido"):
    x_df = x_df.copy()
    x_df["weather_description"] = x_df["weather_description"].fillna(valor)
    return x_df


def factorizar_weather_description(x_df, col="weather_description", sort=True):
    x_df = x_df.copy()
    codes, vocab = pd.factorize(x_df[col], sort=sort)
    x_df[col] = codes.astype(np.int16)
    return x_df, vocab


# =====================================================
# Y transform
# =====================================================
def transformar_y(y):
    y = y.copy()
    y["Energy"] = np.log1p(y["Energy"])
    return y


# =====================================================
# Datetime + features
# =====================================================
def parsear_dt_iso_16(x_df, col="dt_iso", tz_destino=None):
    x_df = x_df.copy()
    s = (
        x_df[col]
        .astype("string")
        .str.strip()
        .str[:16]
        .str.replace("T", " ", regex=False)
    )
    s = pd.to_datetime(s, format="%Y-%m-%d %H:%M", errors="coerce")
    s = s.dt.tz_localize(tz_destino)
    x_df[col] = s
    return x_df


def eliminar_fechas_invalidas(x_df, col):
    return x_df.dropna(subset=[col])


def extraer_features_circulares(df, col_dt="dt_iso"):
    df = df.copy()
    dt = df[col_dt]

    df["hour"] = dt.dt.hour
    df["month"] = dt.dt.month
    df["weekday"] = dt.dt.weekday

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    return df.drop(columns=["hour", "month", "weekday"])


def ordenar_y_indexar_por_fecha(df, col="dt_iso"):
    return df.sort_values(col).set_index(col)


def quitar_timezone_indice(df):
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def replace_na(df, col):
    df = df.copy()
    if col in df.columns:
        df[col] = df[col].fillna(0)
    return df


def eliminar_columnas_no_informativas(x_df, cols):
    x_df = x_df.copy()
    x_df = x_df.drop(columns=[c for c in cols if c in x_df.columns])
    return x_df


# =====================================================
# Split + normalización
# =====================================================
def split_por_indices(df, train_frac=0.7, val_frac=0.15):
    n = len(df)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]


def fit_stats(train_df):
    numeric_cols = []
    for c in train_df.columns:
        if not pd.api.types.is_numeric_dtype(train_df[c]):
            continue
        if train_df[c].dropna().isin([0, 1]).all():
            continue
        if c.endswith("_sin") or c.endswith("_cos"):
            continue
        numeric_cols.append(c)

    mu = train_df[numeric_cols].mean()
    std = train_df[numeric_cols].std().replace(0, 1.0)
    return numeric_cols, mu, std

def fit_stats_y(train_y_df, col="Energy"):
    mu = train_y_df[col].mean()
    std = train_y_df[col].std()
    if std == 0 or np.isnan(std):
        std = 1.0
    return float(mu), float(std)

def apply_stats_y(y_df, mu, std, col="Energy"):
    y_df = y_df.copy()
    y_df[col] = (y_df[col] - mu) / std
    return y_df


def dividir_y_df(y_df):
    train, val, test = split_por_indices(y_df)

    # 1) misma transformación base que ya usabas
    train = transformar_y(train)
    val = transformar_y(val)
    test = transformar_y(test)

    # 2) stats SOLO con TRAIN (como X)
    mu_y, std_y = fit_stats_y(train, col="Energy")

    # 3) aplicar z-score a train/val/test
    train = apply_stats_y(train, mu_y, std_y, col="Energy")
    val = apply_stats_y(val, mu_y, std_y, col="Energy")
    test = apply_stats_y(test, mu_y, std_y, col="Energy")

    return quitar_timezone_indice(train), quitar_timezone_indice(val), quitar_timezone_indice(test)


def apply_stats(df, numeric_cols, mu, std):
    df = df.copy()
    df[numeric_cols] = (df[numeric_cols] - mu) / std
    return df


# =====================================================
# Preparación X / Y
# =====================================================
def preparar_x_df(x_df):
    x_df = rellenar_weather_description(x_df)
    x_df, _ = factorizar_weather_description(x_df)
    x_df = parsear_dt_iso_16(x_df)
    x_df = eliminar_fechas_invalidas(x_df, "dt_iso")
    x_df = extraer_features_circulares(x_df)
    x_df = eliminar_columnas_no_informativas(x_df, ["lat", "lon"])
    x_df = replace_na(x_df, "rain_1h")
    x_df = ordenar_y_indexar_por_fecha(x_df)

    return x_df


def preparar_y_df(y_df):
    y_df = y_df.rename(columns={"Max kWp": "dt_iso", 82.41: "Energy"})
    y_df["dt_iso"] = pd.to_datetime(y_df["dt_iso"], errors="coerce")
    y_df = y_df.dropna(subset=["dt_iso"])
    y_df = ordenar_y_indexar_por_fecha(y_df)
    return y_df


def dividir_x_df(x_df):
    train, val, test = split_por_indices(x_df)
    cols, mu, std = fit_stats(train)
    train = apply_stats(train, cols, mu, std)
    val = apply_stats(val, cols, mu, std)
    test = apply_stats(test, cols, mu, std)
    return quitar_timezone_indice(train), quitar_timezone_indice(val), quitar_timezone_indice(test)

def implement_x_df(x_df):
    cols, mu, std = fit_stats(x_df)
    x_df = apply_stats(x_df, cols, mu, std)
    return quitar_timezone_indice(x_df)


def dividir_y_df(y_df):
    train, val, test = split_por_indices(y_df)

    # normalización z-score con stats SOLO de train (misma técnica que X)
    cols, mu, std = fit_stats(train)   # aquí cols será ["Energy"]
    train = apply_stats(train, cols, mu, std)
    val   = apply_stats(val,   cols, mu, std)
    test  = apply_stats(test,  cols, mu, std)

    return quitar_timezone_indice(train), quitar_timezone_indice(val), quitar_timezone_indice(test)


# =====================================================
# Pipelines
# =====================================================
def unir_x_y(x_df, y_df, y_col="Energy"):
    assert x_df.index.equals(y_df.index)
    df = x_df.copy()
    df[y_col] = y_df[y_col]
    return df


def pipeline_training():
    x_df = pd.concat(
        [cargar_excel(ruta_x, 0), cargar_excel(ruta_x, 1)]
    )
    y_df = pd.concat(
        [cargar_excel(ruta_y, 0), cargar_excel(ruta_y, 1)]
    )

    x_df = preparar_x_df(x_df)
    y_df = preparar_y_df(y_df)

    x_df, y_df = x_df.align(y_df, join="inner", axis=0)

    x_train, x_val, x_test = dividir_x_df(x_df)
    y_train, y_val, y_test = dividir_y_df(y_df)

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    unir_x_y(x_train, y_train).to_excel(OUT_DIR / "train.xlsx")
    unir_x_y(x_val, y_val).to_excel(OUT_DIR / "val.xlsx")
    unir_x_y(x_test, y_test).to_excel(OUT_DIR / "test.xlsx")


def pipeline_inference():
    x_df = preparar_x_df(cargar_excel(ruta_x, 2))
    y_df = preparar_y_df(cargar_excel(ruta_y, 2))

    # alinear
    x_df, y_df = x_df.align(y_df, join="inner", axis=0)

    # ------- obtener stats desde TRAIN (recalculando desde raw, mínimo cambio) -------
    x_full = pd.concat([cargar_excel(ruta_x, 0), cargar_excel(ruta_x, 1)])
    y_full = pd.concat([cargar_excel(ruta_y, 0), cargar_excel(ruta_y, 1)])

    x_full = preparar_x_df(x_full)
    y_full = preparar_y_df(y_full)
    x_full, y_full = x_full.align(y_full, join="inner", axis=0)

    # split TRAIN para stats
    x_train, _, _ = split_por_indices(x_full)
    y_train, _, _ = split_por_indices(y_full)

    # X stats (train)
    cols_x, mu_x, std_x = fit_stats(x_train)

    # Y stats (train) -> misma técnica
    y_train = transformar_y(y_train)          # si mantienes log1p
    cols_y, mu_y, std_y = fit_stats(y_train)  # cols_y = ["Energy"]

    # aplicar stats a inference
    x_df = apply_stats(x_df, cols_x, mu_x, std_x)

    y_df = transformar_y(y_df)                # si mantienes log1p
    y_df = apply_stats(y_df, cols_y, mu_y, std_y)

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    unir_x_y(quitar_timezone_indice(x_df), quitar_timezone_indice(y_df)).to_excel(OUT_DIR / "inference.xlsx")



def main():
    pipeline_training()
    pipeline_inference()


if __name__ == "__main__":
    main()
