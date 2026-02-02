import pandas as pd
import numpy as np
from pathlib import Path
import pickle

#debemos normalizar de tal forma que haya muchos ceros 

# Rutas
ruta_y = "data/Raw/pv_dataset_full.xlsx"
ruta_x = "data/Raw/wx_dataset_full.xlsx"
OUT_DIR = Path("../data/Processed")

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

def unir_x_y(x_df, y_df, y_col="Energy"):
    """
    Une X e Y en un único DataFrame por el índice temporal.
    """
    assert x_df.index.equals(y_df.index), "X e Y no están alineados"

    df = x_df.copy()
    df[y_col] = y_df[y_col]
    return df



def save_joint_splits(train, val, test, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # quitar tz del índice para Excel
    train = quitar_timezone_indice(train)
    val   = quitar_timezone_indice(val)
    test  = quitar_timezone_indice(test)

    train.to_excel(out_dir / "train.xlsx", index=True)
    val.to_excel(out_dir / "val.xlsx", index=True)
    test.to_excel(out_dir / "test.xlsx", index=True)

    print("Guardados: train.xlsx, val.xlsx, test.xlsx")


def save_joint_splits_inference(x_df, y_df, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # quitar tz del índice para Excel
    x_inference = quitar_timezone_indice(x_df)
    y_inference   = quitar_timezone_indice(y_df)
    

    x_inference.to_excel(out_dir / "x_inference.xlsx", index=True)
    y_inference.to_excel(out_dir / "y_inference.xlsx", index=True)
    

    print("Guardados: x_inference, y_inference")

def descargar_excel(url_o_ruta: str, hoja: str | int = 0) -> pd.DataFrame:
    """
    Descarga (si es URL) o abre (si es ruta local) un Excel y devuelve la hoja indicada como DataFrame.
    hoja puede ser nombre ("Sheet1") o índice (0, 1, 2...).
    """
    return pd.read_excel(url_o_ruta, sheet_name=hoja)

def concatenar_hojas(df1: pd.DataFrame, df2: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    """
    Concatena dos DataFrames.
    axis=0 -> uno debajo del otro (mismas columnas)
    axis=1 -> lado a lado (mismas filas / índice)
    """
    return pd.concat([df1, df2], axis=axis, ignore_index=(axis == 0))


def rellenar_weather_description(x_df, valor="desconocido"):
    x_df = x_df.copy()
    x_df["weather_description"] = x_df["weather_description"].fillna(valor)
    return x_df

def one_hot_encode_weather(x_df, col="weather_description"):
    """
    Convierte la columna categórica en múltiples columnas binarias (0/1).
    Ej: weather_description -> w_desc_rain, w_desc_sun, etc.
    """
    x_df = x_df.copy()
    # dtype=int asegura que sean 0 y 1 numéricos, no booleanos
    dummies = pd.get_dummies(x_df[col], prefix="w_desc", dtype=int)
    
    # Concatenar y eliminar la original
    x_df = pd.concat([x_df, dummies], axis=1)
    if col in x_df.columns:
        x_df = x_df.drop(columns=[col])
    return x_df

def factorizar_weather_description(x_df, col="weather_description", sort=True):
    x_df = x_df.copy()
    ids, vocab = pd.factorize(x_df[col], sort=sort)
    x_df[col] = ids
    return x_df

def transformar_y(y):
    y = y.copy()
    y["Energy"] = np.log1p(y["Energy"])
    return y

def parsear_dt_iso_16(x_df, col="dt_iso", tz_destino=None):
    """
    - Se queda con los primeros 16 chars (YYYY-MM-DD HH:MM)
    - Parsea en UTC (tz-aware)
    - Si tz_destino no es None, convierte a esa tz
    """
    x_df = x_df.copy()

    s = (
        x_df[col].astype("string").str.strip().str[:16].str.replace("T", " ", regex=False)
    )

    s = pd.to_datetime(s, format="%Y-%m-%d %H:%M", errors="coerce")

    s = s.dt.tz_localize(tz_destino)

    x_df[col] = s
    return x_df

def eliminar_fechas_invalidas(x_df, col):
    return x_df.dropna(subset=[col])

def extraer_features_circulares(df, col_dt="dt_iso"):
    """
    Mantengo tus fórmulas.
    Asume col_dt datetime64 (tz-aware o naive).
    """
    df = df.copy()
    dt = df[col_dt]

    df["hour"] = dt.dt.hour
    df["month"] = dt.dt.month
    df["weekday"] = dt.dt.weekday

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    df["month_sin"] = np.sin(2*np.pi*(df["month"]-1)/12)
    df["month_cos"] = np.cos(2*np.pi*(df["month"]-1)/12)

    df["weekday_sin"] = np.sin(2*np.pi*df["weekday"]/7)
    df["weekday_cos"] = np.cos(2*np.pi*df["weekday"]/7)

    df = df.drop(columns=["hour", "month", "weekday"])
    return df

def ordenar_y_indexar_por_fecha(df, col="dt_iso"):
    return df.sort_values(col).set_index(col)

def quitar_timezone_indice(df):
    """
    Quita tz del índice sin desplazar horas (Excel-friendly).
    """
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def replace_na(df, col):
    df = df.copy()
    if col in df.columns:
        df[col] = df[col].fillna(0)
    return df

def eliminar_columnas_no_informativas(x_df, cols_a_eliminar):
    x_df = x_df.copy()
    cols_existentes = [c for c in cols_a_eliminar if c in x_df.columns]
    if cols_existentes:
        x_df.drop(columns=cols_existentes, inplace=True)
    return x_df

# -----------------------------
# Split temporal por índices (NO inteligente)
# -----------------------------
def split_por_indices(df, train_frac=0.7, val_frac=0.15):
    """
    Split SECUENCIAL:
      train = primeros train_frac
      val   = siguiente val_frac
      test  = resto
    """
    df = df.copy()
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    return train, val, test

# -----------------------------
# Normalización SIN leakage
# -----------------------------
def fit_stats(train_df):
    # columnas continuas (NO binarias ni cíclicas)
    numeric_cols = []
    for c in train_df.columns:
        

        # no normalizar binarias
        if train_df[c].dropna().isin([0, 1]).all():
            continue

        # no normalizar sin/cos
        if c.endswith("_sin") or c.endswith("_cos"):
            continue

        numeric_cols.append(c)

    mu = train_df[numeric_cols].mean()
    std = train_df[numeric_cols].std().replace(0, 1.0)
    return numeric_cols, mu, std

def apply_stats(df: pd.DataFrame, numeric_cols, mu, std):
    df = df.copy()
    cols_to_normalize = [c for c in numeric_cols if c in df.columns]
    df[cols_to_normalize] = (df[cols_to_normalize] - mu) / std
    return df

def cargar_stats(out_dir=OUT_DIR, filename="stats.pkl"):
    with open(Path(out_dir) / filename, "rb") as f:
        stats = pickle.load(f)

    return (
        stats["X"]["numeric_cols"],
        stats["X"]["mu"],
        stats["X"]["std"],
        stats["Y"]["numeric_cols"],
        stats["Y"]["mu"],
        stats["Y"]["std"],
    )

# -----------------------------
# Preparación X e Y
# -----------------------------
def preparar_x_df(x_df, tz_destino=None, valor_weather_nulo="desconocido"):
    x_df = x_df.copy()
    x_df = rellenar_weather_description(x_df, valor=valor_weather_nulo)
    
    x_df = one_hot_encode_weather(x_df, col="weather_description")
    # parse datetime
    x_df = parsear_dt_iso_16(x_df, tz_destino=tz_destino)
    
    x_df = eliminar_fechas_invalidas(x_df, col="dt_iso")
    
    # features circulares (tus fórmulas)
    x_df = extraer_features_circulares(x_df, col_dt="dt_iso")
    
    # eliminar columnas no informativas
    x_df = eliminar_columnas_no_informativas(x_df, cols_a_eliminar=["lat", "lon"])

    # NA
    x_df = replace_na(x_df, "rain_1h")

    # ordenar e indexar por fecha (como antes)
    x_df = ordenar_y_indexar_por_fecha(x_df, col="dt_iso")
    
    return x_df

def split_por_indices(df, train_frac=0.7, val_frac=0.15):
    df = df.copy()
    n = len(df)

    train_end = int(n * train_frac)
    val_end   = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    return train, val, test

def dividir(x_df, train_frac=0.8, val_frac=0.10):
    # split por índices (temporal)
    train, val, test = split_por_indices(x_df, train_frac=train_frac, val_frac=val_frac)
    return train, val, test

def normalizar_df(train, val, test, quitar_tz_excel=True):
     # normalización sin leakage (fit en train)
    numeric_cols, mu, std = fit_stats(train)
    train = apply_stats(train, numeric_cols, mu, std)
    val   = apply_stats(val, numeric_cols, mu, std)
    test  = apply_stats(test, numeric_cols, mu, std)

    if quitar_tz_excel:
        train = quitar_timezone_indice(train)
        val   = quitar_timezone_indice(val)
        test  = quitar_timezone_indice(test)
    return train, val, test 

def dividir_x_normalizar_df(x_df, train_frac=0.7, val_frac=0.15, quitar_tz_excel=True):
    # split por índices (temporal)
    train, val, test = split_por_indices(x_df, train_frac=train_frac, val_frac=val_frac)

    # normalización sin leakage (fit en train)
    numeric_cols, mu, std = fit_stats(train)
    train = apply_stats(train, numeric_cols, mu, std)
    val   = apply_stats(val, numeric_cols, mu, std)
    test  = apply_stats(test, numeric_cols, mu, std)

    if quitar_tz_excel:
        train = quitar_timezone_indice(train)
        val   = quitar_timezone_indice(val)
        test  = quitar_timezone_indice(test)
    return train, val, test


def normalizar_df(x_df, numeric_cols, mu, std, quitar_tz_excel=True):
    # split por índices (temporal)
    # normalización sin leakage (fit en train)
    x_df = apply_stats(x_df, numeric_cols, mu, std)
    if quitar_tz_excel:
        x_df = quitar_timezone_indice(x_df)
    return x_df

def preparar_y_df(y_df):
    """
    Mantengo tu idea: renombrar columnas.
    Ajusta si tus nombres reales son distintos.
    """
    y_df = y_df.copy()
    y_df = y_df.rename(columns={'Max kWp':'dt_iso', 82.41:'Energy'})

    # parse fecha
    y_df["dt_iso"] = pd.to_datetime(y_df["dt_iso"], errors="coerce")
    y_df = y_df.dropna(subset=["dt_iso"])

    # ordenar e indexar
    y_df = ordenar_y_indexar_por_fecha(y_df, col="dt_iso")
    return y_df

def normalizar_indice_horario(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index).dt.floor("H")
    return df

def dividir(y_df, train_frac=0.8, val_frac=0.10):
    train, val, test = split_por_indices(y_df, train_frac=train_frac, val_frac=val_frac)
    return train, val, test

def dividir_y_normalizar_df(y_df, train_frac=0.7, val_frac=0.15):
    # split temporal
    train, val, test = split_por_indices(y_df, train_frac=train_frac, val_frac=val_frac)

    
    numeric_cols, mu, std = fit_stats(train)
    train = apply_stats(train, numeric_cols, mu, std)
    val   = apply_stats(val, numeric_cols, mu, std)
    test  = apply_stats(test, numeric_cols, mu, std)

    train = quitar_timezone_indice(train)
    val   = quitar_timezone_indice(val)
    test  = quitar_timezone_indice(test)

    return train, val, test

def pipeline():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # cargar raw
    x_df_0 = cargar_excel(ruta_x, hoja=0)
    y_df_0 = cargar_excel(ruta_y, hoja=0)
    x_df_1 = cargar_excel(ruta_x, hoja=1)
    y_df_1 = cargar_excel(ruta_y, hoja=1)

    x_df = pd.concat([x_df_0, x_df_1], axis=0)
    y_df = pd.concat([y_df_0, y_df_1], axis=0)

    # preparar por separado
    x_df = preparar_x_df(x_df)
    y_df = preparar_y_df(y_df)

    # ==================================================
    # ============ MASE SCALE (SERIE REAL) ==============
    # ==================================================
    y_energy_real = y_df["Energy"].values

    m = 24  # estacionalidad diaria (horaria)
    naive_diff = np.abs(y_energy_real[m:] - y_energy_real[:-m])
    mase_scale = np.mean(naive_diff)

    print(f"MASE scale (real series): {mase_scale:.4f}")

    # alinear por tiempo
    x_df, y_df = x_df.align(y_df, join="inner", axis=0)

    # split X

    x_train, x_val, x_test = dividir(x_df, train_frac=0.8, val_frac=0.10)

    train_columns = x_train.columns
    print(f"Columnas en Train (Total {len(train_columns)}):", list(train_columns))

    x_cols, x_mu, x_std = fit_stats(x_train)
    x_train = normalizar_df(x_train, x_cols, x_mu, x_std, quitar_tz_excel=True)
    x_val = normalizar_df(x_val, x_cols, x_mu, x_std, quitar_tz_excel=True)
    x_test = normalizar_df(x_test, x_cols, x_mu, x_std, quitar_tz_excel=True)
    
    y_train, y_val, y_test = dividir(y_df, train_frac=0.8, val_frac=0.10)
    y_cols, y_mu, y_std = fit_stats(y_train)
    y_train = normalizar_df(y_train, y_cols, y_mu, y_std,quitar_tz_excel=True)
    y_val = normalizar_df(y_val, y_cols, y_mu, y_std,quitar_tz_excel=True)
    y_test =  normalizar_df(y_test, y_cols, y_mu, y_std,quitar_tz_excel=True)

    # unir X e Y
    train = unir_x_y(x_train, y_train)
    val   = unir_x_y(x_val, y_val)
    test  = unir_x_y(x_test, y_test)

    # guardar conjuntos
    save_joint_splits(train, val, test, OUT_DIR)
    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)

    print("\n--- Iniciando Normalización de Inferencia ---")
    
    # 1. Cargar datos brutos de la hoja de inferencia
    x_df_2 = cargar_excel(ruta_x, hoja=2)
    y_df_2 = cargar_excel(ruta_y, hoja=2)

    # 2. Preprocesamiento (Cálculo de senos/cosenos, limpieza de columnas)
    x_df_inf = preparar_x_df(x_df_2)
    y_df_inf = preparar_y_df(y_df_2)

    # 3. Alineación temporal
    x_df_inf = normalizar_dt_df(x_df_inf)
    y_df_inf = normalizar_dt_df(y_df_inf)
    x_df_inf, y_df_inf = x_df_inf.align(y_df_inf, join="inner", axis=0)

    # 4. Asegurar que las columnas de x_df_inf coincidan con las de train
    # 1. Añadir columnas faltantes con ceros
    missing_cols = set(train_columns) - set(x_df_inf.columns)
    for c in missing_cols:
        x_df_inf[c] = 0
    
    # 2. Reordenar y eliminar sobrantes para que sea idéntico a Train
    x_df_inf = x_df_inf[train_columns]

    # 5. NORMALIZAR X (Variables de entrada: Temperatura, nubes, etc.)
    # Aquí es donde fallaba: aplicamos las stats de entrenamiento a x_df_inf
    x_df_inf = apply_stats(x_df_inf, x_cols, x_mu, x_std)
    
    # 6. NORMALIZAR Y (Variable objetivo: Energía)
    # Paso A: Logaritmo (porque el entrenamiento se hizo sobre logaritmos)
    # Paso B: Z-Score con medias de entrenamiento
    y_df_inf = apply_stats(y_df_inf, y_cols, y_mu, y_std)

    # 7. Quitar Timezone para exportar a Excel
    x_df_inf = quitar_timezone_indice(x_df_inf)
    y_df_inf = quitar_timezone_indice(y_df_inf)

    # 8. Unir y Guardar
    inference = unir_x_y(x_df_inf, y_df_inf)
    inference.to_excel(OUT_DIR / "inference.xlsx", index=True)
        # ==================================================
    # ============ GUARDAR STATS PARA INFERENCE =========
    # ==================================================

    # y_train está estandarizada → volver a escala REAL
    
    stats = {
        "X": {
            "numeric_cols": x_cols,
            "mu": x_mu,
            "std": x_std,
        },
        "Y": {
            "numeric_cols": y_cols,
            "mu": y_mu,
            "std": y_std,
        },
        "mase": {
            "scale": mase_scale,
            "m": m,
        },
    }

    with open(OUT_DIR / "stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    print("stats.pkl guardado correctamente en data/Processed/")

    print("Variables de X e Y normalizadas con éxito.")
    print("Columnas procesadas en X:", x_cols)
    print("Ejemplo de valores en X (deberían estar cerca de 0):")
    print(x_df_inf[x_cols].head(3))


def main():
    pipeline()

if __name__ == "__main__":
    main()
