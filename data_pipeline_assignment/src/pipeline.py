import os
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "automobileEDA_dirty_training.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "automobileEDA_processed.csv")


# ---------- EXTRACT ----------
def load_data(path):
    df = pd.read_csv(path)
    return df


def inspect_data(df):
    print(df.head())
    print(df.shape)
    print(df.columns.tolist())
    print(df.dtypes)
    print(df.isnull().sum())
    print("duplikat:", df.duplicated().sum())

    for col in df.select_dtypes(exclude="number").columns:
        print(col, ":", df[col].unique().tolist()[:15])


def clean_data(df):
    df = df.copy()  # biar data raw ga ke-overwrite
    rows_before = len(df)
    missing_before = df.isnull().sum().sum()
    changed = []

    for col in df.select_dtypes(exclude="number").columns:
        before = df[col].copy()
        df[col] = df[col].str.strip().str.lower()
        if not df[col].equals(before):
            changed.append(col)

    df = df.replace(["?", "", "nan", "none", "null", "n/a", "-"], np.nan)


    if "num-of-doors" in df.columns:
        df["num-of-doors"] = df["num-of-doors"].replace({"2": "two", "4": "four"})


    for col in df.select_dtypes(exclude="number").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        not_null = df[col].notnull().sum()
        if not_null > 0 and converted.notnull().sum() / not_null > 0.8:
            df[col] = converted
            changed.append(col)

    for col in df.select_dtypes(include="number").columns:
        if col == "symboling":
            continue
        if (df[col] < 0).sum() > 0:
            df.loc[df[col] < 0, col] = np.nan
            changed.append(col)

    n_dup = df.duplicated().sum()
    df = df.drop_duplicates()

    if "price" in df.columns:
        df = df.dropna(subset=["price"])

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
        changed.append(col)

    df = df.reset_index(drop=True)

    print("baris sebelum:", rows_before, "| sesudah:", len(df))
    print("missing sebelum:", missing_before, "| sesudah:", df.isnull().sum().sum())
    print("duplikat dihapus:", n_dup)
    print("kolom berubah:", sorted(set(changed)))

    return df


def transform_data(df):
    df = df.copy()

    scale_cols = []
    for col in ["price", "horsepower", "engine-size", "curb-weight"]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            mn, mx = df[col].min(), df[col].max()
            df[col + "_scaled"] = (df[col] - mn) / (mx - mn)
            scale_cols += [col, col + "_scaled"]
    print(df[scale_cols].head())

    enc_cols = [c for c in ["body-style", "drive-wheels"] if c in df.columns]
    dummies = pd.get_dummies(df[enc_cols], dtype=int)
    df = pd.concat([df, dummies], axis=1)
    print(df[enc_cols + dummies.columns.tolist()].head())

    return df


# load
def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print("disimpan ke:", path, df.shape)


def run_pipeline():
    df = load_data(RAW_PATH)
    inspect_data(df)
    df_clean = clean_data(df)
    df_final = transform_data(df_clean)
    save_data(df_final, OUT_PATH)


if __name__ == "__main__":
    run_pipeline()