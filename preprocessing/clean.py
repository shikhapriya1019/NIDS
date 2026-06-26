"""
preprocessing/clean.py
Raw DataFrame cleaning: column normalisation, missing-value
imputation, infinite-value replacement, and duplicate removal.
"""

import numpy as np
import pandas as pd


def clean_dataframe(df: pd.DataFrame, missing_threshold: float = 0.20) -> pd.DataFrame:
    """
    Clean a raw network-traffic DataFrame.

    Steps
    -----
    1. Strip whitespace from column names and convert to lower-case.
    2. Drop columns where more than *missing_threshold* fraction of
       values are missing.
    3. Replace infinite values (common in flow-rate features) with NaN.
    4. Fill remaining NaN values with the per-column median.
    5. Remove duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame (any of the three benchmark datasets).
    missing_threshold : float
        Maximum fraction of missing values allowed per column (default 0.20).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with reset index.
    """
    df = df.copy()

    # Step 1 — Normalise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Step 2 — Drop columns with too many missing values
    missing_frac = df.isnull().mean()
    keep_cols = missing_frac[missing_frac <= missing_threshold].index
    df = df[keep_cols]

    # Step 3 — Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Step 4 — Median imputation (numeric columns only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Step 5 — Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        print(f"[clean] Removed {before - after} duplicate rows.")

    return df.reset_index(drop=True)
