"""
data_loading.py
----------------
Loads all raw Tourism Dataset Excel files into pandas DataFrames.

Usage:
    from data_loading import load_all_raw
    tables = load_all_raw("data/raw")
"""

import os
import pandas as pd

RAW_FILENAMES = {
    "transaction": "Transaction.xlsx",
    "user": "User.xlsx",
    "city": "City.xlsx",
    "item": "Item.xlsx",
    "type": "Type.xlsx",
    "mode": "Mode.xlsx",
    "continent": "Continent.xlsx",
    "country": "Country.xlsx",
    "region": "Region.xlsx",
    "updated_item": os.path.join("Additional_Data_for_Attraction_Sites", "Updated_Item.xlsx"),
}


def load_all_raw(raw_dir: str) -> dict:
    """
    Load every raw Excel file listed in RAW_FILENAMES.

    Parameters
    ----------
    raw_dir : str
        Path to the folder containing the raw .xlsx files
        (e.g. 'data/raw').

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are short table names (e.g. 'transaction', 'user'),
        values are the loaded DataFrames exactly as read from Excel
        (no cleaning applied here).
    """
    tables = {}
    for key, filename in RAW_FILENAMES.items():
        path = os.path.join(raw_dir, filename)
        if not os.path.exists(path):
            print(f"[WARN] Missing file, skipped: {path}")
            continue
        df = pd.read_excel(path)
        tables[key] = df
        print(f"[OK] Loaded '{key}' from {filename}: shape={df.shape}")
    return tables


if __name__ == "__main__":
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    tables = load_all_raw(raw_dir)
    print("\nLoaded tables:", list(tables.keys()))
