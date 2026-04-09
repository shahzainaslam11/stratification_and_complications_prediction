import pandas as pd
from typing import Tuple


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loading dataset from CSV.
    """
    df = pd.read_csv(file_path)
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removing duplicate patients (based on patient_nbr).
    """
    df = df.drop_duplicates(subset='patient_nbr')
    return df
