import pandas as pd


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dropping columns identified in notebook.
    """
    columns_to_drop = [
        'encounter_id',
        'patient_nbr',
        'weight',
        'payer_code',
        'medical_specialty'
    ]

    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replacing '?' with NaN and drop rows accordingly.
    """
    df = df.replace('?', pd.NA)
    df = df.dropna()
    return df
