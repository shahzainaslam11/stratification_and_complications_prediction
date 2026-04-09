import pandas as pd
from src.data.data_loader import load_data, drop_duplicates
from src.preprocessing.cleaning import drop_unnecessary_columns, handle_missing_values
from src.preprocessing.feature_engineering import replace_age_ranges


def test_data_pipeline():
    df = pd.DataFrame({
        "patient_nbr": [1, 1, 2],
        "age": ["[0-10)", "[10-20)", "[20-30)"],
        "weight": ["?", "?", "?"]
    })

    df = drop_duplicates(df)
    assert df.shape[0] == 2

    df = handle_missing_values(df)
    assert df.isnull().sum().sum() == 0

    df = replace_age_ranges(df)
    assert df["age"].iloc[0] == 5
