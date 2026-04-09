from sklearn.preprocessing import LabelEncoder


def encode_categorical(df):
    """
    Applying Label Encoding (as in notebook).
    """
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df
