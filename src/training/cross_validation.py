import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def perform_5fold_cv_with_resampling(
    model,
    X,
    y,
    resampler=None,
    random_state=42
):
    """
    Performing 5-Fold Cross Validation with optional resampling.
    """

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    fold_results = []
    best_fold = None
    best_score = -np.inf

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # applying resampling ONLY on training set
        if resampler is not None:
            X_train, y_train = resampler(X_train, y_train)

        # train model
        model.fit(X_train, y_train)

        # prediction
        y_pred = model.predict(X_val)

        # evaluation
        acc = accuracy_score(y_val, y_pred)

        fold_results.append({
            "fold": fold + 1,
            "accuracy": acc
        })

        # track best fold
        if acc > best_score:
            best_score = acc
            best_fold = fold + 1

    results_df = pd.DataFrame(fold_results)

    return results_df, best_fold
