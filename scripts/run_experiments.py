import os
import pandas as pd

from src.data.data_loader import load_data, drop_duplicates
from src.preprocessing.cleaning import drop_unnecessary_columns, handle_missing_values
from src.preprocessing.feature_engineering import (
    replace_age_ranges,
    group_admission_types,
    group_admission_source,
    apply_diag_clustering
)
from src.preprocessing.encoding import encode_categorical
from src.training.cv import perform_5fold_cv_with_resampling
from src.training.resampling import apply_smote
from src.models.model_factory import get_model


def run_all_models(data_path):
    df = load_data(data_path)

    df = drop_duplicates(df)
    df = handle_missing_values(df)
    df = drop_unnecessary_columns(df)

    df = replace_age_ranges(df)
    df = group_admission_types(df)
    df = group_admission_source(df)
    df = apply_diag_clustering(df)

    df = encode_categorical(df)

    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]

    models = ["logistic_regression", "random_forest", "decision_tree"]

    all_results = []

    for model_name in models:
        model = get_model(model_name)

        results, best_fold = perform_5fold_cv_with_resampling(
            model,
            X,
            y,
            resampler=apply_smote
        )

        results["model"] = model_name
        results["best_fold"] = best_fold

        all_results.append(results)

    final_df = pd.concat(all_results)

    os.makedirs("experiments/results", exist_ok=True)
    final_df.to_csv("experiments/results/all_models_cv.csv", index=False)

    print(final_df)


if __name__ == "__main__":
    run_all_models("data/raw/diabetic_data.csv")
