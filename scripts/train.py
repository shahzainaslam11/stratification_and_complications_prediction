import argparse
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
from src.utils.seed import set_seed


def main(args):
    set_seed(42)

    df = load_data(args.data_path)

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

    model = get_model(args.model)

    resampler = apply_smote if args.use_smote else None

    results, best_fold = perform_5fold_cv_with_resampling(
        model,
        X,
        y,
        resampler=resampler
    )

    print(results)
    print(f"Best Fold: {best_fold}")

    results.to_csv("experiments/results/cv_results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="logistic_regression")
    parser.add_argument("--use_smote", action="store_true")

    args = parser.parse_args()
    main(args)
