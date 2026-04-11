import pandas as pd


def evaluate_results(results_path: str):
    df = pd.read_csv(results_path)

    print("=== Evaluation Summary ===")
    print(df.describe())


if __name__ == "__main__":
    evaluate_results("experiments/results/cv_results.csv")
