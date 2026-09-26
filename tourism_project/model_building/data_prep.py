"""
Data Preparation Script
------------------------
Cleans the raw tourism dataset, fixes inconsistent category labels, drops
identifier columns that carry no predictive signal, and produces a
stratified train/test split. The resulting files are saved locally and
(optionally) pushed to the Hugging Face Hub dataset repo so that the model
training stage can pull consistent, versioned train/test data.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, create_repo

HF_USERNAME = os.environ.get("HF_USERNAME", "your-hf-username")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-package-dataset"

RAW_PATH = "tourism_project/data/tourism.csv"
TRAIN_PATH = "tourism_project/data/train.csv"
TEST_PATH = "tourism_project/data/test.csv"

TARGET = "ProdTaken"
DROP_COLS = ["Unnamed: 0", "CustomerID"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=col)

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    df = df.drop_duplicates()

    return df


def prepare_data():
    df = pd.read_csv(RAW_PATH)
    print(f"Raw data shape: {df.shape}")

    df = clean_data(df)
    print(f"Cleaned data shape: {df.shape}")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET],
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"Saved train set: {train_df.shape} -> {TRAIN_PATH}")
    print(f"Saved test set:  {test_df.shape} -> {TEST_PATH}")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print(
            "HF_TOKEN not found in environment. Skipping upload of train/test "
            "splits to the Hugging Face Hub."
        )
        return train_df, test_df

    api = HfApi(token=hf_token)
    create_repo(
        repo_id=DATASET_REPO_ID, repo_type="dataset", token=hf_token,
        exist_ok=True, private=False,
    )
    for local_path, repo_path in [(TRAIN_PATH, "train.csv"), (TEST_PATH, "test.csv")]:
        api.upload_file(
            path_or_fileobj=local_path, path_in_repo=repo_path,
            repo_id=DATASET_REPO_ID, repo_type="dataset",
        )
    print(f"Train/test splits pushed to: https://huggingface.co/datasets/{DATASET_REPO_ID}")

    return train_df, test_df


if __name__ == "__main__":
    prepare_data()
