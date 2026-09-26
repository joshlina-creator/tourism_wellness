"""
Data Registration Script
-------------------------
Registers the raw tourism dataset with the Hugging Face Hub so that it is
versioned, centrally accessible, and reusable across the MLOps pipeline
(local run, GitHub Actions, etc.).

Requires an environment variable HF_TOKEN with `write` access to a Hugging
Face account. If the token is not present (e.g. when running this notebook
locally without Hugging Face credentials configured), the script prints
instructions instead of failing, so the rest of the pipeline can still run.
"""

import os
import pandas as pd
from huggingface_hub import HfApi, create_repo

HF_USERNAME = os.environ.get("HF_USERNAME", "your-hf-username")
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-package-dataset"
LOCAL_DATA_PATH = "tourism_project/data/tourism.csv"


def register_dataset():
    hf_token = os.environ.get("HF_TOKEN")

    df = pd.read_csv(LOCAL_DATA_PATH)
    print(f"Loaded dataset with shape: {df.shape}")

    if not hf_token:
        print(
            "HF_TOKEN not found in environment. Skipping the actual upload to "
            "the Hugging Face Hub.\n"
            "To register this dataset for real:\n"
            "  1. Create a Hugging Face account at https://huggingface.co\n"
            "  2. Generate a 'write' access token from Settings > Access Tokens\n"
            "  3. Set it as an environment variable (locally) or as a GitHub "
            "secret named HF_TOKEN (for the CI/CD pipeline)\n"
            "  4. Re-run this script."
        )
        return

    api = HfApi(token=hf_token)
    create_repo(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        token=hf_token,
        exist_ok=True,
        private=False,
    )
    api.upload_file(
        path_or_fileobj=LOCAL_DATA_PATH,
        path_in_repo="tourism.csv",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
    )
    print(f"Dataset registered at: https://huggingface.co/datasets/{DATASET_REPO_ID}")


if __name__ == "__main__":
    register_dataset()
