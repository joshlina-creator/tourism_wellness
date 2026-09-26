import os
from huggingface_hub import HfApi, create_repo

HF_USERNAME = os.environ.get("HF_USERNAME")
SPACE_REPO_ID = f"{HF_USERNAME}/tourism-wellness-package-app"
DEPLOY_DIR = "tourism_project/deployment"


def push_to_space():

    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        print("HF_TOKEN not found")
        return

    api = HfApi(token=hf_token)

    create_repo(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        space_sdk="docker",
        token=hf_token,
        exist_ok=True,
        private=False,
    )

    api.upload_folder(
        folder_path=DEPLOY_DIR,
        repo_id=SPACE_REPO_ID,
        repo_type="space",
    )

    print(
        f"App deployed at: "
        f"https://huggingface.co/spaces/{SPACE_REPO_ID}"
    )


if __name__ == "__main__":
    push_to_space()
  
