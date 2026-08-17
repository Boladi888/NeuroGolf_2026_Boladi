#!/usr/bin/env python3
"""Check Hugging Face token and artifact repo access without printing secrets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


ROOT = Path(__file__).resolve().parents[1]


def load_local_env() -> None:
    load_dotenv(ROOT / ".env", override=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Hugging Face auth and repo access.")
    parser.add_argument("--repo-id", default=None, help="defaults to HF_PRIVATE_REPO_ID, then HF_REPO_ID")
    parser.add_argument("--repo-type", default=None, help="defaults to HF_REPO_TYPE, then model")
    parser.add_argument("--create-private", action="store_true", help="create the repo as private if it does not exist")
    args = parser.parse_args()

    load_local_env()
    token = os.getenv("HF_TOKEN", "").strip()
    repo_id = (args.repo_id or os.getenv("HF_PRIVATE_REPO_ID") or os.getenv("HF_REPO_ID") or "").strip()
    repo_type = (args.repo_type or os.getenv("HF_REPO_TYPE") or "model").strip()

    if not token:
        raise SystemExit("HF_TOKEN is not set")
    if not repo_id:
        raise SystemExit("HF_PRIVATE_REPO_ID or HF_REPO_ID is not set")

    api = HfApi(token=token)
    whoami = api.whoami()
    account = (whoami.get("name") or "").strip()
    if not account:
        raise SystemExit("Could not verify Hugging Face token owner")
    print(f"hf auth ok for {account}")
    namespace = repo_id.split("/", 1)[0]
    if namespace != account:
        raise SystemExit(f"HF_REPO_ID must be under the active HF account ({account}/...), got {repo_id}")

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except HfHubHTTPError as exc:
        if not args.create_private:
            raise SystemExit(f"hf repo check failed for configured repo ({repo_type}); pass --create-private to create it") from exc
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
    print(f"hf repo access ok ({repo_type})")


if __name__ == "__main__":
    main()
