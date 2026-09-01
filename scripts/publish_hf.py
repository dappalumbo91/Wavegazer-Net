"""Create/update the Hugging Face model repo. Token is read from the local key file, never printed."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

KEY_FILE = Path.home() / "Desktop" / "Hugging face API key.txt"
REPO_ID = "dappalumbo91/Wavegazer-Net"


def _token() -> str:
    lines = KEY_FILE.read_text(encoding="utf-8").splitlines()
    keys = [ln.strip() for ln in lines if ln.strip().startswith("hf_")]
    if not keys:
        raise SystemExit(f"no hf_ token in {KEY_FILE}")
    return keys[0]


def main() -> None:
    from huggingface_hub import HfApi, login

    token = _token()
    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)
    info = api.whoami()
    name = info.get("name") or info.get("fullname") or "?"
    print(f"hf user={name}")
    repo = f"{name}/Wavegazer-Net" if name != "?" else REPO_ID
    url = api.create_repo(repo_id=repo, repo_type="model", exist_ok=True, private=False)
    print(f"repo={url}")
    card = ROOT / "HF_README.md"
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="model",
        commit_message="Wavegazer Net model card",
    )
    buf = ROOT / "artifacts" / "wavegazer_buffers.pt"
    if buf.is_file():
        api.upload_file(
            path_or_fileobj=str(buf),
            path_in_repo="wavegazer_buffers.pt",
            repo_id=repo,
            repo_type="model",
            commit_message="Frozen codon buffers (zero trainable)",
        )
    print("uploaded README + buffers")


if __name__ == "__main__":
    main()
