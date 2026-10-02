"""Assemble (and optionally upload) the Hugging Face Space for the Redraw API.

    .venv/bin/python deploy/hf-space/build.py                      # build into build/hf-space/
    HF_TOKEN=hf_... .venv/bin/python deploy/hf-space/build.py --push <user>/<space>

The Space must run on the SAME data/processed as the deployed web viewer (playback
edge indices and building ids must match), so run this from the checkout that built
the viewer snapshot. See deploy/hf-space/DEPLOY.md.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CODE = ["api", "sim", "residents", "pipeline"]
TOP_FILES = ["pyproject.toml", "LICENSE", "ATTRIBUTION.md"]
# Runtime state the Space recreates itself; plans and reactions live in MongoDB.
SKIP_DATA = {"redraw.db", "redraw.db-wal", "redraw.db-shm", "playback"}
REQUIRED_DATA = ["network_nodes.parquet", "network_edges.parquet", "households.parquet", "persons.parquet",
                 "region_meta.json", "schools_resolved.json", "personas.json"]


def _ignore(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n in ("__pycache__", "tests", ".pytest_cache") or n.endswith(".pyc")}


def build(out: Path) -> Path:
    data = ROOT / "data" / "processed"
    missing = [f for f in REQUIRED_DATA if not (data / f).exists()]
    if missing:
        sys.exit(f"data/processed is incomplete (missing {', '.join(missing)}). Run pipeline/build_all.py and "
                 "start the API once (it writes personas.json), then rebuild.")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for d in CODE:
        shutil.copytree(ROOT / d, out / d, ignore=_ignore)
    shutil.copytree(ROOT / "data" / "config", out / "data" / "config")
    shutil.copytree(data, out / "data" / "processed",
                    ignore=lambda _d, names: {n for n in names if n in SKIP_DATA})
    for f in TOP_FILES:
        if (ROOT / f).exists():
            shutil.copy2(ROOT / f, out / f)
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    (out / "requirements.txt").write_text("\n".join(deps) + "\n")
    shutil.copy2(HERE / "Dockerfile", out / "Dockerfile")
    shutil.copy2(HERE / "SPACE_README.md", out / "README.md")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"[hf-space] built {out} ({size:.0f} MB)")
    return out


def push(out: Path, space: str) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN to a Hugging Face access token with write access "
                 "(https://huggingface.co/settings/tokens).")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("pip install huggingface_hub first")
    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=space, repo_type="space",
                      commit_message="Deploy Redraw API", delete_patterns=["*"])
    host = space.replace("/", "-").replace("_", "-").lower()
    print(f"[hf-space] pushed. Space: https://huggingface.co/spaces/{space}  API: https://{host}.hf.space")
    print("[hf-space] Set the DATABASE_URL secret in the Space settings (MongoDB Atlas connection string).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "hf-space")
    ap.add_argument("--push", metavar="USER/SPACE", help="upload to this Hugging Face Space")
    a = ap.parse_args()
    out = build(a.out)
    if a.push:
        push(out, a.push)


if __name__ == "__main__":
    main()
