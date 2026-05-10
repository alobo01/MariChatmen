#!/usr/bin/env python
"""Mirror the MariChatmen source/evaluation datasets from Hugging Face."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download
from marichatmen.constants import ARTIFACT_ROOT

DATASETS = {
    "villanova": "VillanovaAI/villanova-sft-2603",
    "openhermes_es": "Iker/OpenHermes-2.5-Spanish",
    "aya": "CohereLabs/aya_dataset",
    "mentor_es": "projecte-aina/MentorES",
    "alpaca_es": "bertin-project/alpaca-spanish",
    "guanaco_es": "hlhdatscience/guanaco-spanish-dataset",
    "smc_instruct": "somosnlp/SMC-instruct",
}

EVAL_DATASETS = {
    "ifeval_es": "BSC-LT/IFEval_es",
    "esbbq": "BSC-LT/EsBBQ",
}


def _repo_size(info: Any) -> int:
    total = 0
    for sibling in info.siblings or []:
        total += getattr(sibling, "size", None) or 0
    return total


def download_one(alias: str, repo_id: str, out_dir: Path) -> dict[str, Any]:
    api = HfApi()
    started = time.perf_counter()
    target = out_dir / alias
    result: dict[str, Any] = {
        "alias": alias,
        "repo_id": repo_id,
        "repo_type": "dataset",
        "local_dir": str(target),
        "status": "pending",
    }
    try:
        info = api.dataset_info(repo_id, files_metadata=True)
        result.update(
            {
                "gated": getattr(info, "gated", None),
                "private": getattr(info, "private", None),
                "file_count": len(info.siblings or []),
                "expected_bytes": _repo_size(info),
                "files": [
                    {
                        "path": getattr(sibling, "rfilename", ""),
                        "size": getattr(sibling, "size", None),
                    }
                    for sibling in (info.siblings or [])
                ],
            }
        )
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=target,
            resume_download=True,
        )
        actual_bytes = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
        result.update(
            {
                "status": "downloaded",
                "snapshot_path": local_path,
                "actual_bytes": actual_bytes,
                "duration_seconds": time.perf_counter() - started,
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "duration_seconds": time.perf_counter() - started,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(ARTIFACT_ROOT / "data/raw/hf"))
    parser.add_argument(
        "--manifest",
        default=str(ARTIFACT_ROOT / "data/raw/hf_download_manifest.json"),
    )
    parser.add_argument("--include_eval", action="store_true")
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    selected = dict(DATASETS)
    if args.include_eval:
        selected.update(EVAL_DATASETS)
    if args.only:
        unknown = sorted(set(args.only) - set(selected))
        if unknown:
            raise SystemExit(f"Unknown dataset aliases: {unknown}")
        selected = {alias: selected[alias] for alias in args.only}

    results = []
    for alias, repo_id in selected.items():
        print(f"Downloading {alias}: {repo_id}")
        result = download_one(alias, repo_id, out_dir)
        results.append(result)
        print(f"  {result['status']}: {result.get('actual_bytes', 0)} bytes")
        manifest_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote manifest to {manifest_path}")
    failed = [row for row in results if row["status"] != "downloaded"]
    if failed:
        print("Failures:")
        for row in failed:
            print(f"- {row['alias']}: {row.get('error_type')} {row.get('error')}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
