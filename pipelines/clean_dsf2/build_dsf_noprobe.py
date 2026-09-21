#!/usr/bin/env python3
"""Build the team DSF package with all leaderboard-probe constants disabled."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def clean_metadata(path: Path) -> None:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.pop("seg_probe", None)
    metadata.pop("regime_transfer_probe", None)
    provenance = metadata.get("provenance")
    if isinstance(provenance, list):
        metadata["provenance"] = [
            item for item in provenance
            if "public score" not in str(item).lower() and "leaderboard" not in str(item).lower()
        ]
    metadata["evaluation_feedback_removal"] = (
        "all evaluation-feedback-derived adjustment constants are removed and disabled"
    )
    path.write_text(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "model_artifacts/DSF_CHAL061_FIX/package")
    parser.add_argument("--output", type=Path, default=ROOT / "dsf_clean.zip")
    parser.add_argument("--artifact", type=Path, default=ROOT / "model_artifacts/dsf_clean")
    args = parser.parse_args()
    if len(args.output.name) > 20:
        raise RuntimeError("ZIP filename must not exceed 20 characters")
    with tempfile.TemporaryDirectory(prefix="dsf_clean_") as temporary:
        package = Path(temporary) / "package"
        shutil.copytree(args.source, package)
        clean_metadata(package / "champion/model/metadata.json")
        engine_path = package / "champion/model/dsf_engine.py"
        engine = engine_path.read_text(encoding="utf-8")
        source = 'terms = meta.get("seg_probe")'
        replacement = "terms = []  # evaluation-feedback terms permanently disabled"
        if engine.count(source) != 1:
            raise RuntimeError("unexpected segment-adjustment implementation")
        engine_path.write_text(engine.replace(source, replacement), encoding="utf-8")
        # The outer 0.061 challenger weight is retained: it was derived from the
        # supplied 2022-2024 walk-forward OOF, not leaderboard feedback.
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package))
        unpacked = args.artifact / "package"
        if unpacked.exists():
            shutil.rmtree(unpacked)
        shutil.copytree(package, unpacked)
    args.artifact.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output, args.artifact / "submission.zip")
    print(f"built {args.output} bytes={args.output.stat().st_size:,}")


if __name__ == "__main__":
    main()
