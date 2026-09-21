#!/usr/bin/env python3
"""Build a DSF+V3 submission without mutating either frozen source package."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent


def copy_member(source: Path, target: Path) -> None:
    shutil.copytree(source, target)
    script = target / "script.py"
    script.rename(target / "member_script.py")
    requirements = target / "requirements.txt"
    if requirements.exists():
        requirements.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsf_dir", type=Path, default=Path("./DSF_CHAL061_FIX"))
    parser.add_argument("--v3_dir", type=Path, default=Path("./submit_v3"))
    parser.add_argument("--v3_weight", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=Path("./submit_dsf_v3_w20.zip"))
    args = parser.parse_args()
    if not 0.0 <= args.v3_weight <= 1.0:
        raise ValueError("v3_weight must be in [0, 1]")
    for source in (args.dsf_dir, args.v3_dir):
        if not (source / "script.py").is_file() or not (source / "model").is_dir():
            raise FileNotFoundError(f"invalid source submission: {source}")

    with tempfile.TemporaryDirectory(prefix="dsf_v3_package_") as temporary:
        package = Path(temporary) / "package"
        package.mkdir()
        copy_member(args.dsf_dir, package / "dsf")
        copy_member(args.v3_dir, package / "v3")
        (package / "model").mkdir()
        metadata = {
            "schema_version": 1,
            "pipeline": "frozen_DSF_CHAL061_FIX_plus_V3",
            "dsf_weight": 1.0 - args.v3_weight,
            "v3_weight": args.v3_weight,
            "weight_selection": "2022/2023 train OOF stability with 2024 untouched confirmation",
            "rules": "fixed row-aligned blend; no evaluation-batch statistics or leaderboard-derived fitting",
        }
        (package / "model" / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        shutil.copy2(HERE / "inference.py", package / "script.py")
        (package / "requirements.txt").write_text(
            "lightgbm==4.6.0\njoblib==1.5.3\nscikit-learn==1.8.0\n", encoding="utf-8"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package).as_posix())
    print(f"saved {args.output.resolve()} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
