#!/usr/bin/env python3
"""Build clean_dsf2.zip and its identical unpacked artifact."""
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "clean_dsf2.zip")
    parser.add_argument("--artifact", type=Path, default=ROOT / "model_artifacts/clean_dsf2")
    args = parser.parse_args()
    if len(args.output.name) > 20:
        raise RuntimeError("ZIP filename must not exceed 20 characters")
    with tempfile.TemporaryDirectory(prefix="clean_dsf2_") as temporary:
        package = Path(temporary) / "package"
        package.mkdir()
        shutil.copytree(ROOT / "model_artifacts/dsf_clean/package", package / "dsf")
        shutil.copytree(ROOT / "model_artifacts/clean_lookup/package", package / "clean")
        shutil.copy2(ROOT / "pipelines/clean_dsf2/inference.py", package / "script.py")
        shutil.copy2(ROOT / "model_artifacts/clean_lookup/package/requirements.txt", package / "requirements.txt")
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
