#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-lookup", default="<repo>/model_artifacts/clean_lookup/package")
    parser.add_argument("--model", default="./clean_rescat_model")
    parser.add_argument("--output", default="./clean_rescat.zip")
    parser.add_argument("--artifact", default="./model_artifacts/clean_rescat")
    args = parser.parse_args()
    output = Path(args.output)
    if len(output.name) > 20:
        raise RuntimeError("ZIP filename must not exceed 20 characters")
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    with tempfile.TemporaryDirectory(prefix="clean_rescat_") as temporary:
        package = Path(temporary) / "package"
        package.mkdir()
        shutil.copytree(args.clean_lookup, package / "clean_lookup")
        shutil.copytree(args.model, package / "model")
        shutil.copy2(here / "script.py", package / "script.py")
        target = package / "pipelines" / "clean_rescat"
        target.mkdir(parents=True)
        shutil.copy2(here / "features.py", target / "features.py")
        (package / "pipelines" / "__init__.py").write_text("", encoding="utf-8")
        (target / "__init__.py").write_text("", encoding="utf-8")
        requirements = (package / "clean_lookup" / "requirements.txt").read_text(encoding="utf-8")
        if "catboost" not in requirements.lower():
            requirements += "\ncatboost==1.2.8\n"
        (package / "requirements.txt").write_text(requirements, encoding="utf-8")
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package).as_posix())
        artifact = Path(args.artifact)
        if artifact.exists():
            shutil.rmtree(artifact)
        shutil.copytree(package, artifact / "package")
        shutil.copy2(output, artifact / "submission.zip")
    print(f"built {output} bytes={output.stat().st_size:,}")


if __name__ == "__main__":
    main()
