#!/usr/bin/env python3
"""Build clean_regime.zip with clean MoE and newly trained regime models."""
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REGIME_FEATURE_SOURCE = ROOT / "model_artifacts/factor_tabm_router/package/dsf/champion/regime_features.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_package", type=Path, default=Path("./model_artifacts/clean_moe/package"))
    parser.add_argument("--regime_dir", type=Path, default=Path("./regime_stack_final"))
    parser.add_argument("--output", type=Path, default=Path("./clean_regime.zip"))
    args = parser.parse_args()
    if len(args.output.name) > 20:
        raise ValueError("ZIP filename must be at most 20 characters")
    with tempfile.TemporaryDirectory(prefix="clean_regime_") as temporary:
        package = Path(temporary) / "package"
        package.mkdir()
        shutil.copytree(args.clean_package, package / "clean")
        clean_script = package / "clean" / "script.py"
        clean_script.rename(package / "clean" / "member_script.py")
        regime = package / "regime"
        (regime / "model").mkdir(parents=True)
        shutil.copy2(REGIME_FEATURE_SOURCE, regime / "features.py")
        for filename in ("specialist.txt", "state.txt", "invariant.txt", "metadata.json"):
            shutil.copy2(args.regime_dir / filename, regime / "model" / filename)
        shutil.copy2(HERE / "inference.py", package / "script.py")
        (package / "requirements.txt").write_text("lightgbm==4.6.0\n", encoding="utf-8")
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package).as_posix())
    print(f"saved {args.output.resolve()} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
