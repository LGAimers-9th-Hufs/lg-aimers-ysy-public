#!/usr/bin/env python3
"""Build the seed-specific FactorTabM router with a TrackMan safety gate."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsf_dir", type=Path, default=Path("./DSF_CHAL061_FIX"))
    parser.add_argument("--factor_dir", type=Path, default=Path("./factor_tabm_final_seed1"))
    parser.add_argument("--track_dir", type=Path, default=Path("./factor_tabm_trackman_final"))
    parser.add_argument("--output", type=Path, default=Path("./seed1_tm_gate.zip"))
    args = parser.parse_args()
    if len(args.output.name) > 20:
        raise ValueError("ZIP filename must be at most 20 characters")

    with tempfile.TemporaryDirectory(prefix="factor_trackman_safety_") as temporary:
        package = Path(temporary) / "package"
        package.mkdir()
        shutil.copytree(args.dsf_dir, package / "dsf")
        (package / "dsf" / "script.py").rename(package / "dsf" / "member_script.py")
        model = package / "model"
        model.mkdir()
        for source, name in (
            (args.factor_dir / "factor_tabm.pt", "factor_tabm.pt"),
            (args.factor_dir / "preprocessor.joblib", "preprocessor.joblib"),
            (args.factor_dir / "router.joblib", "router.joblib"),
            (args.track_dir / "factor_tabm.pt", "track_factor.pt"),
            (args.track_dir / "preprocessor.joblib", "track_preprocessor.joblib"),
            (args.track_dir / "trackman_command_artifacts.json", "trackman_command_artifacts.json"),
        ):
            shutil.copy2(source, model / name)
        (model / "metadata.json").write_text(
            json.dumps(
                {
                    "router_threshold": 0.50,
                    "router_alpha": 0.30,
                    "track_strength_cutoff": 0.011956274509429932,
                    "disagreement_scale": 0.0,
                    "factor_seed": 60025,
                    "selection": "equal-season normalized Brier on 2023/2024 OOF select hashes",
                    "rules": "official train and TrackMan only; train-frozen row-local inference",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        shutil.copy2(HERE / "inference_factor_trackman_safety.py", package / "script.py")
        for relative in (
            "pipelines/__init__.py",
            "pipelines/dsf_upgrade/__init__.py",
            "pipelines/dsf_upgrade/factor_tabm.py",
            "pipelines/mlp/__init__.py",
            "pipelines/mlp/train_walkforward.py",
            "pipelines/trackman/__init__.py",
            "pipelines/trackman/command_features.py",
            "shared/__init__.py",
            "shared/catboost_features.py",
        ):
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        # The evaluation image already provides NumPy, pandas, sklearn,
        # joblib, and PyTorch.  Installing only LightGBM avoids a large torch
        # download within the competition's ten-minute installation limit.
        (package / "requirements.txt").write_text("lightgbm==4.6.0\n", encoding="utf-8")
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package).as_posix())
    print(f"saved {args.output.resolve()} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
