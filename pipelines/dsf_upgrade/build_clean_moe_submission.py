#!/usr/bin/env python3
"""Build the fully reproducible clean residual MoE submission."""
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _write_inference_only_challenger(source: Path, target: Path) -> None:
    """Remove label-fitting helpers that are unreachable during inference."""
    text = source.read_text(encoding="utf-8")
    start = text.index("KAPPA = {")
    resume = text.index("def _lookup_values", start)
    text = text[:start] + text[resume:]
    start = text.index("def build_time_safe_matrix")
    resume = text.index("def feature_groups", start)
    target.write_text(text[:start] + text[resume:], encoding="utf-8")


def _write_inference_only_trackman(source: Path, target: Path) -> None:
    """Package only the row-local frozen-artifact lookup used by serving."""
    text = source.read_text(encoding="utf-8")
    start = text.index("def enrich_command")
    header = '''\
"""Inference-only TrackMan command features from frozen official-data artifacts."""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS = ("fastball", "breaking", "offspeed")
PROFILE_METRICS = (
    "release_side_std", "release_height_std", "release_ellipse", "release_major",
    "release_minor", "release_corr", "speed_std", "spin_std", "extension_std",
    "movement_ellipse", "movement_major", "movement_minor", "movement_corr",
    "active_spin_proxy_mean", "active_spin_proxy_std", "velo_retention_std",
    "drift_release", "drift_speed", "drift_movement",
)

'''
    target.write_text(header + text[start:], encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenger_dir", type=Path, default=Path("./DSF_CHAL061_FIX/challenger"))
    parser.add_argument("--factor_dirs", nargs=3, type=Path, default=[Path("./factor_tabm_final"), Path("./factor_tabm_final_seed1"), Path("./factor_tabm_final_seed2")])
    parser.add_argument("--track_dir", type=Path, default=Path("./factor_tabm_trackman_final"))
    parser.add_argument("--moe_dir", type=Path, default=Path("./clean_moe_final"))
    parser.add_argument("--output", type=Path, default=Path("./clean_moe.zip"))
    args = parser.parse_args()
    if len(args.output.name) > 20:
        raise ValueError("ZIP filename must be at most 20 characters")
    with tempfile.TemporaryDirectory(prefix="clean_moe_") as temporary:
        package = Path(temporary) / "package"
        package.mkdir()
        shutil.copytree(args.challenger_dir, package / "challenger")
        challenger_script = package / "challenger" / "script.py"
        if challenger_script.exists():
            challenger_script.rename(package / "challenger" / "member_script.py")
        if not (package / "challenger" / "member_script.py").exists():
            raise FileNotFoundError("Challenger inference entry point is missing")
        _write_inference_only_challenger(
            args.challenger_dir / "feature_engine.py",
            package / "challenger" / "feature_engine.py",
        )
        model = package / "model"
        model.mkdir()
        for index, directory in enumerate(args.factor_dirs):
            shutil.copy2(directory / "factor_tabm.pt", model / f"factor_{index}.pt")
            shutil.copy2(directory / "preprocessor.joblib", model / f"preprocessor_{index}.joblib")
        shutil.copy2(args.track_dir / "factor_tabm.pt", model / "track_factor.pt")
        shutil.copy2(args.track_dir / "preprocessor.joblib", model / "track_preprocessor.joblib")
        shutil.copy2(args.track_dir / "trackman_command_artifacts.json", model / "trackman_command_artifacts.json")
        shutil.copy2(args.moe_dir / "clean_moe.joblib", model / "clean_moe.joblib")
        shutil.copy2(args.moe_dir / "metadata.json", model / "metadata.json")
        shutil.copy2(HERE / "inference_clean_moe.py", package / "script.py")
        for relative in (
            "pipelines/__init__.py", "pipelines/dsf_upgrade/__init__.py",
            "pipelines/dsf_upgrade/factor_tabm.py", "pipelines/dsf_upgrade/residual_moe.py",
            "pipelines/mlp/__init__.py", "pipelines/mlp/train_walkforward.py",
            "pipelines/trackman/__init__.py", "pipelines/trackman/command_features.py",
            "shared/__init__.py", "shared/catboost_features.py",
        ):
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        _write_inference_only_trackman(
            ROOT / "pipelines/trackman/command_features.py",
            package / "pipelines/trackman/command_features.py",
        )
        (package / "requirements.txt").write_text("lightgbm==4.6.0\n", encoding="utf-8")
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package).as_posix())
    print(f"saved {args.output.resolve()} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
