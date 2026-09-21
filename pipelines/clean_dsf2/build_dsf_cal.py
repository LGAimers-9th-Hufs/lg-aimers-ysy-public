#!/usr/bin/env python3
"""Build probe-free DSF with a frozen 2024 official-train OOF calibration."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INTERCEPT = -0.008228836300836177
SLOPE = 0.8946876621596727


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "model_artifacts/dsf_clean/package")
    parser.add_argument("--output", type=Path, default=ROOT / "dsf_cal.zip")
    parser.add_argument("--artifact", type=Path, default=ROOT / "model_artifacts/dsf_cal")
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    if len(args.output.name) > 20:
        raise RuntimeError("ZIP filename must not exceed 20 characters")

    with tempfile.TemporaryDirectory(prefix="dsf_cal_") as temporary:
        package = Path(temporary) / "package"
        shutil.copytree(args.source, package)
        script_path = package / "script.py"
        script = script_path.read_text(encoding="utf-8")
        source = "prediction = np.clip((1.0 - weight) * champion + weight * challenger, 0.0, 1.0)"
        if not 0.0 <= args.strength <= 1.0:
            raise RuntimeError("calibration strength must be between 0 and 1")
        effective_intercept = args.strength * INTERCEPT
        effective_slope = 1.0 + args.strength * (SLOPE - 1.0)
        replacement = (
            "prediction = np.clip((1.0 - weight) * champion + weight * challenger, 0.0, 1.0)\n"
            f"    prediction = np.clip(0.5 + ({effective_intercept!r}) + ({effective_slope!r}) * (prediction - 0.5), 0.0, 1.0)"
        )
        if script.count(source) != 1:
            raise RuntimeError("unexpected DSF blend implementation")
        script_path.write_text(script.replace(source, replacement), encoding="utf-8")

        metadata_path = package / "model/metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["calibration"] = {
            "type": "affine_centered",
            "intercept": INTERCEPT,
            "slope": SLOPE,
            "strength": args.strength,
            "effective_intercept": effective_intercept,
            "effective_slope": effective_slope,
            "fit_source": "2024 official-train walk-forward OOF only",
            "test_statistics_used": False,
            "leaderboard_feedback_used": False,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

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
