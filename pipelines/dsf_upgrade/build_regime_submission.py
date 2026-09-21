#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsf_dir", default="./DSF_CHAL061_FIX")
    ap.add_argument("--output", default="./submit_dsf_regime_w25.zip")
    ap.add_argument("--r_weight", type=float, default=0.25)
    ap.add_argument("--f_shift", type=float, default=0.0)
    ap.add_argument("--r_beta", nargs=3, type=float)
    args = ap.parse_args()
    source = Path(args.dsf_dir).resolve()
    with tempfile.TemporaryDirectory(prefix="dsf-regime-") as tmp:
        package = Path(tmp)
        shutil.copytree(source / "champion", package / "champion")
        shutil.copytree(source / "challenger", package / "challenger")
        shutil.copy2(Path(__file__).with_name("inference_regime_blend.py"), package / "script.py")
        (package / "model").mkdir()
        metadata = {
            "version": 1,
            "challenger_weight_by_game_type": {"F": 0.061, "R": args.r_weight},
            "fallback_weight": 0.061,
            "probability_shift_by_game_type": {"F": args.f_shift},
            "beta_calibration_by_game_type": ({"R": args.r_beta} if args.r_beta else {}),
            "selection": "weights and optional F shift are fixed train-OOF artifacts",
        }
        (package / "model/metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (package / "requirements.txt").write_text("lightgbm==4.6.0\njoblib==1.5.3\nscikit-learn==1.8.0\n", encoding="utf-8")
        output = Path(args.output).resolve()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(package))
    print(f"saved {output} bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
