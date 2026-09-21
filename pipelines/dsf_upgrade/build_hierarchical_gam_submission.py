#!/usr/bin/env python3
"""Package frozen DSF with a conservative hierarchical GAM member."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dsf_dir", type=Path, default=Path("./DSF_CHAL061_FIX"))
    ap.add_argument("--gam_dir", type=Path, default=Path("./hierarchical_gam_final")); ap.add_argument("--gam_weight", type=float, default=.05)
    ap.add_argument("--output", type=Path, default=Path("./submit_dsf_hierarchical_gam_w05.zip")); args = ap.parse_args()
    if not 0 <= args.gam_weight <= 1: raise ValueError("gam_weight must be in [0,1]")
    with tempfile.TemporaryDirectory(prefix="dsf_gam_") as temporary:
        package = Path(temporary) / "package"; package.mkdir(); shutil.copytree(args.dsf_dir, package / "dsf")
        (package / "dsf" / "script.py").rename(package / "dsf" / "member_script.py")
        (package / "model").mkdir(); shutil.copy2(args.gam_dir / "hierarchical_gam.joblib", package / "model")
        (package / "model" / "metadata.json").write_text(json.dumps({
            "gam_weight": args.gam_weight, "selection": "fixed conservative architecture probe; no leaderboard fitting",
            "rules": "no test-row aggregation; official train only"
        }, indent=2), encoding="utf-8")
        shutil.copy2(HERE / "inference_hierarchical_gam.py", package / "script.py")
        module_dir = package / "pipelines" / "dsf_upgrade"; module_dir.mkdir(parents=True)
        (package / "pipelines" / "__init__.py").write_text("", encoding="utf-8")
        (module_dir / "__init__.py").write_text("", encoding="utf-8")
        shutil.copy2(HERE / "hierarchical_gam.py", module_dir / "hierarchical_gam.py")
        (package / "requirements.txt").write_text("numpy\npandas\nscipy\nscikit-learn==1.8.0\njoblib\nlightgbm==4.6.0\n", encoding="utf-8")
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file(): archive.write(path, path.relative_to(package).as_posix())
    print(f"saved {args.output.resolve()} bytes={args.output.stat().st_size}")


if __name__ == "__main__": main()
