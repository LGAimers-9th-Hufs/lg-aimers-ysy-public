#!/usr/bin/env python3
"""Build the frozen DSF + FactorTabM conditional-router submission."""
from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dsf_dir",type=Path,default=Path("./DSF_CHAL061_FIX")); ap.add_argument("--factor_dir",type=Path,default=Path("./factor_tabm_final")); ap.add_argument("--output",type=Path,default=Path("./submit_dsf_factor_tabm_router.zip")); args=ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="dsf_factor_router_") as temporary:
        package=Path(temporary)/"package"; package.mkdir(); shutil.copytree(args.dsf_dir,package/"dsf"); (package/"dsf"/"script.py").rename(package/"dsf"/"member_script.py")
        shutil.copytree(args.factor_dir,package/"model"); shutil.copy2(HERE/"inference_factor_tabm_router.py",package/"script.py")
        for relative in ("pipelines/__init__.py","pipelines/dsf_upgrade/__init__.py","pipelines/dsf_upgrade/factor_tabm.py","pipelines/mlp/__init__.py","pipelines/mlp/train_walkforward.py","shared/__init__.py","shared/catboost_features.py"):
            target=package/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/relative,target)
        (package/"requirements.txt").write_text("numpy\npandas\nscikit-learn==1.8.0\njoblib\nlightgbm==4.6.0\ntorch==2.8.0\n",encoding="utf-8")
        with zipfile.ZipFile(args.output,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file(): archive.write(path,path.relative_to(package).as_posix())
    print(f"saved {args.output.resolve()} bytes={args.output.stat().st_size}")


if __name__=="__main__": main()
