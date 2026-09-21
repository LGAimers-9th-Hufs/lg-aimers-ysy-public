#!/usr/bin/env python3
"""Package DSF with multiple FactorTabM seeds and their OOF router."""
from __future__ import annotations

import argparse,shutil,tempfile,zipfile
from pathlib import Path
import joblib

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dsf_dir",type=Path,default=Path("./DSF_CHAL061_FIX")); ap.add_argument("--factor_dirs",nargs="+",type=Path,required=True); ap.add_argument("--router_dir",type=Path,default=Path("./factor_tabm_ensemble_final")); ap.add_argument("--output",type=Path,default=Path("./submit_dsf_factor_tabm_3seed_router.zip")); args=ap.parse_args()
    preprocessors=[joblib.load(directory/"preprocessor.joblib") for directory in args.factor_dirs]
    reference=preprocessors[0]
    for candidate in preprocessors[1:]:
        if candidate.cat_cols!=reference.cat_cols or candidate.num_cols!=reference.num_cols or candidate.maps!=reference.maps: raise RuntimeError("preprocessor mismatch")
    with tempfile.TemporaryDirectory(prefix="dsf_factor_ensemble_") as temporary:
        package=Path(temporary)/"package"; package.mkdir(); shutil.copytree(args.dsf_dir,package/"dsf"); (package/"dsf"/"script.py").rename(package/"dsf"/"member_script.py"); model=package/"model"; model.mkdir(); shutil.copy2(args.factor_dirs[0]/"preprocessor.joblib",model)
        for index,directory in enumerate(args.factor_dirs): shutil.copy2(directory/"factor_tabm.pt",model/f"factor_tabm_{index}.pt")
        shutil.copy2(args.router_dir/"router.joblib",model); shutil.copy2(args.router_dir/"metadata.json",model); shutil.copy2(HERE/"inference_factor_tabm_ensemble_router.py",package/"script.py")
        for relative in ("pipelines/__init__.py","pipelines/dsf_upgrade/__init__.py","pipelines/dsf_upgrade/factor_tabm.py","pipelines/mlp/__init__.py","pipelines/mlp/train_walkforward.py","shared/__init__.py","shared/catboost_features.py"):
            target=package/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/relative,target)
        (package/"requirements.txt").write_text("numpy\npandas\nscikit-learn==1.8.0\njoblib\nlightgbm==4.6.0\ntorch==2.8.0\n",encoding="utf-8")
        with zipfile.ZipFile(args.output,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file(): archive.write(path,path.relative_to(package).as_posix())
    print(f"saved {args.output.resolve()} bytes={args.output.stat().st_size}")


if __name__=="__main__": main()
