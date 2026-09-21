#!/usr/bin/env python3
"""Build champion FactorTabM plus TrackMan-command FactorTabM submission."""
from __future__ import annotations

import argparse,json,shutil,tempfile,zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dsf_dir",type=Path,default=Path("./DSF_CHAL061_FIX"));ap.add_argument("--base_dir",type=Path,default=Path("./factor_tabm_final"));ap.add_argument("--track_dir",type=Path,default=Path("./factor_tabm_trackman_final"));ap.add_argument("--output",type=Path,default=Path("./tm_factor.zip"));args=ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="dsf_factor_trackman_") as temporary:
        package=Path(temporary)/"package";package.mkdir();shutil.copytree(args.dsf_dir,package/"dsf");(package/"dsf"/"script.py").rename(package/"dsf"/"member_script.py");model=package/"model";model.mkdir()
        for source,name in ((args.base_dir/"factor_tabm.pt","base_factor.pt"),(args.base_dir/"preprocessor.joblib","base_preprocessor.joblib"),(args.base_dir/"router.joblib","router.joblib"),(args.track_dir/"factor_tabm.pt","track_factor.pt"),(args.track_dir/"preprocessor.joblib","track_preprocessor.joblib"),(args.track_dir/"trackman_command_artifacts.json","trackman_command_artifacts.json")):shutil.copy2(source,model/name)
        (model/"metadata.json").write_text(json.dumps({"router_threshold":.5,"router_alpha":.3,"track_factor_weight":.2,"track_candidate_share":.15,"selection":"equal-season normalized Brier on 2023/2024 select hashes","rules":"official train and TrackMan only; train-frozen row-local inference"},indent=2),encoding="utf-8")
        shutil.copy2(HERE/"inference_factor_tabm_trackman_blend.py",package/"script.py")
        for relative in ("pipelines/__init__.py","pipelines/dsf_upgrade/__init__.py","pipelines/dsf_upgrade/factor_tabm.py","pipelines/mlp/__init__.py","pipelines/mlp/train_walkforward.py","pipelines/trackman/__init__.py","pipelines/trackman/command_features.py","shared/__init__.py","shared/catboost_features.py"):
            target=package/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
        (package/"requirements.txt").write_text("numpy\npandas\nscikit-learn==1.8.0\njoblib\nlightgbm==4.6.0\ntorch==2.8.0\n",encoding="utf-8")
        with zipfile.ZipFile(args.output,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():archive.write(path,path.relative_to(package).as_posix())
    print(f"saved {args.output.resolve()} bytes={args.output.stat().st_size}")


if __name__=="__main__":main()
