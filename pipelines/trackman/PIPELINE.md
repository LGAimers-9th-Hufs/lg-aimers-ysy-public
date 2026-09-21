# TrackMan 물리 프로파일 파이프라인

## 목적과 특징

주최 측이 제공한 2019~2024 `trackman_history.csv`에서 손 유형·구종군별 물리 프로파일을 미리 만들고, 각 행의 공식 `asof_*` 구종 비율로 기대 물리 특성을 계산하는 CatBoost 후보입니다. 메인 데이터와 TrackMan을 선수 ID로 억지 매칭하지 않습니다.

## 구성

- artifact/피처 생성: `features.py`
- walk-forward 평가: `experiments/evaluate_trackman_walkforward.py`
- 최종 학습: `train.py`
- 성분 추론: `inference_component.py`
- 상세 결과: `REPORT.md`
- 산출물: `trackman_catboost.cbm`, `trackman_artifacts.json`, `artifacts.json`

## 실행

```bash
python -m pipelines.trackman.train --train_path ./data/train.csv --trackman_path ./data/trackman_history.csv --out_dir ./model_trackman
```

## 규정 준수와 한계

공식 제공 이력만 사용하며 2025 TrackMan이나 현재 투구 측정값은 사용하지 않습니다. test 내부 통계도 만들지 않습니다. 시즌 경계를 넘는 집계는 반드시 예측 행보다 과거 정보인지 재검증해야 하며, TrackMan artifact 생성 규칙 변경 시 OOF도 동일 규칙으로 다시 계산해야 합니다.
