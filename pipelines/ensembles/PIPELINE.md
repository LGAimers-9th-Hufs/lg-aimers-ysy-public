# 앙상블 파이프라인

## 목적과 특징

단순 seed 다양화보다 서로 다른 모델의 **OOF residual correlation**을 기준으로 조합을 탐색합니다. CatBoost, MLP, TrackMan, V3/Brier 성분을 동일한 시간순 검증 행에 정렬해 블렌딩 후보를 평가합니다.

## 구성

- residual diversity: `experiments/evaluate_residual_diversity.py`
- CatBoost+MLP+TrackMan 평가: `experiments/evaluate_trackman_ensemble.py`
- 캐시된 TrackMan OOF 결합: `experiments/combine_trackman_cached_oof.py`
- temporal TE/Brier 후보: `experiments/evaluate_walkforward_*.py`
- 제출 추론: `script_residual_ensemble.py`, `script_trackman_ensemble.py`
- 결과: `EXPERIMENT_REPORT.md`, `FINAL_PIPELINE_REPORT.md`

## 실행 원칙

앙상블에 넣는 모든 OOF는 동일한 `row_id`와 동일한 시간순 fold에서 생성되어야 합니다. 혼합 가중치와 calibration은 별도 train 검증 구간에서만 학습하고 untouched 구간으로 최종 비교합니다.

## 규정 준수와 한계

추론 시 각 기반 모델은 현재 test 행만 사용하며 혼합은 고정 가중 합입니다. 리더보드 점수에 맞춰 가중치나 예측 분포를 조정하면 안 됩니다. 추론 스크립트는 제출 패키지 안의 helper 모듈과 하위 모델 디렉터리를 전제로 하므로, 패키징 검사 없이는 독립 실행 파일로 간주하면 안 됩니다.
