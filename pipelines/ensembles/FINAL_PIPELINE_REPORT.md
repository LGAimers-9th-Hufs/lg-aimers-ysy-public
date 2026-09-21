# 최종 파이프라인 실험 보고서

작성일: 2026-08-24

## 결론

최종 추천 제출물은 `submit_residual_ensemble.zip`이다. 기존 전체기간+최근 3시즌 CatBoost보다 엄격한 2024 untouched 평가에서 Brier Skill Score가 **664.891 → 680.763 (+15.872)**로 개선됐다.

## 규정 준수 경계

- 모든 모델, 가중치, calibration, 피처 선택은 공식 `train.csv`만 사용했다.
- test 행 간 집계, 빈도, rolling, expanding, target encoding, 분포 보정은 사용하지 않았다.
- 외부 데이터와 리더보드 점수는 학습·선택에 사용하지 않았다.
- 추론 피처는 현재 한 행의 값만으로 계산한다. test 행 순서 변경 및 단일 행 추론 동일성 검사를 통과했다.
- calibration/앙상블 가중치는 2024 `row_id` hash mod 4의 calibration 조각에서 맞추고, hash mod 4 = 3 조각은 최종 비교에만 사용했다.

## 완료한 네 후보

| 후보 | 엄격 2024 BSS | 결과 |
|---|---:|---|
| PyTorch embedding MLP walk-forward | 573.009 | 단독 성능은 낮지만 CatBoost와의 혼합 다양성 제공 |
| shared-trunk regime interaction CatBoost | 659.434 | 기존 최고 미달 |
| 잔차 상관 기반 CatBoost+MLP 혼합 | **680.763** | 최종 추천 |
| 2021--2023 fold-stability 45피처 CatBoost | 647.459 | 기존 최고 미달 |

잔차 최적 혼합 가중치는 전체기간 CatBoost `0.387748`, 최근 3시즌 CatBoost `0.273565`, embedding MLP `0.338687`이다. untouched 평가에서 세 모델의 잔차 상관은 약 `0.9989--0.9997`로 높았지만, 작은 비상관 성분도 Brier 점수에는 유의미한 개선을 만들었다. 상관 자체를 강제로 0.99 이하로 제한한 후보는 현실적으로 존재하지 않았으며, 가장 다양한 50:50 recent+MLP 후보는 BSS 661.841로 더 낮았다. 따라서 최종안은 상관을 낮추는 것을 목적함수로 삼지 않고 calibration 구간의 Brier loss를 직접 최소화한 convex blend다.

## 최종 패키지 감사

- ZIP 무결성 검사: 통과
- 245,789행 행 단위 추론: 1.18초 (모델 로딩 제외), peak RSS 약 1.80 GiB
- NaN/Inf 및 확률 범위 검사: 통과
- test 행 순서 변경/단일 행 분리 추론: 동일
- train/inference 피처 구현: 각 최종 학습 artifact와 동봉된 추론 helper가 동일 스키마를 검증
- 모델/스크립트: CatBoost 2개, MLP checkpoint, preprocessing artifact의 개수와 경로를 실행 시 검증

## 제출 파일

- 추천: `submit_residual_ensemble.zip`
- 안정 피처 비교 후보: `submit_stable_features.zip`
- 기존 안전 기준선: `submit_catboost_blend.zip`

검증 향상은 로컬 strict OOF 기준이며 실제 리더보드 향상을 보장하지 않는다. 안정 피처 후보는 실험 완료 증빙용으로 만들었지만 점수가 낮으므로 제출을 권하지 않는다.
