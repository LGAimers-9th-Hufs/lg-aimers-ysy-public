# 규정 준수 신규 파이프라인 연구 보고서

작성일: 2026-08-21  
기준 제출물: `submit_catboost_blend.zip`  
기준 strict 2024 untouched BSS: **664.891**

## 1. 최종 결론

이번 연구에서 기존 최고 664.891을 의미 있게 넘는 후보는 발견되지 않았다. 따라서 성능이 확인되지 않은 새 모델·제출 ZIP은 만들지 않았고, 기존 `submit_catboost_blend.zip`을 최고 제출물로 유지한다.

가장 유망했던 새 후보는 Brier Score를 직접 최소화하는 CatBoostRegressor였다. walk-forward에서 기존 logloss CatBoost보다 소폭 개선됐지만, 기존 전체기간+최근3시즌 앙상블에 추가하면 strict 2024 BSS가 658.132로 하락했다. 예측 상관이 0.978로 너무 높아 앙상블 다양성이 부족했다.

## 2. 규정 준수 감사

모든 실험은 다음 원칙을 지켰다.

- 공식 `train.csv`만 사용
- test 실제 데이터와 test 행 간 정보 미사용
- test groupby, 빈도, rolling, expanding, shift, lag, 누적 통계 미사용
- test 평균·표준편차·분위수·결측률 등 분포 미사용
- test 예측 평균·순위·분포 보정 미사용
- 리더보드 점수로 target 평균, label, calibration을 역산하지 않음
- 외부 데이터/API 미사용
- 모델 선택·TE·blend·calibration은 train OOF label만 사용
- Trackman soft matching은 키가 불명확해 구현하지 않음

규정상 애매한 방법은 구현하지 않았다.

## 3. 검증 설계

### Walk-forward

- 2019~2021 → 2022
- 2019~2022 → 2023
- 2019~2023 → 2024

각 fold에서 미래 시즌 label은 학습·피처 생성에 사용하지 않았다. 이전 시즌 calibration 실험은 직전 fold의 OOF label과 prediction만 사용했다.

### Strict 2024

2024 `row_id` hash mod 4:

- 0/1: early stopping
- 2: blend weight 및 calibration fit
- 3: untouched final evaluation 63,751행

hash=3 label은 early stopping, 모델 선택, blend, calibration에 사용하지 않았다.

## 4. 전체 실험 비교

| 후보 | 핵심 결과 | 판정 |
|---|---:|---|
| 기존 전체기간+최근3시즌 CatBoost | strict BSS **664.891** | 기준/유지 |
| Temporal TE CatBoost | 2024 raw 668.228 vs baseline 703.746 | 탈락 |
| Brier 직접 CatBoostRegressor | walk-forward raw 평균 1003.740 vs baseline 1001.012 | 보류 |
| 3-way: full+recent+Brier | strict calibrated **658.132** | 탈락 |
| Depth-6 CatBoost | strict calibrated **637.327** | 탈락 |
| game_type F/R experts | strict calibrated **596.419** | 탈락 |

## 5. Walk-forward 결과

### 기존 CatBoost 기준선

| 검증 시즌 | Raw BSS | 직전 시즌 calibration BSS |
|---|---:|---:|
| 2022 | 2299.290 | - |
| 2023 | 0.000 | 0.000 |
| 2024 | 703.746 | 171.220 |

- Raw 평균: 1001.012
- Raw 표준편차: 961.928
- 직전 시즌 calibration 평균(2023~2024): 85.610
- 최악 fold: 0

### Temporal TE

| 검증 시즌 | Raw BSS | 직전 시즌 calibration BSS |
|---|---:|---:|
| 2022 | 2213.595 | - |
| 2023 | 0.000 | 0.000 |
| 2024 | 668.228 | 147.787 |

- Raw 평균: 960.608
- 기준 대비: -40.404
- 직전 시즌 calibration 평균: 73.893
- 기준 대비: -11.717

Temporal TE는 각 행 시즌보다 이전 시즌 label만 사용해 누수 없이 구현했지만 CatBoost ordered CTR과 중복됐고 시즌 이동 오차를 키웠다.

### Brier 직접 회귀

| 검증 시즌 | Raw BSS | 직전 시즌 calibration BSS |
|---|---:|---:|
| 2022 | 2299.616 | - |
| 2023 | 0.000 | 0.000 |
| 2024 | 711.604 | 186.728 |

- Raw 평균: 1003.740
- 기준 대비: +2.728
- 직전 시즌 calibration 평균: 93.364
- 기준 대비: +7.754
- 예측 범위는 모든 fold에서 `[0,1]` 내부였다.

개별 개선 폭이 작아 바로 승격하지 않고 3-way OOF blend를 확인했다.

## 6. 3-way OOF 앙상블

Calibration 구간에서 정한 가중치:

- 전체기간 logloss: 0.321738
- 최근3시즌 logloss: 0.257153
- Brier regressor: 0.421109

Untouched prediction correlation:

| | 전체기간 | 최근3시즌 | Brier |
|---|---:|---:|---:|
| 전체기간 | 1.000 | 0.960 | 0.978 |
| 최근3시즌 | 0.960 | 1.000 | 0.965 |
| Brier | 0.978 | 0.965 | 1.000 |

결과:

- Raw BSS: 632.325
- Calibrated BSS: 658.132
- 기존 2-way: 664.891
- 변화: -6.759

Brier 모델은 개별 walk-forward에서 소폭 좋았지만 기존 모델과 지나치게 유사해 최종 blend를 악화시켰다.

## 7. 추가 탈락 실험

### Depth-6 CatBoost

- Best iteration: 369
- Raw BSS: 617.897
- Calibrated BSS: 637.327

낮은 깊이가 과적합을 줄일 수 있다는 가설이었으나 resolution 손실이 더 컸다.

### game_type experts

- F 모델: train 130,994행, best iteration 218
- R 모델: train 1,090,591행, best iteration 298
- Raw BSS: 565.221
- Calibrated BSS: 596.419

행의 `game_type`만으로 라우팅해 규정에는 맞지만 F 표본 감소와 공통 패턴 공유 손실 때문에 성능이 하락했다.

## 8. 데이터 체제 변화

Walk-forward 결과는 2023 regime change가 매우 크다는 것을 보여준다. 2022에서는 2200 이상의 BSS가 나오지만 2023에는 모든 주요 모델의 raw BSS가 0으로 하락했다. 직전 시즌 calibration 역시 이 변화를 안정적으로 따라가지 못했다.

따라서 한 시즌에서 학습한 calibration 계수를 다음 시즌으로 단순 전이하는 방법은 사용하지 않는 것이 좋다. 2025 target 평균을 추정하거나 하드코딩하는 방법은 규정상 금지되므로 고려하지 않는다.

## 9. 생성된 연구 코드

- `evaluate_walkforward_temporal_te.py`
- `evaluate_walkforward_brier_regressor.py`
- `evaluate_catboost_depth6.py`
- `evaluate_catboost_game_type_experts.py`
- 수정된 `evaluate_catboost_blend.py`의 3-way 실험

모든 코드는 기존 최고 제출물·모델을 덮어쓰지 않는다.

## 10. 최종 제출 판정

새 후보 중 664.891을 의미 있게 넘는 모델이 없으므로 신규 모델/artifact/ZIP은 생성하지 않았다.

현재 유지할 제출물:

- `submit_catboost_blend.zip`
- SHA-256: `37d16edff61c87b7629ebe9f530b6f24ddab108784fe3efd8ae76d563cbfe1a5`

기존 제출물은 이미 다음 검증을 통과했다.

- train/inference feature parity
- 단일행/batch/shuffle 동일성
- ID 완전성
- finite 및 `[0,1]` 확률
- 245,789행 약 0.43초
- peak RSS 약 0.99GB
- 제출 ZIP 구조

## 11. 다음 실험 우선순위

1. CatBoost와 오차 상관이 낮은 PyTorch embedding MLP의 walk-forward OOF
2. `game_type`을 분리하지 않고 regime interaction만 강화한 shared-trunk 모델
3. Brier regressor의 seed/depth 다양화가 아니라 residual correlation을 직접 최소화하는 후보 탐색
4. 2023 regime 이전/이후에 공통으로 안정적인 피처만 선택하는 fold-stability ablation
5. 명확한 공식 key를 확인할 수 있을 때만 Trackman 고정 lookup 실험

PyTorch 모델도 개선과 상보성이 OOF에서 확인될 때만 제출물에 포함한다.
