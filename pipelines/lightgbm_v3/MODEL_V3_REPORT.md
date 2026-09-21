# Model V3 보고서

## 결론

현재 837점 V2를 보존하고, 공식 train만 사용한 V3 후보를 별도로 만들었다.

- 제출 파일: `submit_v3.zip`
- 학습 코드: `train_v3.py`
- 추론 코드: 기존 검증 완료 `script.py`
- 모델: LightGBM 3개(실제 blend에서 모델 1·2 사용)
- test 행 간 집계·빈도·rolling·expanding·분포 보정 없음
- 외부 데이터 미사용

## 변경점

V2의 79개 피처와 행 독립 추론 코드는 그대로 유지했다. 검증에서 새 상호작용 피처와 native categorical은 이득이 없거나 과적합하여 최종 제출에서 제외했다.

V3에서 채택한 변경은 두 가지다.

1. 최근 시즌 학습 가중치를 `1 + 0.18 × (season - 2019)`로 강화
2. 2024 calibration 전용 구간에서 Brier Score를 최소화하도록 nonnegative/sum-to-one blend weight 적합

최종 blend weight:

- `lgb0`: 0.000000
- `lgb1`: 0.812062
- `lgb2`: 0.187938

## 엄격한 검증

2024를 안정적인 row_id hash mod 4로 분리했다.

- early stopping: hash 0/1, 126,338행
- blend/calibration fit: hash 2, 63,418행
- untouched final evaluation: hash 3, 63,751행

| 구성 | Untouched 2024 BSS |
|---|---:|
| 동일 분할 V2 기준선 | 628.853 |
| V3 최근 가중+최적 blend | **646.048** |
| 개선 | **+17.195** |

V3 raw blend BSS는 609.935, logit-affine calibration 적용 후 646.048이다.

## 탈락 실험

- 선수 ID native categorical 포함: 391.323
- 저카디널리티 native categorical + 새 상호작용: 627.130
- 새 상호작용만: 629.124

탈락 실험은 `submit_v3.zip`에 포함하지 않았다. 리더보드 837점은 모델 선택, calibration 또는 상수 역산에 사용하지 않았다.

## 규정 준수

V3 추론은 각 test 행의 값과 공식 train으로 사전 저장한 artifact lookup만 사용한다. test 다른 행, test 빈도, test 분포, test prediction 평균·순위·분위수를 사용하지 않는다.
