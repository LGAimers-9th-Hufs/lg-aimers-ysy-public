# V3 중심 앙상블 보고서

작성일: 2026-08-26

## 결론

실제 리더보드 최고 모델인 V3를 주력으로 되돌리고, 예측 상관이 낮은 기존 CatBoost+MLP 모델을 보조로 결합한 `submit_v3_anchor.zip`을 만들었다.

- V3: 80%
- 기존 CatBoost 2 + embedding MLP 앙상블: 20%
- 추가 calibration: 없음. 각 component의 train-only OOF calibration만 유지한다.

## 계층형 TE 실험 결과

`pitcher/batter × game_type/count/hand/inning/pressure` 계층형 시간순 Target Encoding을 구현하고 이전 시즌 데이터만으로 2022--2024 walk-forward 평가했다. 모든 주요 fold에서 V3 기준선보다 낮아 최종 모델에는 포함하지 않았다.

| 시즌 | V3 calibrated BSS | 계층형 TE calibrated BSS |
|---|---:|---:|
| 2022 | 2162.946 | 2127.928 |
| 2023 | 90.206 | 45.322 |
| 2024 | 606.110 | 470.706 |

2023·2024에서 계층형 모델의 최적 blend weight도 0이었으므로 residual correction과 행별 gating으로 확장하지 않았다.

## V3 활용 근거

2024 OOF에서 V3와 기존 CatBoost+MLP 예측 상관은 약 `0.876`으로 낮아 정보 다양성이 있다. 다만 로컬 2024 최적화는 실제 2025 리더보드와 일치하지 않았으므로, 로컬 최적 가중치를 사용하지 않고 실제 최고 모델 V3를 80%로 보존하는 보수적 고정 결합을 사용했다. 이 가중치는 리더보드 정답이나 점수 역산으로 결정하지 않았다.

## 규정 준수 및 실행 감사

- 공식 train으로 학습된 기존 artifact만 사용
- test 행 간 집계·빈도·rolling·expanding·분포 보정 없음
- 외부 데이터 없음
- 행 순서 변경/단일 행 추론 동일성 통과
- 245,789행 추론 약 `4.64초`
- peak RSS 약 `1.81 GiB`
- 기존 V3 및 939점 ZIP은 덮어쓰지 않고 보존

실제 리더보드 향상은 보장되지 않으므로 V3 원본을 기준선으로 유지해야 한다.
