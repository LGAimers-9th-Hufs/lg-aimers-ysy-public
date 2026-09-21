# Field-aware DeepFM

공식 train 행만 사용한 신규 구조 실험이다. 범주별 embedding, 명시적 교차 필드,
FM 2차 interaction, 작은 MLP를 Brier loss로 공동 학습한다.

## Walk-forward 결과

- 학습: 2023 공식 train 245,525행
- 검증: 2024 전체 253,507행
- 외부 데이터, test 집계, LB feedback 없음

| epoch | raw select BSS | raw evaluate BSS |
|---:|---:|---:|
| 1 | 461.066 | 472.334 |
| 2 | 242.312 | 228.219 |
| 4 | -838.826 | -966.791 |

1 epoch 예측을 select에서 affine calibration하면 evaluate BSS는 `575.801`이다.
하지만 `dsf_cal` 및 `dsf_cal + clean_lookup`에 대한 최적 추가 가중치는 모두 `0.0`이었다.
잔차 상관도 `0.998515`로 높아 최종 제출 후보에서 제외했다.

결론: 동일한 ID/count interaction을 더 복잡한 신경망으로 학습하는 것만으로는
2023→2024 regime drift를 이기지 못한다. 다음 신규 모델은 ID embedding보다 물리적
Trackman 요약과 공식 asof 변화량에 강한 단조·shape-constrained 구조가 필요하다.
