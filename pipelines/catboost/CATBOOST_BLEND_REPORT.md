# 전체기간 + 최근3시즌 CatBoost OOF 앙상블

## 결과

- 전체기간 OOF raw BSS: 633.701
- 최근3시즌 OOF raw BSS: 609.966
- OOF blend raw BSS: 637.544
- OOF blend calibrated BSS: **664.891**
- 단일 전체기간 calibrated BSS: 661.814
- 개선: **+3.077 BSS**

가중치는 2024 hash=2 calibration 전용 구간의 Brier Score만으로 정했다.

- 전체기간: 0.605971
- 최근3시즌: 0.394029
- untouched 평가 예측 상관: 0.960247

hash=3 untouched 63,751행은 모델 early stopping, 가중치 적합, calibration에 사용하지 않았다.

## 규정 준수

- 공식 train만 사용
- test/리더보드 정보 미사용
- test 행 간 집계·빈도·rolling·expanding 없음
- test 분포 또는 예측 평균 보정 없음
- 추론 시 고정된 두 모델, 고정 가중치, 고정 calibration만 행별 적용
- 단일행/batch/shuffle 예측 차이 0

## 제출물

- `submit_catboost_blend.zip`
- 245,789행 모사 추론 약 0.43초
- peak RSS 약 0.99GB
