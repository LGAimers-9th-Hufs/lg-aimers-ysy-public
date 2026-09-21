# CatBoost 신규 파이프라인 보고서

## 결과

- 신규 모델: CatBoost Ordered Boosting
- 검증: 2024 row_id hash mod 4 엄격 분리
- early stopping: 126,338행
- calibration fit: 63,418행
- untouched final evaluation: 63,751행
- best iteration: 193
- untouched raw BSS: 633.701
- untouched calibrated BSS: **661.814**
- 동일 분할 V2 기준선: 628.853
- 기준선 대비: **+32.961 BSS**

## 파이프라인

LightGBM TE lookup 파이프라인과 독립적으로 작성했다. CatBoost ordered boosting이 선수·타자·팀·경기 상태 범주를 학습하며, 범주 통계는 공식 train 학습 중에만 만들어져 저장 모델에 고정된다.

추론은 test 각 행의 값과 고정 `catboost.cbm`만 사용한다. test의 다른 행을 이용한 groupby, 빈도, rolling, expanding, target encoding, 결측 통계, 예측 분포 보정은 없다. 외부 데이터도 없다.

## 실행 검증

- train/inference feature parity: 66개 피처 완전 동일
- 단일행 대 batch: 차이 0
- shuffle 대 원본: 차이 0
- 245,789행 모사: 약 0.40초
- peak RSS: 약 0.99GB
- 모델 크기: 약 836KB

## 제출

`submit_catboost.zip`을 제출한다. 기존 V2/V3 파일은 보존되어 있다.
