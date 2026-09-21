# Model V2 학습·검증 보고서

## 결론

감사에서 확인된 제출 차단 문제를 수정하고, 대회 규정을 지키는 LightGBM 3모델 앙상블을 재학습했다.

- test 행 간 집계·빈도·rolling·expanding·target encoding·분포 보정 없음
- 외부 데이터 미사용
- 학습과 추론이 동일한 `script.build_features()` 사용
- Target Encoding은 학습행 시즌보다 이전 시즌의 train label만 사용
- 2025 추론 map은 공식 2019~2024 train에서 사전 계산 후 행별 lookup만 수행
- calibration fit 행과 최종 validation 평가 행 분리
- 기존 모델은 `model_legacy_20260819/`에 보존

## 모델

| 모델 | Best iteration | 2024 전체 raw BSS |
|---|---:|---:|
| LightGBM 0 | 253 | 698.709 |
| LightGBM 1 | 333 | 697.624 |
| LightGBM 2 | 178 | 610.131 |

세 모델은 고정 동일가중으로 결합했다. validation label에 맞춰 blend weight를 과최적화하지 않았다.

## Calibration 독립 평가

2024 validation 253,507행을 안정적인 `row_id` hash parity로 나눴다.

- calibration fit: 126,645행
- calibration 미사용 평가: 126,862행
- 평가 절반 raw ensemble BSS: **699.899**
- 평가 절반 calibrated BSS: **743.545**
- logit calibration: `a=1.0944615300`, `b=-0.0534798070`

기존 artifact의 769.704는 calibration을 학습한 행에서 다시 측정한 값이었다. V2의 743.545는 calibration에 사용하지 않은 별도 행에서 측정했다.

## 피처와 artifact

- 총 피처: 79개
- 모델 파일: `lgb0.txt`, `lgb1.txt`, `lgb2.txt`
- artifact: `artifacts.json`
- 모델+artifact 크기: 약 6.2 MB
- 제출 zip: 약 2.7 MB
- artifact에 train SHA-256, 행 수, 패키지 버전, seed, validation 결과 저장

Trackman 전체기간 통계를 과거 validation 행에 적용하던 시점 오염을 피하기 위해 V2에서는 Trackman 파생 통계를 제외했다. 공식 row-level `asof_*` 피처와 동일 행의 상황 피처만 사용했다.

## 독립성·성능 검사

- 동일 행 단독 예측 vs 5행 batch 예측: `atol=1e-12` 통과
- 원본 순서 vs shuffle 후 ID 복원: `atol=1e-12` 통과
- 245,789행 모의 batch: 1.602초 (현재 로컬 머신)
- 245,789행 예측: 전부 finite, `[0, 1]` 범위 통과
- 제출 staging에서 `script.py` 직접 실행: 통과
- 출력: 5행 × 2컬럼, ID/NaN/확률 범위 검사 통과

## 현재 제출 구조

```text
submit/
├── model/
│   ├── artifacts.json
│   ├── lgb0.txt
│   ├── lgb1.txt
│   └── lgb2.txt
├── requirements.txt
└── script.py
```

생성된 제출 파일: `submit_v2.zip`

## 재학습

```bash
pip install -r requirements_train.txt
python train.py --train_path ./data/train.csv --out_dir ./model_v2
```

평가 서버의 추론 requirements는 `lightgbm==4.6.0`만 요구한다. pandas와 numpy는 대회 서버 기본 패키지를 사용한다.
