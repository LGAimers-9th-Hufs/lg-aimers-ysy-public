# LightGBM V2 파이프라인

## 목적과 특징

경량 LightGBM 3개를 혼합하는 안정적인 기준선입니다. `asof_*` 이력, 상황 피처, train에서만 만든 시간순 통계와 확률 보정을 사용하며, 서버에서 빠르고 작은 제출물을 만드는 데 초점을 둡니다.

## 구성

- 학습: `train.py`
- 추론: `inference.py`
- 상세 결과: `REPORT.md`
- 의존성: `requirements.txt`
- 산출물: `lgb0.txt`, `lgb1.txt`, `lgb2.txt`, `artifacts.json`

## 실행

```bash
python -m pipelines.lightgbm_v2.train --train_path ./data/train.csv --out_dir ./model_v2
```

제출 패키지에서는 `inference.py`를 `script.py`로 두고 산출물을 `model/` 아래 배치합니다.

## 규정 준수와 한계

추론은 test 현재 행과 고정 artifact만 사용합니다. test 내부 집계나 분포 보정은 없습니다. 다만 과거 감사에서 validation 분리와 학습/추론 TE 표현 차이가 지적되었으므로 점수 비교 시 `REPORT.md`와 `docs/AUDIT.md`를 함께 확인해야 합니다.
