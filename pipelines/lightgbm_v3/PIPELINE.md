# LightGBM V3 파이프라인

## 목적과 특징

V2를 기반으로 시간순 target encoding, Bayesian smoothing, 상황 상호작용을 강화한 중심 파이프라인입니다. 별도의 LightGBM Brier regressor를 보조 성분으로 학습하고 V3 확률과 OOF 기반으로 혼합하는 실험도 포함합니다.

## 구성

- 기본 V3 학습: `train.py`
- Brier 보조 학습: `train_brier.py`
- V3 중심 앙상블 추론: `inference_anchor.py`
- Brier 성분/혼합 추론: `inference_brier_component.py`, `inference_brier_blend.py`
- 실험: `experiments/`의 최근기간, 계층형 TE, 학습 창, Brier 평가
- 결과 문서: `MODEL_V3_REPORT.md`, `V3_ANCHOR_REPORT.md`, `V3_RECENCY_BRIER_REPORT.md`

## 실행 예시

```bash
python -m pipelines.lightgbm_v3.train --train_path ./data/train.csv --out_dir ./model_v3
python -m pipelines.lightgbm_v3.train_brier --train_path ./data/train.csv --out_dir ./model_v3_brier
```

## 규정 준수와 한계

TE와 calibration은 train의 과거 구간 또는 OOF에서만 적합하고 추론에서는 고정 lookup을 사용합니다. 앙상블 추론 파일은 제출 ZIP 안의 `helper_v3`, `helper_brier`, `helper_residual` 배치를 전제로 하므로 단독 파일만으로는 실행되지 않습니다.
