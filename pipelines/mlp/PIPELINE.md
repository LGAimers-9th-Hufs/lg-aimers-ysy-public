# PyTorch Embedding MLP 파이프라인

## 목적과 특징

CatBoost와 오차 상관이 낮은 후보를 만들기 위한 shared-trunk 신경망입니다. 범주형 변수는 embedding으로, 수치형 변수는 고정 전처리 후 입력하며 BCE와 Brier loss를 혼합합니다.

## 구성

- walk-forward OOF: `train_walkforward.py`
- 전체 학습: `train.py`
- 추론: `inference.py`
- 산출물: `mlp.pt`, `preprocess.json`, `artifacts.json`
- 의존성: `requirements.txt`

## 실행

```bash
python -m pipelines.mlp.train_walkforward --train_path ./data/train.csv
python -m pipelines.mlp.train --train_path ./data/train.csv --out_dir ./model_mlp
```

## 규정 준수와 한계

전처리기는 train에서만 적합되고 test에는 고정 변환만 적용됩니다. test 행 간 연산은 없습니다. 단독 성능보다 CatBoost/V3와의 residual diversity가 목적이며, 제출 시 PyTorch 용량과 245,789행 CPU 추론 시간을 반드시 실측해야 합니다.
