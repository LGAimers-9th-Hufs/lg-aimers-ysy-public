# CatBoost 파이프라인

## 목적과 특징

범주형 ID와 상황 변수를 CatBoost ordered boosting으로 직접 다루는 계열입니다. 전체기간 모델, 최근 3시즌 모델, 두 모델의 OOF 혼합, game-type regime interaction, fold-stable 피처 후보를 한곳에 모았습니다.

## 변형별 파일

| 변형 | 학습 | 추론 | 핵심 특징 |
| --- | --- | --- | --- |
| 전체기간 | `train.py` | `inference.py` | 전체 시즌 CatBoost 기준선 |
| 최근 3시즌 | `train_recent.py` | `inference.py` | 최근 체제에 더 큰 적합도 |
| 전체+최근 혼합 | 위 두 모델 | `inference_blend.py` | 시간순 OOF로 혼합 계수 결정 |
| shared regime | `train_regime_shared.py` | `inference_regime_shared.py` | game_type 분리 없이 interaction 강화 |
| stable features | `train_stable_features.py` | `inference_stable_features.py` | 여러 fold에서 안정적인 피처만 사용 |

평가·ablation은 `experiments/`, 결과는 같은 폴더의 세 보고서에 있습니다.

## 실행 예시

```bash
python -m pipelines.catboost.train --train_path ./data/train.csv --out_dir ./model_catboost
python -m pipelines.catboost.experiments.evaluate_catboost_blend --help
```

## 규정 준수와 한계

모든 추론 피처는 현재 test 행만으로 생성합니다. 전체/최근 혼합 가중치와 calibration은 train OOF에서만 결정합니다. 최근 시즌 모델은 분포 이동에 유리할 수 있지만 표본 감소로 분산이 커질 수 있으므로 단독 채택보다 untouched fold 성능과 residual correlation을 함께 봐야 합니다.
