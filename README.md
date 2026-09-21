# LG Aimers 9기 Phase 2 — 투구 제구 확률 예측

> 팀 **ABS깡통존**(LG Aimers 9기 × LG 트윈스 해커톤, DACON 236743)의 팀원 저장소 **공개 스냅샷**입니다. 팀 최종 제출(네 모델의 확률 균등 블렌드, Public 1058.60)에 들어간 네 모델 중 두 개 — 계층 잔차 lookup(`pipelines/clean_forest`)과 Clean 2024 residual MoE(`pipelines/dsf_upgrade`) — 가 이 저장소에서 나왔습니다. 전체 맥락과 대표 저장소: [LG-Aimers-Pitch-Command-Hackathon-public](https://github.com/LGAimers-9th-Hufs/LG-Aimers-Pitch-Command-Hackathon-public). 무엇을 뺐는지는 [`PUBLISHING_NOTE.md`](PUBLISHING_NOTE.md).

대회에서 시도한 모델을 **파이프라인 단위**로 정리한 저장소입니다. 학습 데이터와 OOF 예측은 포함하지 않으며, 제출 ZIP·모델 가중치·선수별 lookup은 공개본에서 제외했고(`PUBLISHING_NOTE.md`), 각 파이프라인의 `train*.py` / `build_*submission.py`로 재생성합니다.

## 규정 준수 원칙

> **2026-08-29 재감사:** 팀원 원본 `DSF_CHAL061_FIX` champion에서 공개 리더보드 probe로 산출한 보정값이 발견됐다. 이 모델과 이를 포함한 파생 ZIP은 제출에 사용하지 않는다. 현재 규정 준수 신규 후보는 `clean_lookup`이다.

- test의 각 행을 독립적으로 예측합니다.
- test 다른 행을 이용한 집계·빈도·rolling·expanding·target encoding·분포 보정을 사용하지 않습니다.
- 현재 투구 이후 정보와 2025 TrackMan 데이터를 사용하지 않습니다.
- TrackMan 파이프라인은 주최 측이 제공한 2019~2024 이력만 사용합니다.
- calibration과 블렌딩 계수는 train의 시간순 OOF 또는 분리된 검증 구간에서만 적합합니다.

상세 감사 결과는 [`docs/AUDIT.md`](docs/AUDIT.md)를 참고하세요. 로컬의 공식 대회 안내 원문과 금지 규칙 가이드도 검토했지만 대회 원문 데이터는 GitHub에 배포하지 않습니다. 실제 제출 전 최신 원본 규칙을 다시 대조해야 합니다.

## 파이프라인 안내

| 파이프라인 | 모델/역할 | 특징 문서 |
| --- | --- | --- |
| LightGBM V2 | 경량 3모델 기준선 | [`pipelines/lightgbm_v2/PIPELINE.md`](pipelines/lightgbm_v2/PIPELINE.md) |
| LightGBM V3 | 시간순 TE 강화 및 Brier 보조 모델 | [`pipelines/lightgbm_v3/PIPELINE.md`](pipelines/lightgbm_v3/PIPELINE.md) |
| CatBoost | 전체기간·최근 3시즌·regime·안정 피처 후보 | [`pipelines/catboost/PIPELINE.md`](pipelines/catboost/PIPELINE.md) |
| Embedding MLP | CatBoost와 다른 오차 구조를 노린 신경망 | [`pipelines/mlp/PIPELINE.md`](pipelines/mlp/PIPELINE.md) |
| TrackMan | 공식 과거 물리 프로파일 후보 | [`pipelines/trackman/PIPELINE.md`](pipelines/trackman/PIPELINE.md) |
| Ensembles | OOF 기반 파이프라인 혼합 연구 | [`pipelines/ensembles/PIPELINE.md`](pipelines/ensembles/PIPELINE.md) |
| Clean Regime Stack | clean_moe + 공식 train 재학습 regime 전문가 | [`pipelines/regime_stack/PIPELINE.md`](pipelines/regime_stack/PIPELINE.md) |
| Clean Forest/Lookup | 대형 ExtraTrees 반증 실험 + 계층 잔차 lookup | [`pipelines/clean_forest/PIPELINE.md`](pipelines/clean_forest/PIPELINE.md) |
| DSF Upgrade | 팀원 DSF 제출본을 앵커로 한 실험군 — V3/regime 블렌드, F 전문가, FactorTabM+라우터, 계층 GAM, **Clean 2024 residual MoE**(최종 블렌드 레그); `COMPLIANCE_REAUDIT.md` | [`pipelines/dsf_upgrade/PIPELINE.md`](pipelines/dsf_upgrade/PIPELINE.md) |
| Clean DSF2 | 프로브 상수를 제거한 DSF + clean_lookup 블렌드(2024 OOF로만 가중) | [`pipelines/clean_dsf2/PIPELINE.md`](pipelines/clean_dsf2/PIPELINE.md) |
| Clean Residual CatBoost | clean_lookup 잔차를 CatBoost로 학습 | `pipelines/clean_rescat/` |
| DeepFM | field-aware DeepFM(PyTorch), 2023→2024 검증 후 기각 | [`pipelines/deepfm/README.md`](pipelines/deepfm/README.md) |
| TrackMan GAM | 선수 ID 없이 쓰는 spline GAM, 2023→2024 검증 후 기각 | [`pipelines/trackman_gam/README.md`](pipelines/trackman_gam/README.md) |

## 디렉터리 구조

```text
.
├── pipelines/
│   ├── lightgbm_v2/       # train, inference, report, requirements
│   ├── lightgbm_v3/       # V3/Brier 학습·추론·실험
│   ├── catboost/           # CatBoost 계열 후보와 실험
│   ├── mlp/                # PyTorch embedding MLP
│   ├── trackman/           # 공식 TrackMan 이력 피처 모델
│   ├── ensembles/          # 파이프라인 간 OOF 혼합
│   ├── regime_stack/       # clean_moe + regime 전문가 스택
│   ├── clean_forest/       # ExtraTrees 반증 + 계층 잔차 lookup  ← 최종 블렌드 레그
│   ├── dsf_upgrade/        # DSF 앵커 실험군, Clean 2024 residual MoE  ← 최종 블렌드 레그
│   ├── clean_dsf2/         # 프로브 제거 DSF + lookup 블렌드
│   ├── clean_rescat/       # 잔차 CatBoost
│   ├── deepfm/             # DeepFM (기각)
│   └── trackman_gam/       # spline GAM (기각)
├── shared/                 # 여러 파이프라인이 공유하는 행 단위 피처
├── docs/                   # 프로젝트 감사 문서
└── requirements.txt        # 공통 개발 환경 의존성
```

모든 명령은 저장소 루트에서 모듈 방식으로 실행합니다.

```bash
python -m pipelines.catboost.train --help
python -m pipelines.lightgbm_v3.experiments.evaluate_v3_brier --help
```

## 데이터 배치

```text
data/
├── train.csv
├── test.csv
├── sample_submission.csv
└── trackman_history.csv
```

모델, OOF 예측, 선수별 artifact, 제출 ZIP은 저장소에 없습니다(`.gitignore` + 공개본 제외). 팀 차원의 규정 준수 기록은 대표 저장소의 `phase3/COMPLIANCE_DISCLOSURE.md`를 참고하세요.
