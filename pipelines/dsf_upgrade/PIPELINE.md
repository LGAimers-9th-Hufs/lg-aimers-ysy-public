# DSF 중심 업그레이드 파이프라인

> **규정 재감사 경고 (2026-08-28):** 원본 DSF champion의 `metadata.json`에 공개 리더보드 양방향 probe로 산출한 `seg_probe t=0.00570`과 점수 역산으로 구한 regime 혼합 가중치가 명시돼 있다. 익명 train 투수 ID와 공식 TrackMan ID의 교집합이 0인데도 투수별 TrackMan 보정 맵의 생성 provenance가 남아 있지 않다. 따라서 아래 DSF 기반 실험은 역사적 결과로만 보존하며 제출 후보로 사용하지 않는다.

팀원 DSF 제출물을 고정 anchor로 두고 우리 모델의 시간순 OOF가 추가 정보를 제공하는지 검증한다.

## 원칙

- 모든 비교는 동일 `row_id`의 train OOF에서 수행한다.
- test나 리더보드 점수로 가중치·보정값을 선택하지 않는다.
- 최종 추론은 각 test 행의 기반 모델 확률과 고정 가중치만 사용한다.
- 제공된 `champion_reconstructed`는 원본 학습 OOF가 아니라 기존 혼합에서 복원된 참고 성분임을 명시한다.

## 실험 결과

`evaluate_v3_blend.py`로 2022~2024 walk-forward V3 OOF를 새로 생성하고 DSF OOF와 `row_id` 기준으로 정렬했다. Brier Skill Score는 다음과 같았다.

| 검증 시즌 | DSF 최종 | V3 | DSF 80% + V3 20% |
|---|---:|---:|---:|
| 2023 | -1336.304 | -1030.137 | -1266.306 |
| 2024 (untouched) | 897.155 | 695.656 | **905.971** |

2022/2023 결과를 먼저 보고 둥근 고정 후보 `V3 20%`를 선택했다. 2024는 선택에 사용하지 않은 최종 확인 구간이다. 2024만 보고 얻을 수 있는 17%는 사용하지 않았다. DSF와 V3의 residual correlation이 약 0.998~0.999로 높아 큰 앙상블 이득은 기대하기 어렵지만, 2024 OOF에서는 DSF 단독보다 개선됐다.

최종 제출 후보는 기존 DSF 제출 80%와 V3 20%를 행별 고정 혼합한다. test 전체의 빈도·통계·분포를 계산하지 않으며, 5개 샘플을 전체/한 행씩 추론한 결과 최대 절대차는 `0.0`이었다.

## 재현

```bash
python -m pipelines.dsf_upgrade.evaluate_v3_blend \
  --train_path ./data/train.csv \
  --dsf_oof "/path/to/meta_blend_wf_2022_2024.npz"
```

```bash
python -m pipelines.dsf_upgrade.build_submission \
  --dsf_dir ./DSF_CHAL061_FIX \
  --v3_dir ./submit_v3 \
  --v3_weight 0.20 \
  --output ./submit_dsf_v3_w20.zip
```

`DSF_CHAL061_FIX/`와 `submit_v3/`는 모델 자산이며 저장소에 커밋하지 않는다. 생성된 ZIP에도 학습 데이터나 OOF 파일은 포함하지 않는다.

## DSF regime blend

V3 혼합의 리더보드 하락 이후 DSF 내부 Champion/Challenger만 다시 평가했다. 일반 선형 residual correction은 2023과 2024 walk-forward에서 모두 악화되어 폐기했다.

정규시즌 `R`에서는 Challenger 비중을 높이는 방향이 2022와 2023에서 모두 개선됐다. 두 구간의 최적값은 각각 약 39%, 29%였으므로 더 보수적인 25%를 사전 선택했다. `F`는 원본 6.1%를 유지했다. 선택에 사용하지 않은 2024에서도 원본 DSF 대비 `+5.035 BSS`였다.

| 시즌 | 원본 DSF | regime blend | 차이 |
|---|---:|---:|---:|
| 2022 | 850.450 | 881.344 | +30.893 |
| 2023 | -1336.304 | -1331.954 | +4.349 |
| 2024 (untouched) | 897.155 | 902.750 | +5.595 |

```bash
python -m pipelines.dsf_upgrade.build_regime_submission \
  --dsf_dir ./DSF_CHAL061_FIX \
  --output ./submit_dsf_regime_w25.zip
```

추론 시 `game_type`은 현재 행의 공식 입력값만 읽는다. 전체 test 집계는 없으며, 전체 배치와 한 행씩 실행한 예측의 최대 절대차는 `0.0`이다.

## 최근 F calibration 후보

Regime blend가 리더보드에서 원본보다 낮아진 뒤에는 원본 DSF 가중치를 완전히 복원했다. 2024의 `F` 행 30,010개를 `row_id` 안정 해시 4-fold로 나눠 절편 보정을 교차검증했다. 복잡한 행 피처 residual 모델은 악화되어 제외했고, 단순 보정의 75% shrink만 채택했다.

- 원본 가중치: F/R 모두 Challenger 6.1%
- F에만 고정 확률 이동: `-0.0051953538`
- R: 원본 예측 그대로
- 2024 F cross-fit Brier 감소: `0.0000356243`
- 2024 전체 환산 예상 BSS 증가: 약 `+1.69`

```bash
python -m pipelines.dsf_upgrade.build_regime_submission \
  --dsf_dir ./DSF_CHAL061_FIX \
  --r_weight 0.061 \
  --f_shift -0.005195353779245401 \
  --output ./submit_dsf_fcal_s75.zip
```

이 후보의 기대 상승은 작다. 리더보드 점수로 shift를 선택하지 않았으며, 2024 공식 train OOF 교차검증만 사용했다.

## 논문 기반 recent specialist와 R Beta calibration

CatBoost ordered boosting, temporal model selection, diversity-aware ensemble, Beta calibration 아이디어를 순서대로 검증했다.

- 최근 F CatBoost: 2023 F 학습 → 2024 F 검증. DSF와 residual correlation `0.9980`; raw 혼합은 악화했고 Beta 이후 F에서만 `+10.3 BSS`. 전체 기여가 작아 모델을 제출 패키지에 포함하지 않았다.
- 최근 R CatBoost: 2023 R 학습 → 2024 R 검증. 선택 가중치 13%에서 untouched R `+7.6 BSS`였지만 residual correlation `0.9985`로 높았다.
- DSF 단독 R Beta calibration: specialist 없이 더 큰 개선. 2024 R 4-fold cross-fit에서 `+36.156 BSS`; 각 fold 개선은 `+46.302`, `+0.601`, `+52.975`, `+44.567`이었다.
- 시간순 확인: 2024년 3~6월로 calibration을 fit하고 7~9월에 적용했을 때 `+55.676 BSS`.

최종 후보는 원본 DSF 가중치를 그대로 유지하고 R 행에만 다음 Beta mapping을 적용한다.

`sigmoid(0.901548·log(p) - 0.879944·log(1-p) - 0.016988)`

F는 원본 DSF 그대로다. 2024 전체 환산 cross-fit 기대 상승은 약 `+31.9 BSS`이며, test에서는 현재 행의 `game_type`과 원본 확률만 사용한다.

```bash
python -m pipelines.dsf_upgrade.build_regime_submission \
  --dsf_dir ./DSF_CHAL061_FIX \
  --r_weight 0.061 \
  --r_beta 0.9015482577778559 0.8799440292267364 -0.01698789277630549 \
  --output ./submit_dsf_r_beta.zip
```

### 리더보드 결과

- 원본 DSF: 약 `1062`
- R Beta calibration 후보: `1035`
- 변화: `-27`

2024 R 4-fold 및 월 순서 검증의 개선이 2025 평가 분포로 일반화되지 않았다. 이 후보는 **폐기**하며 재제출하지 않는다. 이 결과를 이용해 다른 calibration 계수를 맞추는 리더보드 튜닝도 하지 않는다. 현재 최고 기준 모델은 다시 원본 DSF다.

## 통합 current-season + hierarchy + TrackMan Brier residual

Champion 재구축을 제외한 신규 피처 방향을 하나의 LightGBM Brier residual 회귀기로 검증했다.

- 이전 시즌 endpoint에서 현재 시즌 투구 수와 success/reverse/middle/ball/strike rate 분해
- 타자 현재 시즌 success·middle rate 및 투수 pitch-mix 분해
- train 과거 시즌만 사용하는 계층형 pitcher/batter/team/context shrinkage
- 공식 2019~2024 TrackMan의 시점 안전 hand×pitch-group 기대 물리 피처
- 목표값 `y - DSF_OOF`, 최대 보정 폭과 shrink를 사전 grid로 제한

Walk-forward 결과에서 통과 후보가 없었다. 가장 보수적인 `cap=0.003, shrink=0.1`도 다음과 같았다.

| 구간 | DSF 대비 BSS 변화 |
|---|---:|
| 2023 R | +0.067 |
| 2023 F | -52.977 |
| 2024 R | +0.042 |
| 2024 F | -0.833 |

R의 개선은 실질적으로 0에 가까웠고 F를 악화시켰다. 따라서 최종 모델과 제출 ZIP은 생성하지 않았다. 관련 코드는 `novel_features.py`와 `train_integrated_residual.py`에 재현 가능하게 보존한다.

## Factorization TabM + conditional router

기존 embedding MLP와 다른 오차를 만들기 위해 8개 parameter-efficient head와 명시적 ID 상호작용을 결합한 FactorTabM을 검증했다. 사용한 상호작용은 pitcher×batter, pitcher×count, pitcher×game type, pitcher×batter hand, team×team, batter×pitcher hand다. 모든 범주 사전과 수치 전처리 통계는 각 walk-forward fold의 과거 시즌에서만 적합했다.

| 시즌 | DSF BSS | FactorTabM BSS | 단순 혼합 BSS | 선택 가중치 | 잔차 상관 |
|---|---:|---:|---:|---:|---:|
| 2023 eval hash | -1254.184 | 50.597 | -766.117 | 20% | 0.99299 |
| 2024 eval hash | 817.391 | 428.193 | 818.376 | 12% | 0.99774 |

기존 MLP보다 잔차 상관은 낮아졌지만, 미래 대용 2024에서 단순 혼합 개선은 `+0.985 BSS`에 불과했다. 이어서 2023 stop hash에서 FactorTabM이 DSF보다 좋은 행을 분류하는 얕은 LightGBM router를 학습하고, 2023 select hash에서 threshold와 correction strength를 선택했다. 2023 eval hash에서 `+623.370 BSS`였지만 완전히 분리한 2024 eval hash에서는 `+1.378 BSS`만 개선됐다.

사전 통과 기준인 2024 `+5 BSS`를 만족하지 못했고 2023 특이 분포에 의존하는 신호로 판단했다. 최종 학습, 제출 ZIP, 리더보드 제출은 하지 않는다. `factor_tabm.py`, `evaluate_factor_tabm.py`, `evaluate_factor_router.py`로 결과를 재현할 수 있다.

## DSF failure-segment stability audit

2023 급락을 분해하면 정규시즌 R은 `684 BSS`였고 F만 `-19,012 BSS`였다. F 평균 오차의 방향도 2022와 2023 사이에서 반전되어 F 전용 보정은 시간적으로 안정적이지 않았다. 선수 OOV, 누적 표본량, 카운트, 월, game type을 조사했으며 OOV는 각 검증 시즌보다 과거인 공식 train ID만으로 정의했다.

2022와 2023에서 동시에 Champion/Challenger 가중치 증가가 유리했던 R 카운트 중 2024에서도 개선된 `0-0`, `0-1`, `1-2`, `2-1`, `3-1`만 남겼다. 해당 구간 가중치 router의 전체 결과는 다음과 같다.

| 시즌 | 원본 DSF 대비 BSS 변화 |
|---|---:|
| 2022 | +13.443 |
| 2023 | +2.986 |
| 2024 untouched | +2.646 |

2024 월별로는 6월 `-2.138`, 10월 `-8.136 BSS`로 방향이 깨졌다. 사전 기준인 2024 전체 `+5 BSS` 및 모든 월 비악화를 충족하지 못해 제출 ZIP을 만들지 않았다. `analyze_failure_segments.py`가 이 결과와 거부 판정을 재현한다.

## Hierarchical logistic GAM architecture probe

트리 모델과 다른 구조를 만들기 위해 투수·타자·팀·카운트 효과와 pitcher×count, batter×hand, team×team×count 상호작용을 L2 정규화 희소 로지스틱 모델로 학습했다. 누적 능력·최근 폼·경기 상황에는 선형, 제곱, tanh 항을 함께 사용했다. 범주 사전과 수치 통계는 각 fold의 과거 train에서만 적합하며 추론은 행별로 독립적이다.

초기 자동 학습률은 확률 포화로 폐기했다. 고정 `eta0=0.0002`, `alpha=0.00005`에서 단독 GAM 결과는 다음과 같았다.

| 시즌 | DSF BSS | GAM BSS | 선택된 GAM 가중치 |
|---|---:|---:|---:|
| 2023 eval hash | -1254.184 | -626.071 | 30% (grid 상한) |
| 2024 eval hash | 817.391 | -86.514 | 0% |

성능 후보는 아니지만 실제 리더보드 구조 차이를 확인하려는 요청에 따라 DSF 95% + GAM 5% 실험 ZIP을 만들었다. 이 고정 5%의 eval 변화는 2023 `+51.090 BSS`, 2024 `-4.611 BSS`이므로 원본보다 낮아질 가능성이 높다. 가중치는 리더보드 결과로 선택하지 않았다. 로컬 5행 test에서 패키지 실행과 행별/배치별 GAM 예측 최대 차이 `0.0`을 확인했다. 실제 245,789행 test는 로컬에 없어 전체 규모 시간·메모리 실측은 하지 못했다.

```bash
python -m pipelines.dsf_upgrade.train_hierarchical_gam_final \
  --train_path ./data/train.csv --output_dir ./hierarchical_gam_final
python -m pipelines.dsf_upgrade.build_hierarchical_gam_submission \
  --dsf_dir ./DSF_CHAL061_FIX --gam_dir ./hierarchical_gam_final \
  --gam_weight 0.05 --output ./submit_dsf_hierarchical_gam_w05.zip
```

## Final Factorization TabM conditional-router package

OOF 검증을 마친 FactorTabM router를 실제 제출 패키지로 만들었다. FactorTabM은 전체 2019~2024 공식 train 1,475,092행으로 2 epoch 재학습했다. PyTorch와 LightGBM을 같은 학습 프로세스에서 초기화할 때 macOS 네이티브 런타임 충돌이 발생해 TabM과 router 학습 스크립트를 분리했으며 데이터 범위나 모델 설정은 변경하지 않았다.

- FactorTabM: 8개 parameter-efficient heads와 6개 명시적 factor interaction
- router fit: 2023 walk-forward OOF의 stop hash만 사용
- 선택: 2023 select hash에서 threshold `0.50`, correction strength `0.30`
- 추론: router score가 threshold 이상인 행만 `DSF + 0.30 × (FactorTabM - DSF)`
- 2024 untouched eval: 원본 DSF 대비 `+1.378 BSS`, routed fraction `31.84%`

로컬 5행 제출 환경에서 전체 패키지 실행을 확인했다. FactorTabM의 배치/단일 행 최대 절대차는 `2.98e-08`이며, 이는 부동소수점 batch 연산 순서 차이다. 5행을 단순 반복한 245,789행 합성 부하에서 FactorTabM 경로는 CPU 약 `0.83초`였고 모든 출력이 유한했다. 실제 test 행 간 집계·빈도·rolling·target encoding·분포 보정은 없다.

```bash
python -m pipelines.dsf_upgrade.train_factor_tabm_final \
  --train_path ./data/train.csv --output_dir ./factor_tabm_final \
  --epochs 2 --min_season 2019
python -m pipelines.dsf_upgrade.train_factor_router_final \
  --factor_oof_2023 ./factor_tabm_oof/factor_2023.npz \
  --output_dir ./factor_tabm_final
python -m pipelines.dsf_upgrade.build_factor_tabm_router_submission \
  --dsf_dir ./DSF_CHAL061_FIX --factor_dir ./factor_tabm_final \
  --output ./submit_dsf_factor_tabm_router.zip
```

### 3-seed FactorTabM ensemble router

단일-seed 패키지가 리더보드 `1175`를 기록한 뒤 모델 구조와 router 선택 원칙은 고정하고 seed 분산만 줄였다. seed offset `0`, `1000`, `2000`으로 2023·2024 walk-forward OOF를 생성하고 FactorTabM 확률 평균으로 router를 다시 학습했다. 리더보드 점수는 seed, threshold, alpha 선택에 사용하지 않았다.

| 후보 | 2024 untouched DSF 대비 BSS | routed fraction |
|---|---:|---:|
| 단일 seed router | +1.378 | 31.84% |
| 3-seed 평균 단순 혼합 | +8.511 | 해당 없음 |
| 3-seed 평균 router | +11.446 | 33.78% |

3-seed router도 2023 select hash에서 threshold `0.50`, correction strength `0.30`을 선택했다. 최종 세 member는 각각 전체 2019~2024 공식 train 1,475,092행, 2 epoch로 학습했다. 로컬 패키지 통합 실행에 성공했으며 245,789행 합성 FactorTabM 경로는 CPU 약 `1.40초`, 배치/단일 행 최대 차이는 `2.98e-08`이었다.

리더보드 결과는 단일-seed router `1075`, 3-seed 평균 router `1072`로 ensemble이 3점 낮았다. 따라서 3-seed 평균 제출 모델은 폐기하고 단일-seed 1075 모델을 champion으로 유지한다.

후속 seed-disagreement gate는 단일 seed 보정을 기본으로 하고 세 seed의 표준편차가 큰 행을 DSF로 되돌렸다. 기존 router는 2023 stop OOF로 고정하고 gate cutoff만 2024 select hash에서 선택했다. 2024 eval에서 hard gate는 DSF 대비 `+3.649 BSS`로 단일 router의 `+1.378`보다 나았지만 통과 기준 `+5`에 미달했다. soft gate는 `+2.726 BSS`였다. 새 ZIP은 생성하지 않았다.

## Paper-inspired TrackMan command FactorTabM

투구 위치 연구에서 제시한 릴리스 위치·구속·회전축의 결합과 투수별 민감도, 릴리스 포인트 confidence ellipse, active-spin/movement 관계를 공식 TrackMan에 적용했다.

- Kusafuka et al., *Influence of Release Parameters on Pitch Location in Skilled Baseball Pitching*: https://pmc.ncbi.nlm.nih.gov/articles/PMC7739723/
- *Relationship between ball release point variability and pitching performance*: https://pmc.ncbi.nlm.nih.gov/articles/PMC11608975/
- Young & Nathan, *Studies of the Active Spin Ratio*: https://baseball.physics.illinois.edu/ActiveSpin-v3.pdf

train의 익명 `pitcher_id`와 `pitcher_trackman_id`는 직접 연결할 수 없으므로 억지 매칭하지 않았다. TrackMan 내부에서 투수-시즌-구종별 release/movement covariance, 95% ellipse, 속도·회전·extension 변동성, active-spin proxy, 시즌 간 mechanics drift를 먼저 계산하고, 예측 시즌보다 과거인 `pitcher hand × pitch group` prior로 집계했다. test에서는 현재 행의 손 유형과 공식 as-of 구종 비율로 frozen prior를 결합한다.

동일 seed FactorTabM 비교에서 TrackMan command 피처를 추가한 DSF 20% 혼합은 2024 eval에서 `+23.549 BSS`였다. TrackMan 전용 router는 `+1.921`로 오히려 좋은 보정을 차단해 제외했다. 기존 1075 router와 TrackMan 후보를 2023·2024 select의 시즌 동일 가중 normalized Brier로 결합했으며 TrackMan 후보 비중 15%가 선택됐다.

| 시즌 | 최종 후보의 DSF 대비 BSS |
|---|---:|
| 2023 eval | +610.278 |
| 2024 eval | +7.374 |

최종식은 `0.85 × 기존 FactorTabM router + 0.15 × (0.80 × DSF + 0.20 × TrackMan FactorTabM)`이다. `tm_factor.zip`은 확장자 포함 13자이며, 245,789행 합성 두 신경망 경로는 CPU 약 1.65초였다. 실제 test 행 간 집계·rolling·분포 보정은 없다.

```bash
python -m pipelines.dsf_upgrade.train_factor_tabm_final \
  --train_path ./data/train.csv --trackman_path ./data/trackman_history.csv \
  --output_dir ./factor_tabm_trackman_final --epochs 2 --min_season 2019
python -m pipelines.dsf_upgrade.build_factor_tabm_trackman_submission \
  --dsf_dir ./DSF_CHAL061_FIX --base_dir ./factor_tabm_final \
  --track_dir ./factor_tabm_trackman_final --output ./tm_factor.zip
```

```bash
python -m pipelines.dsf_upgrade.train_factor_ensemble_router \
  --oof_dirs ./factor_tabm_oof ./factor_tabm_oof_seed1 ./factor_tabm_oof_seed2 \
  --output_dir ./factor_tabm_ensemble_final
python -m pipelines.dsf_upgrade.build_factor_tabm_ensemble_submission \
  --dsf_dir ./DSF_CHAL061_FIX \
  --factor_dirs ./factor_tabm_final ./factor_tabm_final_seed1 ./factor_tabm_final_seed2 \
  --router_dir ./factor_tabm_ensemble_final \
  --output ./submit_dsf_factor_tabm_3seed_router.zip
```

## Seed1 router + TrackMan safety gate

3-seed 평균이 리더보드에서 단일 seed보다 낮았으므로 seed 평균 대신 각 seed에 전용 router를 다시 적합했다. 모든 router는 2023 walk-forward OOF stop hash에서만 학습하고 2023 select hash에서 threshold `0.50`, alpha `0.30`을 선택했다.

| 후보 | 2023 eval DSF 대비 BSS | 2024 eval DSF 대비 BSS |
|---|---:|---:|
| seed0 router | +623.370 | +1.378 |
| seed1 router | +528.924 | +5.474 |
| seed2 router | +232.874 | +5.165 |

seed1을 새 기준으로 선택한 뒤 TrackMan FactorTabM을 직접 섞지 않고 안전 신호로만 사용했다. seed1 보정과 TrackMan 보정의 방향이 반대이고, TrackMan과 DSF의 차이가 train OOF에서 고정한 `0.0119562745` 이상인 행만 seed1 보정을 취소한다. test 전체 통계나 분위수는 계산하지 않는다.

| 후보 | 2023 eval DSF 대비 BSS | 2024 eval DSF 대비 BSS |
|---|---:|---:|
| seed1 router | +528.924 | +5.474 |
| seed1 + TrackMan safety gate | +536.162 | +10.673 |

추가로 2023 OOF stop residual만 학습하는 깊이 2의 LightGBM regressor를 적합했다. 입력은 현재 행의 DSF·seed1·TrackMan 예측과 고정 router/gate 신호뿐이며, alpha `0.30`, correction cap `0.010`은 2023·2024 OOF select의 시즌 동일 가중 normalized Brier로 선택했다.

| 후보 | 2023 eval DSF 대비 BSS | 2024 eval DSF 대비 BSS |
|---|---:|---:|
| safety gate | +536.162 | +10.673 |
| safety gate + residual | +589.000 | +11.998 |

두 ZIP 모두 확장자를 포함해 20자 이하이다. `seed1_tm_res.zip`은 5행/단일 행 최대 차이 `8.94e-09`, 행 순서 반전 차이 `0.0`을 기록했다. 245,789행 합성 전체 파이프라인은 로컬 CPU에서 약 12초였고 출력은 모두 유한하며 `[0, 1]` 범위였다. 추론 중 test 행 간 집계·빈도·rolling·expanding·target encoding·분포 보정은 없다.

```bash
python -m pipelines.dsf_upgrade.train_factor_router_final \
  --factor_oof_2023 ./factor_tabm_oof_seed1/factor_2023.npz \
  --output_dir ./factor_tabm_final_seed1
python -m pipelines.dsf_upgrade.train_factor_residual_final \
  --factor_oof ./factor_tabm_oof_seed1 \
  --track_oof ./factor_tabm_trackman_command_oof \
  --output_dir ./factor_tabm_final_seed1
python -m pipelines.dsf_upgrade.build_factor_trackman_safety_submission \
  --dsf_dir ./DSF_CHAL061_FIX --factor_dir ./factor_tabm_final_seed1 \
  --track_dir ./factor_tabm_trackman_final --output ./seed1_tm_gate.zip
python -m pipelines.dsf_upgrade.build_factor_trackman_residual_submission \
  --dsf_dir ./DSF_CHAL061_FIX --factor_dir ./factor_tabm_final_seed1 \
  --track_dir ./factor_tabm_trackman_final --output ./seed1_tm_res.zip
```

## Clean 2024 residual MoE

재감사에서 DSF champion의 `seg_probe`와 `regime_transfer_probe`가 공개 리더보드 피드백으로 산출됐음을 확인했다. 또한 train `pitcher_id` 792개와 공식 TrackMan `pitcher_trackman_id` 906개의 교집합은 0인데, champion의 투수별 TrackMan 보정 맵을 생성한 원본 코드와 provenance가 남아 있지 않았다. 원 요청의 “규정상 애매한 방법은 구현하지 않는다” 원칙에 따라 DSF champion과 모든 파생 보정을 clean 후보에서 제외했다.

clean 후보는 다음 성분만 사용한다.

- 학습 코드와 2022~2024 walk-forward OOF가 남아 있는 Independent Challenger
- 시즌 이전 데이터만 학습한 seed0·1·2 FactorTabM OOF
- 공식 TrackMan 과거 시즌 prior만 사용하는 TrackMan command FactorTabM OOF
- 현재 행의 모델 예측과 공식 입력 컬럼만 받는 깊이 4 LightGBM residual MoE

MoE는 2024 stop hash에서 학습하고 select hash에서 `d4`, alpha `1.0`, cap `0.04`를 선택한 뒤 evaluate hash를 한 번 확인했다. 배포 모델은 구조를 고정한 후 2024 strict OOF 전체로 재적합했다.

| 시즌 | Challenger BSS | clean MoE BSS | 차이 |
|---|---:|---:|---:|
| 2023 eval | -1557.889 | -525.479 | +1032.410 |
| 2024 untouched eval | 685.747 | 822.466 | +136.719 |

2024의 game type F/R 및 투수 이력 low/mid/high 구간이 모두 개선됐다. 제출 패키지는 단일 행/배치와 행 순서 변경에서 최대 차이 `0.0`, 245,789행 합성 추론 약 `7.02초`, 유한한 `[0, 1]` 확률을 확인했다. 최종 ZIP에는 추론에 쓰이지 않는 학습 helper도 넣지 않았으며 `groupby/rolling/expanding/cumsum/cumcount/shift` 정적 검색 결과는 0건이다.

```bash
python -m pipelines.dsf_upgrade.evaluate_clean_2024_moe \
  --train_path ./data/train.csv \
  --challenger_oof /path/to/challenger_wf_2022_2024.npz
python -m pipelines.dsf_upgrade.train_clean_moe_final \
  --train_path ./data/train.csv \
  --challenger_oof /path/to/challenger_wf_2022_2024.npz \
  --output_dir ./clean_moe_final
python -m pipelines.dsf_upgrade.build_clean_moe_submission \
  --challenger_dir ./DSF_CHAL061_FIX/challenger \
  --factor_dirs ./factor_tabm_final ./factor_tabm_final_seed1 ./factor_tabm_final_seed2 \
  --track_dir ./factor_tabm_trackman_final --moe_dir ./clean_moe_final \
  --output ./clean_moe.zip
```
