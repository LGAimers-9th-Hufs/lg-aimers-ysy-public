# Clean DSF2

LB probe를 제거한 팀 DSF와 `clean_lookup`을 공식 train walk-forward OOF로 혼합한다.

- `dsf_clean`: 원본 DSF의 `seg_probe`와 `regime_transfer_probe`를 삭제하고 추론 코드에서도 평가 피드백 보정을 영구적으로 0 처리
- 최종 가중치: dsf_clean `0.18`, clean_lookup `0.82`, shift `-0.002`
- 가중치 출처: 2024 OOF select only
- 2024 untouched: clean_lookup `859.662` → clean_dsf2 `872.645` (`+12.983 BSS`)
- F: `855.436` → `895.520`
- R: `806.358` → `815.784`

추가 신규 후보 결과:

- hashed sparse linear: 가중치 0 또는 untouched 하락
- Ordered CatBoost: 가중치 0
- crossed-effects: `+4.323 BSS`
- crossed-effects + lookup: `+14.261 BSS`였으나, 최종 재현성과 기존 팀 DSF 활용 가능성을 고려해 DSF-clean OOF blend를 우선 제출 후보로 선택

test의 각 행은 독립적으로 처리되며 test 집계·빈도·rolling·분포 보정은 사용하지 않는다.

## DSF Cal (`dsf_cal.zip`)

충돌하는 PyTorch 기반 `clean_lookup`을 포함하지 않고, probe 제거 DSF에 가장 최근
공식 학습 데이터의 2024 walk-forward OOF로 적합한 전역 affine calibration만 적용한다.

- 보정식: `clip(0.5 - 0.008228836300836177 + 0.8946876621596727 * (p - 0.5), 0, 1)`
- 계수 적합: 2024 공식 train walk-forward OOF 253,507행
- 사전 분리 select에서 적합 후 untouched evaluate 성능:
  `817.391 -> 857.221 BSS` (`+39.830`)
- 최종 제출 계수는 검증 완료 후 2024 OOF 전체로 재적합
- leaderboard 점수, 평가 라벨, test 통계·분포를 계수 산출에 사용하지 않음
- 245,789행 로컬 추론: 9.709초, 유효 확률 245,789개
- 단일 행 vs 전체 행 최대 차이: `0.0`
- 원순서 vs 역순 최대 차이: `0.0`
- ZIP SHA256: `9c8b74602204f7bf579b4082f33bec9e01b395888222db41460b3f9c42b46b8f`

`feature_engine.py`의 `groupby`는 공식 train으로 frozen artifact를 만드는 학습 함수에만
존재한다. 평가 서버의 실제 추론 경로는 `transform(test, artifacts)`이며 test에 대한
집계 함수를 호출하지 않는다.

## DSF Safe (`dsf_safe.zip`)

`dsf_cal` 보정 방향의 75%만 적용한 분포 이동 보수형 후보다.

- 6개 contiguous leave-block-out: 모든 구간 상승, 평균 `+30.276`, 최저 `+4.973 BSS`
- 5개 chronological forward validation: 모든 구간 상승, 평균 `+29.186`, 최저 `+2.998 BSS`
- full-strength affine는 평균 기대 상승이 더 크지만 한 초기 구간에서 `-0.450 BSS`
- 따라서 최고 기대값은 `dsf_cal`, 안정성 우선은 `dsf_safe`로 구분한다.

## Residual specialist 재검증

2024 DSF OOF 잔차에 작은 LightGBM을 학습하고 5개 chronological forward fold에서
검증했다. 최선 후보는 100 trees, cap `0.005`, strength `0.30`이었다.

- fold delta: `+4.983`, `+1.087`, `-2.871`, `-0.353`, `+0.661`
- 평균: `+0.701 BSS`
- 두 fold 하락으로 제출 후보 탈락

규정 준수 `clean_regime`과의 OOF 혼합도 확인했다.

- select/evaluate 단일 분할에서 25% 혼합: `+5.279 BSS`
- chronological forward 최선 평균: 20% 혼합 `+3.267 BSS`
- 최악 fold: `-13.256 BSS`
- 시간 안정성 부족으로 제출 후보 탈락

따라서 현재 검증상 `dsf_cal` 및 `dsf_safe`보다 신뢰할 만한 추가 residual/ensemble
후보는 없다. 리더보드 피드백을 이용해 혼합 비율을 조정하지 않는다.

## DSF 독립 모델 앙상블 탐색

2024 OOF의 공통 select/evaluate 행에서 모든 후보를 다시 정렬했다. 각 독립 후보는
select에서 affine calibration과 DSF 혼합 가중치를 정하고 evaluate에는 그대로 적용했다.

| 후보 | DSF 혼합 가중치 | untouched delta BSS | standalone BSS | residual corr |
|---|---:|---:|---:|---:|
| clean_lookup | 0.500 | +22.860 | 857.252 | 0.999539 |
| crossed combo | 0.500 | +19.906 | 853.425 | 0.999560 |
| sparse combo | 0.500 | +16.835 | 846.959 | 0.999557 |
| clean MoE | 0.500 | +13.753 | 821.276 | 0.999360 |
| clean regime | 0.318 | +4.533 | 802.388 | 0.999489 |
| clean temporal | 0.039 | +1.541 | 98.958 | 0.995836 |

FactorTabM seed 0/1/2, Trackman Factor, ordered CatBoost, hierarchical GAM, clean forest는
동일 선택 절차에서 가중치가 0에 가깝거나 untouched 성능이 하락했다.

상위 `clean_lookup` 50% 혼합은 6개 full-fit diagnostic time block에서도 모두 양수였다
(`+65.95` 이상). 정직한 select→evaluate 기준 예상 추가 이득은 `+22.86 BSS`를 사용한다.
단, 기존 clean_lookup 패키지는 Mac의 Torch/native ABI 충돌이 있었으므로 실제 제출 전
평가 서버 기본 Python 3.11, torch 2.7.1, scikit-learn 1.8.0과 일치하는 격리 실행 구조를
검증해야 한다.

## DSF Lookup (`dsf_lookup.zip`)

- 구성: `0.50 * dsf_cal + 0.50 * calibrated(clean_lookup)`
- clean lookup 최종 calibration: intercept `0.004027927737338711`, slope `1.1120487007912476`
- calibration은 공식 2024 OOF 전체에 최종 재적합
- 혼합 가중치 0.50은 2024 select에서 선택, untouched evaluate delta `+22.860 BSS`
- 두 member는 별도 Python subprocess로 실행
- requirements는 평가 서버 기본 torch/sklearn을 사용하고 `lightgbm==4.6.0`만 설치
- LB probe, test 분포, test 행 간 집계는 사용하지 않음
