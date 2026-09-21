# TrackMan 물리 프로파일 파이프라인 보고서

작성일: 2026-08-26

## 결론

공식 `trackman_history.csv`만 사용한 물리 프로파일 CatBoost를 기존 3모델에 추가했다. strict 2024 untouched BSS가 `680.763`에서 **`694.257`**로 `+13.494` 개선되어 `submit_trackman_ensemble.zip`을 생성했다.

## 규정 준수 설계

- 외부 데이터, 인터넷 데이터, 2025 TrackMan을 사용하지 않았다.
- 익명 `pitcher_id`와 `pitcher_trackman_id`, 서로 다른 팀 코드를 억지로 연결하지 않았다.
- 각 학습 행은 자신의 시즌보다 이전 시즌 TrackMan 프로파일만 사용한다.
- 2024 OOF는 2019--2023 TrackMan만, 2025 최종 추론은 2019--2024 TrackMan만 사용한다.
- TrackMan 집계는 학습 시 고정 JSON artifact로 저장하며 추론에서는 현재 test 행의 시즌·투수 손·공식 구종 비율로 lookup/가중 결합만 한다.
- test 다른 행의 집계·빈도·rolling·분포·보정은 전혀 사용하지 않는다.

## TrackMan 피처

`pitcher_hand × pitch_type_group` 계층에서 구속, 회전수, 수직/수평 무브먼트, extension, 릴리스 위치, zone speed, 속도 유지율의 recency-weighted 평균·표준편차를 계산한다. 현재 행의 공식 fastball/breaking/offspeed 비율로 기대 물리값을 만들고, fastball-breaking 차이와 무브먼트 분리도를 추가했다.

## Strict OOF 결과

| 모델 | 2024 untouched BSS |
|---|---:|
| TrackMan 물리 CatBoost 단독 calibrated | 679.906 |
| 기존 CatBoost 2 + MLP | 680.763 |
| CatBoost 2 + MLP + TrackMan | **694.257** |

최종 가중치:

- 전체기간 CatBoost: `0.216973`
- 최근 3시즌 CatBoost: `0.180436`
- embedding MLP: `0.277874`
- TrackMan 물리 CatBoost: `0.324717`

Calibration은 2024 hash mod 4 = 2 구간에서만 적합했고, hash mod 4 = 3 구간은 최종 평가에만 사용했다.

## 제출 감사

- ZIP 무결성: 통과
- 행 순서 변경 및 단일 행 추론 동일성: 통과
- 245,789행 추론: 약 `3.15초` (모델 로딩 제외)
- peak RSS: 약 `1.76 GiB`
- NaN/Inf 및 `[0,1]` 범위: 통과
- ZIP SHA-256: `195db34c30d7e8008342b4e643444c205f030ea4945cebd04bf11f42cf609684`

로컬 OOF 개선은 리더보드 개선을 보장하지 않으므로 기존 939점 ZIP은 보존한다.

## 논문 기반 command profile 후속 실험

기존 raw-pitch 평균·표준편차를 투수-시즌 단위 release/movement covariance, confidence ellipse, mechanics drift, active-spin proxy로 확장했다. 익명 ID를 TrackMan ID와 연결하지 않고 손 유형·구종군의 시점 안전 prior만 사용했다.

TrackMan command FactorTabM 20%와 DSF의 고정 혼합은 2024 eval에서 `+23.549 BSS`였다. 기존 1075 FactorTabM router를 85%, TrackMan 후보를 15% 사용하는 안정 혼합은 2023 eval `+610.278`, 2024 eval `+7.374 BSS`였다. 최종 제출 파일은 `tm_factor.zip`이다.
