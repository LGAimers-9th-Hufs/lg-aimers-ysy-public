# DSF compliance re-audit — 2026-08-28

## 결론

`DSF_CHAL061_FIX` champion과 이를 포함한 파생 제출물은 더 이상 제출 후보로 사용하지 않는다. 신규 `clean_moe`는 해당 champion을 전혀 포함하지 않는다.

## 확인된 문제

1. `champion/model/metadata.json`의 provenance에 `seg_probe t=0.00570`이 공개 리더보드의 `+delta/-delta` 제출 점수로부터 산출됐다고 명시돼 있다.
2. 같은 metadata의 `regime_transfer_probe`는 `purpose: one-point exact quadratic blend inversion`과 공개 점수 형태의 값을 저장하고, 추론 코드는 해당 혼합 가중치를 하드코딩한다.
3. train의 익명 `pitcher_id` 792개와 공식 TrackMan의 `pitcher_trackman_id` 906개 사이 교집합은 0이다. 그런데 champion에는 train 투수 ID로 조회하는 TrackMan 물리 프로필이 존재하며, 이 매핑을 재현할 원본 코드와 provenance가 번들에 없다.

1·2는 대회 규칙의 리더보드 프로빙 금지에 저촉될 가능성이 높다. 3은 공식 데이터만으로 만들어졌는지 검증할 수 없어 “규정상 애매한 방법은 구현하지 않는다”는 프로젝트 원칙에 따라 제외한다.

## 영향 범위

다음 artifact는 모두 원본 DSF champion을 포함하므로 역사적 결과로만 보존하고 제출하지 않는다.

- `DSF_CHAL061_FIX`
- `submit_dsf_*`
- `factor_tabm_router`
- `submit_dsf_factor_tabm_3seed_router`
- `tm_factor`
- `seed1_tm_gate`
- `seed1_tm_res`

## clean_moe provenance

- Independent Challenger: 공식 train만 사용하며 학습 코드와 2022~2024 walk-forward OOF가 존재한다.
- FactorTabM 세 seed: 각 검증 시즌보다 이전 시즌만 학습한 OOF다.
- TrackMan command FactorTabM: 공식 2019~2024 TrackMan만 사용하며 예측 시즌보다 이전 시즌의 hand×pitch-group prior만 제공한다.
- Residual MoE: 2024 OOF stop에서 적합하고 select에서 구조·cap을 선택한 뒤 evaluate를 한 번 확인했다. 최종 모델은 구조 고정 후 2024 OOF 전체로 재적합했다.
- 추론: test 각 행의 공식 컬럼, 사전 학습 모델 예측, frozen artifact만 사용한다.

## 독립성 및 실행 검증

- 한 행 실행과 5행 배치 실행 최대 차이: `0.0`
- 입력 행 순서 반전 최대 차이: `0.0`
- 245,789행 합성 추론: 약 `7.02초`
- 출력: 전부 유한하며 `[0, 1]` 범위
- ZIP: 약 4.1MB, 설치 requirements는 `lightgbm==4.6.0`만 포함
- 추론 ZIP에서 `groupby/rolling/expanding/cumsum/cumcount/shift` 정적 검색: 0건
- ZIP SHA-256: `9d61bf8b450fb1042dc8e5aaee71aa7a5fbb13c9bebc25d256b1fa2b5f51e61e`
