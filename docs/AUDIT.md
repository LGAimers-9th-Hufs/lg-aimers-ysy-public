# LG Aimers 9기 Phase 2 프로젝트 감사 보고서

감사일: 2026-08-20 (Asia/Seoul)  
감사 범위: `<repo>`의 현재 루트 코드, 모델, 제출 디렉터리, `submit_v2.zip`, 문서, 공식 제공 데이터 스키마  
원칙: 기존 학습·추론 코드와 모델은 수정하지 않고 읽기·실행 검증만 수행

## 1. 결론

현재 권장 제출물 `submit_v2.zip`은 **정적 감사와 로컬 실행 검증 범위에서 대회 규정을 준수하며 제출 가능한 상태**다.

- test 다른 행을 이용한 집계, 빈도, rolling, expanding, target encoding, 분포 보정이 없다.
- 외부 데이터, 외부 API, 네트워크를 사용하지 않는다.
- test 각 행의 피처는 해당 행 값과 공식 train label로 미리 저장한 정적 lookup만 사용한다.
- 학습과 추론은 동일한 `script.build_features()`를 사용하고, 79개 피처의 이름·순서가 세 모델과 일치한다.
- 루트, `submit/`, `submit_v2/`, ZIP 내부의 script/model/artifact는 SHA-256 기준 동일하다.
- 실제 5행 실행, 단일행 대 batch, 행 순서 shuffle, 245,789행 모사 검사를 통과했다.
- 245,789행 피처 생성+3모델 예측은 로컬에서 2.53초, peak RSS 약 1.0GB였다. 10분/28GB 제한 위험은 낮다.

**명시적인 부정행위 또는 규정 위반 구현은 발견되지 않았다.** 다만 검증 점수의 독립성, 학습-배포 TE 표현 차이, 문서 불일치는 아래와 같이 개선해야 한다. 규정상 애매한 아이디어는 구현하지 말아야 한다.

## 2. 감사 전제와 문서 상태

사용자가 지정한 `docs/data_description.md`, `docs/PROJECT_SUMMARY.md`는 전부 읽었다.

그러나 지정된 `docs/competition_rules.txt`는 프로젝트에 존재하지 않는다. 대신 프로젝트에 있는 다음 두 문서를 규칙 원문으로 읽고, 더 엄격한 기준을 적용했다.

- `docs/lg-aimers-대회-내용.txt`: 대회 설명, 공식 규칙, 코드 제출 환경
- `docs/prohibited_rules_guide (1).md`: 행 독립성 및 금지 행위 가이드

감사 재현성을 위해 실제 규칙 파일을 `docs/competition_rules.txt`라는 정확한 이름으로 보존하는 것을 권장한다. 단, 이번 감사에서는 파일을 새로 만들거나 다른 코드를 수정하지 않았다.

## 3. 감사 대상 현재 상태

| 구성 | 현재 상태 |
|---|---|
| 학습 코드 | `train.py`, LightGBM 3개 |
| 공용 피처 생성 | `script.py::build_features`, 학습 시 import하여 동일 코드 사용 |
| 모델 | `lgb0.txt`, `lgb1.txt`, `lgb2.txt` |
| 피처 | 79개 |
| 앙상블 | 고정 동일가중 1/3씩 |
| calibration | 2024 hash-even 절반으로 적합한 logit-affine |
| 정적 lookup | 공식 train label로 만든 pitcher/batter/player-team TE map |
| Trackman | V2에서 미사용 |
| 제출 requirements | `lightgbm==4.6.0` |
| ZIP 크기 | 약 2.7MB, 압축 해제 약 6.46MB |

과거 LGB+ET+HGB 구성은 `model_legacy_20260819/`와 `submit_legacy_20260819/`에 보존되어 있다. 이 legacy 경로는 현재 권장 제출물이 아니며, 이전 감사의 차단 이슈는 현재 V2에 그대로 적용되지 않는다.

## 4. 우선순위별 발견 사항

### P1-1. 보고된 validation BSS는 완전한 untouched holdout 점수가 아님

`train.py`는 2024 전체를 LightGBM `eval_set`으로 사용해 early stopping iteration을 고른다. 이후 2024를 row_id hash parity로 나누어 한 절반으로 calibration을 적합하고 다른 절반에서 BSS 743.545를 보고한다.

calibration label은 평가 절반과 분리됐으므로 이전 버전보다 명백히 개선됐다. 그러나 평가 절반의 label도 이미 세 모델의 early stopping에 사용됐다. 따라서 743.545는 calibration에는 독립이지만 **모델 선택에는 독립이 아니다**. 규정 위반이나 test leakage는 아니지만 일반화 BSS가 다소 낙관적일 수 있다.

안전한 개선:

1. 2023 이전으로 모델을 학습하고 2023으로 iteration을 고정한다.
2. 고정된 iteration으로 2019~2023 모델을 다시 적합한다.
3. 2024 일부는 calibration fit, 나머지는 모델 선택과 calibration 모두에 쓰지 않은 최종 평가로 둔다.
4. 더 바람직하게는 시즌 walk-forward OOF로 raw/calibration 성능을 평가한다.

### P1-2. 학습 TE와 2025 추론 TE의 표현·표본량 차이

학습 피처의 TE는 각 행 시즌보다 이전 시즌의 label만 사용한다. 예를 들어 2024 학습행은 2019~2023 map을 쓴다. 반면 2025 추론은 공식 2019~2024 전체 train label map을 쓴다.

이는 test 행 간 정보를 쓰지 않으므로 **규정 준수**이며 미래 target leakage도 아니다. 다만 최종 모델이 학습에서 본 TE 분포와 배포 시 TE 분포가 다르다. 특히 2019행은 history가 없어 0.5 부근만 보고, test는 가장 풍부한 map을 본다.

안전한 개선은 시즌 walk-forward validation에서 이 배포 조건을 정확히 재현하고, cross-fitting 또는 leave-fold-out 방식으로 더 풍부하면서 누수 없는 train TE를 만드는 것이다. 같은 행의 target이 자기 TE에 들어가는 단순 full-train TE는 사용하면 안 된다.

### P1-3. `PROJECT_SUMMARY.md`가 현재 V2와 크게 불일치

문서 상단은 V2 보고서를 우선하라고 안내하지만 본문에는 legacy 내용이 남아 있다.

- Trackman 파생 통계를 사용한다고 설명하지만 V2는 Trackman을 사용하지 않는다.
- 최종 validation 769.7이라고 쓰지만 현재 독립 calibration 평가 보고값은 743.545다.
- 모델 하이퍼파라미터, 최근 시즌 가중치, calibration 종류가 현재 `train.py`와 다르다.
- `--trackman_path` 인자를 안내하지만 현재 `train.py`에는 해당 인자가 없다.
- 파일 구조에 실제 존재하지 않는 `requirements_submit.txt`, `README.md`를 적었다.
- 루트 `requirements.txt`를 느슨한 학습용이라고 설명하지만 실제 내용은 `lightgbm==4.6.0` 한 줄이며, 학습 의존성은 `requirements_train.txt`에 있다.

이는 모델 실행 오류는 아니지만 재현·검증 시 잘못된 코드를 제출하거나 잘못된 점수를 인용하게 할 수 있다. `MODEL_V2_REPORT.md`와 실제 코드 기준으로 문서를 다시 작성해야 한다.

### P2-1. artifact provenance가 모델 파일 자체를 완전히 고정하지 않음

현재 artifact에는 train SHA-256, train 행 수, Python/LightGBM/pandas/numpy 버전, seed가 있다. 실제 `data/train.csv` SHA-256은 artifact의 `d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff`와 일치했다.

그러나 다음 정보는 없다.

- `train.py`와 `script.py` SHA-256 또는 git commit
- 각 `lgb*.txt` SHA-256
- `artifacts.json` 생성 시각
- OS/CPU 정보
- calibration fit/eval row_id hash 목록 또는 그 digest
- 제출 ZIP hash

모델/artifact 혼합을 더 확실히 차단하려면 이 값을 별도 manifest에 저장하고 제출 전 검증해야 한다.

### P2-2. 학습·평가 서버 패키지 버전 차이

artifact 학습 환경은 Python 3.12.14, pandas 2.2.3, LightGBM 4.6.0이다. 평가 서버는 Python 3.11.15, 기본 pandas 2.0.3, numpy 1.26.4이며 제출 requirements는 LightGBM 4.6.0만 설치한다.

모델은 LightGBM 텍스트 형식이고 pandas 객체를 직렬화하지 않아 위험은 낮다. 사용 API도 pandas 2.0.3에서 지원된다. 그래도 최종 preflight는 서버와 동일한 Python 3.11/pandas 2.0.3에서 수행하는 것이 안전하다.

### P2-3. calibration 타입 필드 미검증

artifact는 `calibration.type = logit_affine`을 저장하지만 `apply_calibration()`은 type을 검사하지 않고 항상 `a`, `b`를 적용한다. 현재 artifact와는 일치하므로 오류가 없지만, 향후 artifact schema가 바뀌면 조용한 오적용 가능성이 있다. schema/type 검증을 추가하는 것이 안전하다.

### P3-1. 범주형 결측값 문자열 처리

범주형은 `astype(str)` 후 map한다. 학습 cat map을 만들 때는 `dropna()`를 하므로 결측값은 inference에서 `"nan"`이 되어 -1로 처리된다. 이는 train/inference parity는 맞고 LightGBM도 처리 가능하지만, 명시적 missing category와 실제 문자열 `"nan"`이 구분되지 않는다. 현재 데이터에서 문제가 되는지는 별도 분포 점검이 필요하다.

## 5. 요청 항목별 감사 결과

### 5.1 Train/inference feature parity

**통과.**

- `train.py:11`이 `script.build_features`를 직접 import한다.
- 학습과 추론 모두 artifact 기반 동일 피처 함수를 사용한다.
- artifact는 79개 `feature_cols`를 고정한다.
- 실제 세 LightGBM 모델의 feature name과 순서가 artifact와 완전히 같았다.
- 모델별 tree 수는 artifact best rounds와 일치했다: 253, 333, 178.
- missing/extra 피처가 있으면 즉시 예외를 발생시키며 0 대치 fallback이 없다.

주의점은 피처 코드 자체 parity가 아니라 P1-2의 TE 학습 표현과 배포 표현 차이다.

### 5.2 Data leakage 가능성

**test leakage는 발견되지 않았다.**

- 추론은 test target을 읽지 않는다.
- test의 다른 행 값 또는 예측값을 피처에 사용하지 않는다.
- calibration과 TE는 공식 train label로 사전 적합된 고정 artifact다.
- V2는 Trackman 전체기간 집계를 제외해 과거 validation의 temporal contamination도 제거했다.

검증 설계상 주의:

- P1-1처럼 2024 평가 절반 label이 early stopping에 사용된다.
- 이는 제출 규정 위반이 아니라 validation estimate contamination이다.
- `cat_maps`는 모든 train 시즌의 범주 목록으로 만든다. target이나 빈도를 쓰지 않는 단순 vocabulary이므로 성능 누수 영향은 작지만, 엄격한 fold simulation에서는 history 범위에서만 만드는 편이 더 깨끗하다.

### 5.3 대회 규정 위반 가능성

**현재 `script.py`는 정적·동적 검사 모두 통과.**

| 금지 항목 | 결과 | 근거 |
|---|---|---|
| test groupby/집계/빈도 | 없음 | 추론 코드에 groupby 없음 |
| rolling/expanding/shift/cumsum/cumcount | 없음 | 해당 연산 없음 |
| test target encoding | 없음 | train-only 정적 JSON map 조회 |
| test mean/std/min/max/median scaling | 없음 | 고정 상수와 artifact만 사용 |
| 예측 rank/quantile/평균 맞춤 | 없음 | 각 확률에 고정 logit-affine만 적용 |
| 외부 데이터 | 없음 | 추론은 test, sample submission, 로컬 model만 읽음 |
| 외부 API/네트워크 | 없음 | 관련 import/호출 없음 |
| 다른 test 행 종속 | 없음 | 단일행·batch·shuffle 결과 정확히 동일 |

### 5.4 모델 파일과 `script.py` 불일치

**현재 V2는 통과.**

- `script.py` SHA-256은 루트, `submit/`, `submit_v2/`가 모두 동일하다.
- 모델 3개와 artifact도 세 위치가 각각 동일한 SHA-256이다.
- ZIP에는 최상위 `script.py`, `requirements.txt`, `model/`만 있고 추가 상위 폴더가 없다.
- artifact의 `model_files`, weight 개수, 모델 feature name을 실행 시 검증한다.

과거 legacy 모델은 현재 V2 script와 섞어 제출하면 안 된다.

### 5.5 245,789행 메모리/속도

**위험 낮음.**

동일한 5개 공식 샘플 행을 245,789행으로 복제하고 row_id만 고유하게 만든 보수적 실행 검증 결과:

- 피처 생성 + 세 모델 예측: 2.528초
- 원본 대형 DataFrame deep memory: 약 132.3MB
- 전체 프로세스 peak RSS: 약 1.045GB (macOS 측정)
- 출력은 finite이며 범위는 `[0,1]`
- 5행 ZIP staging 전체 실행은 첫 import/라이브러리 초기화를 포함해 약 13초

평가 서버가 더 느려도 600초와 28GB 한도에는 큰 여유가 있다. 다만 실제 비공개 데이터의 문자열 다양성과 Linux/Python 3.11 환경 차이를 반영한 서버 동일 환경 측정이 최종 확인이다.

### 5.6 학습-추론 artifact 불일치

**현재 파일 일치성은 통과, provenance는 보강 필요.**

- train file hash와 artifact provenance가 일치한다.
- feature list/model files/blend/calibration이 script 계약과 일치한다.
- 루트/제출 디렉터리/ZIP의 파일 내용이 같다.
- P1-2의 TE 표현 차이는 의도된 temporal OOF 대 full-history deployment 차이지만 검증으로 성능 영향을 확인해야 한다.
- P2-1처럼 코드/모델 hash까지 artifact가 스스로 증명하지는 않는다.

### 5.7 Brier Skill Score 관점 개선점

우선순위는 더 복잡한 모델보다 **out-of-time 확률 품질을 정직하게 측정하는 것**이다.

1. 모델 선택, calibration fit, 최종 평가를 시간 또는 시즌 기준으로 완전히 분리한다.
2. walk-forward OOF에서 raw Brier, calibrated Brier, BSS를 fold별로 기록한다.
3. reliability diagram과 확률 bin별 예측 평균/실제 성공률/표본 수를 확인한다.
4. logit-affine, Platt, beta calibration, 제한된 isotonic을 동일 OOF에서 비교한다.
5. 고정 동일가중 ensemble이 단일 최우수 모델보다 OOF Brier를 실제로 낮추는지 검증한다.
6. 최근 시즌 가중치 0.10의 민감도를 fold별로 비교한다. 리더보드 결과로 test 평균이나 calibration 상수를 역산해서는 안 된다.
7. TE prior 80/800 및 smoothing 25/100/400은 오직 train OOF에서 튜닝한다.
8. 공식 row-level `asof_*` 피처의 missingness indicator와 단조성/상호작용을 OOF로 검증할 수 있다.
9. 모델별 예측 상관과 residual covariance를 보고, 상관이 충분히 낮을 때만 ensemble을 늘린다.
10. BSS는 resolution뿐 아니라 calibration에 민감하므로, calibration 개선이 여러 시즌에서 일관될 때만 배포한다.

## 6. 규정상 애매하여 구현하면 안 되는 아이디어

다음은 현재 구현에 없으며, 운영진의 서면 확인 없이는 구현하지 말아야 한다.

- test에서 같은 투수·타자·팀·경기의 행 수나 빈도를 계산하는 것
- test 시간순 정렬 후 이전 행을 사용하는 rolling/expanding/lag
- test 전체 결측률, 평균, 표준편차, 최소·최대, 분위수를 이용하는 전처리
- test 예측 평균을 train 성공률이나 리더보드에서 추정한 성공률에 맞추는 보정
- 리더보드 점수 변화로 test target 평균 또는 일부 label을 역산하는 프로빙
- test에 나타난 ID 목록·빈도·공동출현을 이용해 train lookup 또는 prior를 갱신하는 것
- 평가 데이터의 여러 행을 묶어 그래프/sequence/batch context를 만드는 것
- 외부 웹, 공개 야구 기록, API, 별도 수집 데이터 사용
- Trackman과 선수를 명시적 키 없이 test 분포를 이용해 사후 매칭하는 것

공식 `trackman_history.csv` 자체는 사용 가능하지만, train 단계에서만 통계를 만들고 test에서는 단일 행의 키로 고정 lookup하는 명확한 방식만 권장한다. 키가 불명확한 soft matching은 운영진 확인 전에는 보류한다.

## 7. 제출 전 체크리스트

- [x] test 다른 행 기반 집계/빈도/rolling/expanding 없음
- [x] test 분포/예측 분포 보정 없음
- [x] 외부 데이터/API/네트워크 없음
- [x] 단일행과 batch 예측 완전 동일
- [x] shuffle 전후 동일 행 예측 완전 동일
- [x] train/inference 공용 feature builder
- [x] 모델 3개와 artifact의 79개 피처 일치
- [x] root/submit/submit_v2/ZIP hash 일치
- [x] test/submission ID 유일성·집합·행 수 검사
- [x] NaN/Inf 검사와 확률 범위 만족
- [x] 245,789행 모사 시간·메모리 제한 통과
- [x] ZIP 최상위 구조 정상
- [ ] 실제 `docs/competition_rules.txt` 원문 복구 또는 정확한 파일명으로 정리
- [ ] Python 3.11.15 + pandas 2.0.3 Linux 환경 최종 preflight
- [ ] 모델 선택에도 미사용한 완전 독립 validation 점수 산출
- [ ] `PROJECT_SUMMARY.md`를 현재 V2와 동기화
- [ ] 코드·모델·ZIP SHA-256 manifest 추가

## 8. 최종 판정

현재 `submit_v2.zip`에서 **명백한 규정 위반, test 행 간 정보 사용, 외부 데이터 사용, train/inference parity 오류, 모델-script 불일치는 발견되지 않았다.** 기존 legacy 제출 차단 문제는 V2에서 해결됐다.

따라서 현재 파일은 제출 가능 판정이다. 다만 BSS 743.545는 calibration 평가 분리에는 성공했어도 model early stopping과 완전히 독립된 holdout 점수는 아니므로, 성능 수치의 보수적 해석과 nested temporal 검증을 권장한다. 규정상 애매한 test 기반 피처나 분포 보정은 어떤 형태로도 추가하지 않는다.

## 9. 감사 한계

- 실제 평가 test 245,789행은 비공개이므로 공식 5행을 복제해 성능을 모사했다.
- 로컬 실행은 macOS ARM64, Python 3.12.14, pandas 2.2.3, LightGBM 4.6.0이었다. 평가 서버 Linux x86_64/Python 3.11과 완전히 같지는 않다.
- 외부 데이터 미사용 판정은 현재 프로젝트 파일과 코드 참조를 기준으로 한다. 프로젝트 밖의 과거 학습 과정까지 법의학적으로 증명한 것은 아니다.
- 지정된 `docs/competition_rules.txt`가 없어 프로젝트 내 대회 원문과 금지 가이드를 대체 기준으로 사용했다.
