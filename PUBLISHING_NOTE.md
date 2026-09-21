# PUBLISHING_NOTE — `lg-aimers-ysy-public`

원본: `LGAimers-9th-Hufs/lg-aimers-ysy` @ `99aac056adc8` · 생성: `publish/export.py` (fresh history, 단일 커밋)

> 데이콘(2026-09-21): "대회 종료 이후 참가자가 직접 작성한 코드는 개인 GitHub 등을 통해 공개하셔도 무방하며 … 대회에서 제공된 데이터 및 파일 자체, 개인정보·민감정보 등 외부 공개가 제한되는 내용은 포함하여 공개할 수 없으므로 반드시 제외해 주시기 바랍니다."

이 저장소는 대회 기간 중 사용한 private 저장소 `lg-aimers-ysy`의 **정제 스냅샷**(단일 커밋)입니다.

제외한 것:
- `model_artifacts/**` 전체 — 제출 zip 25종, 압축 해제본의 모델 가중치(`*.joblib`, `*.cbm`, `*.txt`, `*.pt`), train 파생 룩업(`lookup.json`, `metadata.json`). 코드는 `pipelines/`·`shared/`에 있습니다.
- `docs/AUDIT.md`와 `pipelines/clean_rescat/*.py`·`pipelines/clean_dsf2/train_residual_specialist.py`의 로컬 홈 디렉터리 경로(기본 인자)를 `<repo>`/`<local>`로 치환

## 제외 내역 (사유별 파일 수 / 크기)

| 제외 사유 | 파일 수 | 크기 | 예시 |
|---|---:|---:|---|
| 제출 zip·압축 해제본(가중치·룩업) | 942 | 1060.6 MB | `model_artifacts/clean_rescat/package/model/residual.cbm` |
| 명시 제외 파일(내부 조율 문서·판정 파일·강의 요약·데이터 설명서 사본 등) | 1 | 21.8 MB | `clean_rescat.zip` |

전체 목록은 원본(private) 저장소의 `publish/manifests/` manifest(json)에 기록되어 있습니다.
