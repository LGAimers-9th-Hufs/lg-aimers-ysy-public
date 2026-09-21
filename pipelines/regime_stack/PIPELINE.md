# Clean regime stack

공식 `train.csv`만으로 새로 학습한 specialist/state/invariant LightGBM과 규정 준수 `clean_moe`를 결합한다. 팀원 DSF의 모델 예측, 공개 리더보드 보정, TrackMan ID 매핑은 사용하지 않는다.

`regime/features.py`의 행 단위 피처 공식은 기존 DSF 패키지에서 코드만 분리해 감사한 뒤 재사용했다. 해당 피처는 한 행 내부 값만 변환하며, DSF의 학습 모델·예측값·블렌딩 값은 가져오지 않았다. 세 LightGBM 모델은 공식 `train.csv`로 처음부터 다시 학습한다.

- specialist: 가장 최근 학습 시즌만 사용
- state/invariant: 연도당 `0.50` 감쇠
- 2024 select에서 고정한 regime 내부 혼합: `0.10 / 0.05 / 0.85`, shift `-0.014`
- clean/regime 최종 혼합: `0.70 / 0.30`, shift `-0.002`
- 2024 untouched: clean 대비 `+28.281 BSS`
- F: `+120.448`, R: `+16.175`
- 단일 행/배치/역순 최대 차이: `0.0`
- 245,789행 합성 추론: 약 `8.48초`

모든 추론 피처는 현재 행과 train에서 동결한 artifact만 사용한다.
