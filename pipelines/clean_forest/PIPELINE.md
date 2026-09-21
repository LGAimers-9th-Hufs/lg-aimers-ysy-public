# Clean forest and lookup research

공식 `train.csv`만 사용해 `clean_regime`과 오차가 다른 후보를 탐색한다. test의 다른 행, 외부 데이터, 리더보드 점수를 학습·보정에 사용하지 않는다.

- ExtraTrees recent: 26MB, 2024 untouched 단독 `677.774 BSS`
- ExtraTrees temporal: 51MB, 2024 untouched 단독 `583.569 BSS`
- forest 최종 혼합: clean 대비 약 `+1.07 BSS`로 채택하지 않음
- temporal nonlinear residual: clean 대비 `-0.87 BSS`로 채택하지 않음
- F/R 분리 lookup: clean 대비 `-11.44 BSS`로 채택하지 않음
- 전역 hierarchical lookup: clean 대비 `+8.915 BSS`, F/R 모두 개선

최종 `clean_lookup`은 2024 공식 학습 라벨과 OOF 예측으로 투수·타자·카운트 계층 잔차 lookup을 동결한다. 추론 시 각 행의 키로 동봉 lookup을 조회할 뿐, 다른 test 행을 집계하지 않는다.
