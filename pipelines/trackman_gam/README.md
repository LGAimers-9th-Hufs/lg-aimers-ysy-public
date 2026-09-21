# ID-free TrackMan spline GAM

선수 ID를 제거하고 공식 `asof_*`, 상황 피처, 공식 TrackMan 과거 로그에서 미리 만든
손/구종군 물리 안정성 피처만 사용한 shape-constrained 대체 모델 실험이다.

## 2023 -> 2024 OOF

| Ridge alpha | select BSS | untouched evaluate BSS |
|---:|---:|---:|
| 10 | 0.173 | -2.019 |
| 30 | 0.292 | -1.892 |
| 100 | 1.621 | -2.700 |
| 300 | 4.947 | -1.264 |
| 1000 | 9.024 | 1.135 |

최선 모델도 `dsf_lookup`에 대한 select 최적 가중치가 `0.0`이었다. 0.5% 혼합은
untouched에서 `+0.016 BSS`지만 select에서 `-0.248 BSS`여서 채택할 수 없다.

TrackMan의 익명 pitcher ID와 메인 데이터 pitcher ID를 공식적으로 연결할 수 없기 때문에,
규정 준수 범위의 손/구종군 평균은 행 간 변별력이 거의 없다. 외부 매핑 없이 이 물리
집계만 복잡하게 만드는 접근은 추가 개선 가능성이 낮아 제출 후보에서 제외한다.

사용 데이터는 공식 train과 공식 TrackMan뿐이며 test 집계, 외부 데이터, LB feedback은 없다.
