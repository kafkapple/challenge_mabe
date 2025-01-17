# MABe 2022 Mouse Triplets

https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022/problems/mabe-2022-mouse-triplets/leaderboards
https://www.youtube.com/watch?v=rPclAkXtTEY

- Dataset
%aicrowd ds dl -c {aicrowd_challenge_name} -o data # Download all files
%aicrowd ds dl -c {aicrowd_challenge_name} -o data *submission_data* # download only the submission keypoint data
%aicrowd ds dl -c {aicrowd_challenge_name} -o data *user_train* # download data with the public task labels provided

## 주요 클래스 및 함수

### MouseTrackingData
- 데이터 다운로드 및 로드
- 키포인트 홀 채우기
- 제출용 임베딩 생성

### BaseEmbedder
- PCAEmbedder: PCA 기반 임베딩
- RandomProjectionEmbedder: 랜덤 프로젝션 임베딩
- 공통 transform 인터페이스 제공

### EvaluationMetrics
- 데이터셋 정보 출력
- 학습/검증 데이터 분할
- 평가 메트릭 계산

### MouseVisualizer
- 마우스 포즈 시각화
- 시퀀스 애니메이션 생성

## 성능 최적화 옵션

1. 병렬 처리:
   - `n_jobs` 파라미터로 작업 수 조절
   - 기본값 -1은 모든 가용 코어 사용

2. 데이터 샘플링:
   - `sampling_ratio`로 학습 데이터 비율 조절
   - 메모리 사용량과 학습 시간 최적화

3. 임베딩 방법 선택:
   - PCA: 정확한 차원 축소
   - Random Projection: 빠른 처리 속도
## Others
### New baseline code
https://github.com/AndrewUlmer/MABe_2022_TVAE
### Other dataset
MABe 2022 Mouse Triplets Video Data (Not implemented yet)
https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022/problems/mabe-2022-mouse-triplets-video-data/leaderboards
