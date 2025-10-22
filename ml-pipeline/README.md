# MLOps Kubeflow Pipeline

Kubeflow Pipelines를 사용한 MLOps 학습 파이프라인입니다. PostgreSQL 데이터베이스에서 데이터를 로드하고, PyTorch Lightning 모델을 학습하며, MLflow를 통해 모델을 평가하고 등록합니다.

## 주요 기능

- **데이터 로딩**: PostgreSQL 데이터베이스에서 데이터 로드 및 전처리
- **모델 학습**: PyTorch Lightning을 사용한 신경망 모델 학습
- **모델 평가**: 다양한 메트릭을 사용한 모델 성능 평가
- **MLflow 통합**: 실험 추적, 모델 등록, 버전 관리
- **Kubeflow 호환**: Kubeflow Pipelines에서 실행 가능한 컴포넌트

## 파이프라인 구조

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Load     │    │     Train       │    │   Evaluation    │
│                 │    │                 │    │                 │
│ - PostgreSQL    │───▶│ - PyTorch       │───▶│ - Metrics       │
│ - Preprocessing │    │ - Lightning     │    │ - MLflow        │
│ - Data Split    │    │ - MLflow Log    │    │ - Model Reg     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 컴포넌트

### 1. Data Load Component
- PostgreSQL 데이터베이스에서 데이터 로드
- 데이터 전처리 (스케일링, 인코딩, 결측값 처리)
- Train/Validation/Test 데이터 분할
- 전처리 아티팩트 저장

### 2. Train Component
- PyTorch Lightning 모델 학습
- MLflow를 통한 실험 로깅
- 체크포인트 저장 및 조기 종료
- 학습된 모델 저장

### 3. Evaluation Component
- 테스트 데이터로 모델 평가
- 회귀/분류 메트릭 계산
- MLflow Model Registry에 모델 등록
- 평가 결과 저장

## 설정

### 파이프라인 설정 (`config/pipeline_config.yaml`)

```yaml
# Database Configuration
database:
  host: "localhost"
  port: 5432
  database: "mlops"
  username: "postgres"
  password: ""
  query: "SELECT * FROM training_data WHERE created_at >= '2024-01-01'"

# Preprocessing Configuration
preprocessing:
  feature_columns: ["feature_1", "feature_2", ...]
  target_column: "target"
  scale_features: true
  encode_categorical: true
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

# Model Configuration
model:
  input_size: 10
  hidden_size: 64
  output_size: 1
  dropout: 0.2
  learning_rate: 0.001

# Training Configuration
training:
  max_epochs: 10
  batch_size: 32
  early_stopping_patience: 5
  save_checkpoints: true

# MLflow Configuration
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "mlops-training"
  model_name: "my-model"
```

## 사용 방법

### 1. 로컬 개발

```bash
# 의존성 설치
pip install -r requirements.txt

# 개별 컴포넌트 테스트
python components/data_load/data_loader.py --config config/pipeline_config.yaml --output ./output/data
python components/train/model_trainer.py --config config/pipeline_config.yaml --data-path ./output/data --output ./output/model
python components/evaluation/model_evaluator.py --config config/pipeline_config.yaml --model-path ./output/model --data-path ./output/data --output ./output/results
```

### 2. Docker 빌드

```bash
# Docker 이미지 빌드
docker build -f docker/Dockerfile -t mlops-pipeline-component .
```

### 3. Kubeflow 파이프라인 실행

```python
import kfp

# 파이프라인 컴파일
kfp.compiler.Compiler().compile(
    pipeline_func=mlops_training_pipeline,
    package_path="mlops_training_pipeline.yaml"
)

# Kubeflow에서 파이프라인 실행
client = kfp.Client()
run = client.create_run_from_pipeline_package(
    pipeline_file="mlops_training_pipeline.yaml",
    arguments={
        "config_path": "/config/pipeline_config.yaml",
        "experiment_name": "mlops-training"
    }
)
```

## 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `MLFLOW_TRACKING_URI` | MLflow 서버 주소 | `http://localhost:5000` |
| `MLFLOW_EXPERIMENT_NAME` | 실험 이름 | `mlops-training` |
| `MODEL_NAME` | 모델 이름 | `my-model` |
| `DB_HOST` | PostgreSQL 호스트 | `localhost` |
| `DB_PORT` | PostgreSQL 포트 | `5432` |
| `DB_NAME` | 데이터베이스 이름 | `mlops` |
| `DB_USER` | 데이터베이스 사용자 | `postgres` |
| `DB_PASSWORD` | 데이터베이스 비밀번호 | `` |

## 데이터베이스 스키마

### 예시 테이블 구조

```sql
CREATE TABLE training_data (
    id SERIAL PRIMARY KEY,
    feature_1 FLOAT,
    feature_2 FLOAT,
    feature_3 FLOAT,
    feature_4 FLOAT,
    feature_5 FLOAT,
    feature_6 FLOAT,
    feature_7 FLOAT,
    feature_8 FLOAT,
    feature_9 FLOAT,
    feature_10 FLOAT,
    target FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 출력 파일

### Data Load Component
- `train_data.csv`: 학습 데이터
- `validation_data.csv`: 검증 데이터
- `test_data.csv`: 테스트 데이터
- `preprocessing_artifacts.pkl`: 전처리 아티팩트
- `data_info.json`: 데이터 정보

### Train Component
- `model/`: MLflow 모델 디렉토리
- `training_metadata.json`: 학습 메타데이터

### Evaluation Component
- `evaluation_results.json`: 평가 결과

## 확장 가능성

### 새로운 데이터 소스 추가
1. `components/data_load/` 디렉토리에 새로운 로더 구현
2. 설정 파일에서 데이터 소스 타입 선택

### 새로운 모델 타입 추가
1. `components/train/` 디렉토리에 새로운 트레이너 구현
2. 설정 파일에서 모델 타입 선택

### 새로운 평가 메트릭 추가
1. `components/evaluation/` 디렉토리에서 메트릭 계산 로직 수정
2. 설정 파일에서 메트릭 목록 정의

## 문제 해결

### 일반적인 문제

1. **데이터베이스 연결 실패**
   - PostgreSQL 서버가 실행 중인지 확인
   - 연결 정보가 올바른지 확인

2. **MLflow 연결 실패**
   - MLflow 서버가 실행 중인지 확인
   - 네트워크 연결 확인

3. **메모리 부족**
   - 배치 크기 줄이기
   - 데이터 샘플링

4. **GPU 사용 불가**
   - CUDA 설치 확인
   - PyTorch GPU 버전 확인

## 라이선스

MIT License
