# MLOps Model Wrapper

MLOps 플랫폼을 위한 모델 래퍼 라이브러리입니다. PyTorch Lightning과 LLM 모델의 학습, 평가, 추론을 위한 통합 인터페이스를 제공하며, Kubeflow Pipelines에서 Docker 이미지로 사용할 수 있습니다.

## 주요 기능

- **다중 모델 지원**: PyTorch Lightning과 LLM 모델 지원
- **MLflow 통합**: 자동 로깅, 모델 등록, 버전 관리
- **데이터 로더**: 로컬 파일과 피처 스토어 지원
- **CLI 인터페이스**: 명령줄에서 학습/평가/추론 실행
- **Kubeflow 호환**: Docker 이미지로 Kubeflow에서 실행 가능
- **설정 기반**: YAML 설정 파일로 모든 파라미터 관리

## 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │    │  Model Wrappers │    │  Data Loaders   │
│                 │    │                 │    │                 │
│  train          │───▶│ PyTorchWrapper  │    │ LocalLoader     │
│  evaluate       │    │ LLMWrapper      │    │ FeatureStore    │
│  predict        │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Config        │    │   MLflow        │    │   Data Sources  │
│   Manager       │    │   Utils         │    │ - CSV/Parquet   │
│                 │    │                 │    │ - Feature Store │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 설치 및 사용

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 설정 파일 준비

`config/training_config.yaml` 파일을 수정하여 모델과 데이터 설정을 구성합니다:

```yaml
# Model Configuration
model:
  type: "pytorch"
  input_size: 10
  hidden_size: 64
  output_size: 1
  learning_rate: 0.001

# Training Configuration
training:
  max_epochs: 10
  batch_size: 32

# Data Configuration
data:
  loader_type: "local"
  file_path: "/data/training_data.csv"
  feature_columns: ["feature_1", "feature_2", ...]
  target_column: "target"

# MLflow Configuration
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "mlops-training"
```

### 3. CLI 사용

#### 모델 학습
```bash
python cli.py train \
  --config config/training_config.yaml \
  --output /path/to/output
```

#### 모델 평가
```bash
python cli.py evaluate \
  --config config/training_config.yaml \
  --model-uri "models:/my-model/1"
```

#### 모델 추론
```bash
python cli.py predict \
  --config config/training_config.yaml \
  --model-uri "models:/my-model/1" \
  --input '{"features": [1.0, 2.0, 3.0]}'
```

## Docker 사용

### 이미지 빌드
```bash
docker build -f docker/Dockerfile -t mlops-model-wrapper .
```

### 컨테이너 실행
```bash
docker run -v /path/to/data:/data \
  -v /path/to/output:/app/output \
  -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 \
  mlops-model-wrapper
```

## Kubeflow Pipelines 통합

### 1. Docker 이미지 준비
```bash
# 이미지 빌드 및 레지스트리에 푸시
docker build -f docker/Dockerfile -t your-registry/mlops-model-wrapper:latest .
docker push your-registry/mlops-model-wrapper:latest
```

### 2. Kubeflow 컴포넌트 정의
```python
from kfp import dsl

@dsl.component(
    base_image="your-registry/mlops-model-wrapper:latest",
    packages_to_install=[]
)
def train_model_component(
    config_path: str,
    data_path: str,
    output_path: str,
    mlflow_tracking_uri: str
):
    """Kubeflow 컴포넌트로 모델 학습."""
    import subprocess
    
    cmd = [
        "python", "cli.py", "train",
        "--config", config_path,
        "--output", output_path
    ]
    
    env = {
        "MLFLOW_TRACKING_URI": mlflow_tracking_uri,
        "DATA_SOURCE_PATH": data_path
    }
    
    subprocess.run(cmd, env=env, check=True)
```

### 3. 파이프라인 정의
```python
@dsl.pipeline(
    name="mlops-training-pipeline",
    description="MLOps training pipeline"
)
def mlops_pipeline(
    data_path: str = "/data",
    config_path: str = "/config/training_config.yaml",
    mlflow_tracking_uri: str = "http://mlflow-server:5000"
):
    # 데이터 준비
    get_data_task = get_data_component(data_path=data_path)
    
    # 모델 학습
    train_task = train_model_component(
        config_path=config_path,
        data_path=get_data_task.outputs["data_path"],
        output_path="/output",
        mlflow_tracking_uri=mlflow_tracking_uri
    )
    
    # 모델 평가
    evaluate_task = evaluate_model_component(
        config_path=config_path,
        model_uri=train_task.outputs["model_uri"],
        data_path=get_data_task.outputs["test_data_path"]
    )
    
    # 모델 등록 (조건부)
    register_task = register_model_component(
        model_uri=train_task.outputs["model_uri"],
        evaluation_score=evaluate_task.outputs["score"],
        threshold=0.8
    )
```

## 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `MLFLOW_TRACKING_URI` | MLflow 서버 주소 | `http://localhost:5000` |
| `MLFLOW_EXPERIMENT_NAME` | 실험 이름 | `mlops-training` |
| `MODEL_TYPE` | 모델 타입 | `pytorch` |
| `DATA_SOURCE_PATH` | 데이터 파일 경로 | - |
| `DATA_LOADER_TYPE` | 데이터 로더 타입 | `local` |
| `BATCH_SIZE` | 배치 크기 | `32` |
| `MAX_EPOCHS` | 최대 에포크 수 | `10` |
| `LEARNING_RATE` | 학습률 | `0.001` |

## 개발 가이드

### 새로운 모델 래퍼 추가

1. `src/models/` 디렉토리에 새로운 래퍼 클래스 생성
2. `BaseModelWrapper` 추상 클래스를 상속
3. `train()`, `evaluate()`, `load()`, `predict()`, `save_model()` 메서드 구현
4. CLI에서 새로운 모델 타입 지원 추가

### 새로운 데이터 로더 추가

1. `src/data_loaders/` 디렉토리에 새로운 로더 클래스 생성
2. `BaseDataLoader` 추상 클래스를 상속
3. `load_data()`, `get_train_data()`, `get_validation_data()`, `get_test_data()` 메서드 구현
4. 설정 파일에서 새로운 로더 타입 지원 추가

## 라이선스

MIT License
