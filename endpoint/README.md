# MLOps Endpoint API

MLOps 플랫폼을 위한 모델 서빙 API 엔드포인트입니다. PyTorch 모델과 LLM 모델을 모두 지원하며, MLflow를 통한 모델 관리와 피처 보강 기능을 제공합니다.

## 주요 기능

- **다중 모델 지원**: PyTorch Lightning 모델과 LangGraph/LLM 모델 지원
- **MLflow 통합**: MLflow Model Registry를 통한 모델 로딩 및 관리
- **피처 보강**: Feature Store 연동을 통한 실시간 피처 보강
- **RESTful API**: FastAPI 기반의 표준화된 API 엔드포인트
- **컨테이너화**: Docker를 통한 배포 지원

## 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │  Model Service  │    │ Feature Service │
│                 │    │                 │    │                 │
│  /predict       │───▶│ ModelWrapper    │    │ Feature Store   │
│  /invoke        │    │ - PyTorch       │    │ Integration     │
│  /health        │    │ - LLM           │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│   MLflow        │    │   Data Sources   │
│   Model Registry│    │ - Local Files    │
│                 │    │ - Feature Store │
└─────────────────┘    └─────────────────┘
```

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 설정

`config/config.yaml` 파일을 수정하여 설정을 변경할 수 있습니다:

```yaml
# MLflow Settings
MLFLOW_TRACKING_URI: "http://localhost:5000"
MLFLOW_MODEL_NAME: "my-model"
MLFLOW_MODEL_STAGE: "Production"

# Model Settings
MODEL_TYPE: "pytorch"  # Options: "pytorch", "llm"
```

### 3. 개발 서버 실행

```bash
python -m app.main
```

또는

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### 4. Docker로 실행

```bash
# 이미지 빌드
docker build -t mlops-endpoint .

# 컨테이너 실행
docker run -p 8080:8080 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MLFLOW_MODEL_NAME=my-model \
  -e MLFLOW_MODEL_STAGE=Production \
  mlops-endpoint
```

## API 엔드포인트

### 1. 헬스체크

```bash
GET /health
```

응답:
```json
{
  "status": "healthy",
  "message": "API is running and model is loaded",
  "model_loaded": true,
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

### 2. 예측 (PyTorch 모델)

```bash
POST /predict
Content-Type: application/json

{
  "user_id": "123",
  "features": [1.0, 2.0, 3.0],
  "metadata": {}
}
```

응답:
```json
{
  "prediction": [0.5, 0.3, 0.2],
  "model_type": "pytorch",
  "status": "success"
}
```

### 3. 예측 (LLM 모델)

```bash
POST /predict
Content-Type: application/json

{
  "user_id": "123",
  "prompt": "What is machine learning?",
  "metadata": {}
}
```

응답:
```json
{
  "response": "Machine learning is a subset of artificial intelligence...",
  "model_type": "llm",
  "status": "success"
}
```

## 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `MLFLOW_TRACKING_URI` | MLflow 서버 주소 | `http://localhost:5000` |
| `MLFLOW_MODEL_NAME` | 모델 이름 | `my-model` |
| `MLFLOW_MODEL_STAGE` | 모델 스테이지 | `Production` |
| `MODEL_TYPE` | 모델 타입 | `pytorch` |
| `API_HOST` | API 호스트 | `0.0.0.0` |
| `API_PORT` | API 포트 | `8080` |
| `FEATURE_STORE_ENABLED` | 피처 스토어 사용 여부 | `false` |
| `OPENAI_API_KEY` | OpenAI API 키 | `` |

## 개발 가이드

### 새로운 모델 래퍼 추가

1. `app/models/` 디렉토리에 새로운 래퍼 클래스 생성
2. `ModelWrapper` 추상 클래스를 상속
3. `load()` 및 `predict()` 메서드 구현
4. `ModelService`에서 새로운 모델 타입 지원 추가

### 피처 스토어 연동

1. `FeatureService` 클래스 수정
2. `_enrich_from_feature_store()` 메서드 구현
3. 실제 피처 스토어 API 연동 코드 추가

## 라이선스

MIT License
