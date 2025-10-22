# MLOps 플랫폼 구축을 위한 개발 요구사항 명세서

## 1. 개요 (Overview)

본 문서는 자체 학습 모델(PyTorch)과 LLM 기반 모델(GPT, LangGraph)을 모두 지원하는 확장 가능한 MLOps 플랫폼 구축을 위한 기술 요구사항을 정의합니다.

목표는 모델 서빙을 위한 API 엔드포인트와 모델 학습/평가를 위한 모듈화된 파이프라인을 구축하여, 신속하고 안정적인 모델 배포 및 관리를 자동화하는 것입니다.

## 2. 핵심 아키텍처 원칙 (Core Principles)

1. **관심사의 분리 (Separation of Concerns):**
    - **서빙 API (Endpoint):** 실시간 추론 요청/응답 처리에만 집중합니다.
    - **모델 래퍼 (Wrapper):** 모델의 실제 로직(학습, 추론, 평가)을 캡슐화합니다.
    - **데이터 로더 (Loader):** 데이터 소스 접근 및 전처리를 캡슐화합니다.
    - **파이프라인 (Pipeline):** KFP를 통해 학습/평가/배포 워크플로우를 오케스트레이션합니다.
2. **모델 추상화 (Model Abstraction):** 서빙 API는 모델이 PyTorch인지 LangGraph인지 알 필요 없이, 표준화된 `ModelWrapper` 인터페이스를 통해 모델을 로드하고 실행합니다.
3. **중앙집중식 아티팩트 관리 (Centralized Artifacts):** 모든 모델 아티팩트, 코드, 프롬프트, 평가 결과는 **MLflow**를 통해 버저닝되고 중앙에서 관리됩니다.

## 3. 컴포넌트 1: 추론 엔드포인트 API (Inference Endpoint)

모델의 추론 결과를 실시간으로 제공하는 경량 API 서버입니다.

- **기술 스택:**
    - Web Framework: **FastAPI**
    - ASGI Server: **Uvicorn**
- **배포 (Deployment):**
    - `Dockerfile`을 통해 컨테이너 이미지로 빌드되어야 합니다.
    - `CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]` 형태로 Uvicorn을 통해 실행 가능해야 합니다.
- **주요 기능 (Features):**
    1. **R/R 엔드포인트:** `/predict` 또는 `/invoke` 엔드포인트를 통해 JSON 형식의 요청(Request)을 받고, 모델의 추론 결과(Response)를 JSON으로 반환합니다.
    2. **모델 로딩 (Model Loading):**
        - API 서버 시작 시, 설정된 `MLFLOW_MODEL_NAME`과 `MLFLOW_MODEL_STAGE` (예: "Production")를 기준으로 MLflow Model Registry에서 해당 모델 아티팩트를 다운로드하여 메모리에 로드합니다.
        - (참조: `컴포넌트 2: 모델 래퍼`)
    3. **피처 보강 (Feature Enrichment):**
        - 요청 본문(Request Body)에 포함된 `user_id` 등의 키를 사용하여, 추론 실행 *전*에 Feature Store 또는 기타 DB에서 필요한 피처를 조회(lookup)하는 로직을 포함할 수 있어야 합니다.
        - 이 로직은 API 핸들러 내에서 `ModelWrapper` 호출 전에 수행됩니다.
    4. **추상화된 추론 호출:** API는 로드된 `ModelWrapper` 객체의 `.predict(request_data)` 메서드만 호출합니다.

## 4. 컴포넌트 2: 모델 래퍼 및 관리 (Model Wrappers)

모든 모델(자체학습/LLM)의 생명주기(학습, 평가, 추론) 로직을 캡슐화하는 Python 클래스입니다.

### 4.1. 래퍼 A: 자체 학습 모델 (Self-Trained Model Wrapper)

- **대상:** PyTorch, Scikit-learn 등
- **클래스 (예시):** `PytorchLightningWrapper`
- **기능:**
    1. **`train(data_loader, config)`:**
        - **PyTorch Lightning**의 `Trainer`를 사용하여 모델 학습을 수행합니다.
        - **MLflow Logger** (`pytorch_lightning.loggers.MLflowLogger`)를 `Trainer`에 연결하여 모든 파라미터, 메트릭, 체크포인트를 MLflow Tracking Server에 **자동으로 로깅**해야 합니다.
    2. **`evaluate(data_loader, config)`:**
        - `trainer.test()` 또는 별도의 평가 로직을 수행합니다.
        - 최종 평가 결과를 MLflow에 명시적으로 로깅합니다.
    3. **`load(model_uri)`:**
        - `mlflow.pyfunc.load_model` 또는 `mlflow.pytorch.load_model`을 사용하여 MLflow URI로부터 모델 아티팩트(`model.pkl`, `weights.ckpt` 등)를 로드합니다.
    4. **`predict(input_data)`:**
        - 로드된 모델을 사용하여 추론(inference)을 수행합니다.

### 4.2. 래퍼 B: LLM 기반 모델 (LLM-Based Model Wrapper)

- **대상:** GPT, Claude 등 API 벤더 또는 LangGraph로 구성된 로직
- **클래스 (예시):** `LangGraphWrapper`
- **기능:**
    1. **`train(config)`:**
        - N/A (대부분의 경우 해당 없음).
    2. **`evaluate(data_loader, config)`:**
        - "Golden Dataset" (`data_loader`)를 입력받아 LLM API 또는 LangGraph를 실행합니다.
        - RAGAs, BLEU, 또는 커스텀 평가 로직을 통해 성능을 측정하고, 그 결과를 MLflow에 로깅합니다.
    3. **`load(model_uri)`:**
        - `model_uri` (MLflow)로부터 아티팩트(예: `langgraph.py`, `prompts.yaml`, `config.json`)를 다운로드합니다.
        - LangGraph 객체를 컴파일하거나, LLM API Client(예: `OpenAI`)를 초기화합니다.
    4. **`predict(input_data)`:**
        - 초기화된 LangGraph/Client를 `invoke` 또는 `call` 하여 추론을 수행합니다.

## 5. 컴포넌트 3: 데이터 로더 (Data Loader)

학습 및 평가 래퍼에 데이터를 공급하는 모듈입니다.

- **목적:** 다양한 데이터 소스로부터 데이터를 읽어, 모델 래퍼가 사용할 수 있는 표준 형식(예: `torch.utils.data.DataLoader`, `pandas.DataFrame`)으로 변환합니다.
- **요구사항:**
    - **모듈식 구현:** 데이터 소스별로 로더 클래스를 구현해야 합니다.
        - `LocalDataLoader` (로컬 디렉토리의 Parquet/CSV/JSON 파일 로드)
        - `FeatureStoreDataLoader` (Feast, Tecton 등 피처 스토어 API를 통한 온라인/오프라인 피처 조회)
        - (Future) `SQLDataLoader` (DB에서 직접 쿼리)
    - 설정 파일(`config.yaml`)을 통해 사용할 로더와 소스 경로를 동적으로 선택할 수 있어야 합니다.

## 6. 컴포넌트 4: 학습 파이프라인 (Training Pipeline - KFP)

자체 학습 모델(컴포넌트 4.1)의 학습/평가/등록 과정을 자동화하는 Kubeflow Pipelines (KFP)입니다.

- **유념사항:** 각 단계는 격리된 컨테이너(Pod)로 Kubeflow에서 실행 가능해야 합니다.
- **파이프라인 구성 (예시):**
    1. **Component 1: `get-data`**
        - `Data Loader` (컴포넌트 5)를 실행하여 학습/검증/테스트 데이터를 준비하고 KFP 아티팩트로 출력합니다.
    2. **Component 2: `train-model`**
        - `PytorchLightningWrapper.train()` (컴포넌트 4.1) 로직을 실행합니다.
        - 학습 데이터를 입력으로 받고, 완료된 MLflow `run_id`를 출력합니다.
    3. **Component 3: `evaluate-model`**
        - `PytorchLightningWrapper.evaluate()` 로직을 실행합니다.
        - `train-model`의 `run_id`와 테스트 데이터를 입력으로 받고, 평가 점수(예: `accuracy`)를 출력합니다.
    4. **Component 4: `register-model` (조건부)**
        - `evaluate-model`의 `accuracy`가 사전에 정의된 임계값(Threshold)을 넘는지 확인합니다.
        - 통과 시, `run_id`를 사용하여 MLflow Model Registry에 모델을 등록하고 "Staging" 스테이지로 승격시킵니다.

## 7. 환경 및 설정 관리 (Configuration Management)

모든 컴포넌트(FastAPI, KFP)의 동작은 외부 설정을 통해 제어되어야 합니다 (하드코딩 금지).

- **방식:** 환경 변수 (Environment Variables) 또는 마운트된 `config.yaml` 파일.
- **필수 설정 변수 (예시):**
    - `MLFLOW_TRACKING_URI`: (전체) MLflow 서버 접속 주소.
    - `MLFLOW_MODEL_NAME`: (API용) 서빙할 MLflow Registry 모델 이름.
    - `MLFLOW_MODEL_STAGE`: (API용) 서빙할 스테이지 (예: "Production").
    - `MLFLOW_EXPERIMENT_NAME`: (KFP용) 학습 결과를 로깅할 실험 이름.
    - `DATA_LOADER_TYPE`: (KFP용) 사용할 데이터 로더 (예: "local", "feature_store").
    - `DATA_SOURCE_PATH`: (KFP용) 데이터 소스 경로 (파일 경로, DB URI 등).
    - `OPENAI_API_KEY`: (LLM 래퍼용)