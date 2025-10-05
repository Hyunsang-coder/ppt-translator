# PPT 번역캣

Streamlit 기반 PowerPoint 번역 프로토타입으로, LangChain과 OpenAI GPT-5 계열 모델을 활용해 슬라이드 텍스트를 고품질로 번역합니다. 원본 서식을 최대한 유지하면서 용어집, 자동 언어 감지, 상세 진행률 표시를 지원하며, "PPT 번역캣" 테마 UI를 제공합니다.

## 주요 기능
- 사이드바에서 `PPT 번역`과 `텍스트 추출` 워크플로를 전환하며, 번역캣 시그니처 이미지와 타이틀을 제공
- PPT/PPTX 업로드(최대 200MB) 후 메모리에서 직접 처리해 원본 파일을 보존
- 반복 문구 사전 처리 옵션이 기본 활성화되어 중복 문장은 한 번만 번역하고 결과를 재사용
- 소스/타겟 언어 자동 감지 및 추론 (`langdetect`)과 사용자 지정 프롬프트 지원
- Excel 용어집 업로드 + 템플릿 다운로드 + 번역 전/후 용어 치환
- LangChain 기반 동적 배치/동시성 제어와 진행률·로그 패널 제공
- 번역된 PPTX 즉시 다운로드 및 임시 저장 최소화

## 설치 방법
1. Python 3.10 이상을 준비합니다.
2. 저장소 클론 후 디렉터리로 이동합니다.
3. 가상 환경을 만들고 활성화합니다(선택).
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
4. 요구 사항을 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```
5. `.env.example`을 복사해 `.env`를 만들고 OpenAI API 키를 설정합니다.
   ```bash
   cp .env.example .env
   # .env 파일에서 OPENAI_API_KEY 값을 입력
   ```

## 실행 방법
```bash
streamlit run app.py
```
앱이 브라우저에서 열리면 사이드바 상단의 번역캣 이미지를 확인하고, PPT 파일을 업로드한 뒤 `🚀 번역 시작` 버튼을 눌러주세요.

## 용어집 파일 형식
- `glossary_template.xlsx` 템플릿을 다운로드하여 사용합니다.
- A열: 원문, B열: 번역 (예: PUBG → 배틀그라운드)
- 헤더를 포함한 단일 시트 구조만 지원합니다.
- 업로드 시 용어집은 프롬프트에 포함되며, 번역 결과에도 강제로 치환됩니다.

## 언어 자동 감지 동작
- 업로드된 PPT의 상위 50개 문단을 샘플로 추출합니다.
- `langdetect`로 소스 언어를 감지하여 UI에 표시합니다.
- 타겟 언어가 `Auto`일 경우 한국어↔영어를 우선으로 자동 설정합니다.
- 감지에 실패하면 기본값으로 영어를 사용하며, 사용자에게 직접 선택을 권장합니다.

## 핵심 로직 설명
- **LangChain 체인 구성**: `ChatOpenAI(model=model_name)`을 temperature 없이 초기화하고, PPT 컨텍스트/용어집/사용자 프롬프트를 주입한 뒤 번역 결과를 JSON 배열로 반환하도록 유도합니다. 파싱이 실패하면 `|||` 또는 줄바꿈 기반 분할로 폴백합니다.
- **배치 처리 흐름**: `helpers.chunk_paragraphs`가 문단을 순차 배치로 나누고, `translate_with_progress`가 재시도 로직과 함께 배치를 처리합니다. 설정값(`TRANSLATION_MAX_CONCURRENCY`, `TRANSLATION_BATCH_SIZE`, `TRANSLATION_WAVE_MULTIPLIER` 등)에 따라 동시 실행 수와 배치 크기를 동적으로 조정합니다.
- **진행률 및 로그**: `ProgressTracker`가 Streamlit progress bar와 상태 텍스트를 갱신하고, 로그 큐를 비워 UI 패널을 최신 상태로 유지합니다.
- **언어 감지 및 추론**: `LanguageDetector`가 langdetect 결과를 UI용 언어명으로 변환하고, 한국어/영어 우선 규칙으로 타겟 언어를 자동 추론합니다.
- **서식 유지 전략**: `PPTWriter`가 기존 run 길이를 기반으로 번역 텍스트를 분배해 가능한 한 원본 서식을 유지합니다.

## 테스트 방법
- 빠른 유닛 테스트는 표준 라이브러리 `unittest`로 제공됩니다.
  ```bash
  python -m unittest tests/test_translation.py
  ```
- 샘플 PPT로 번역을 검증하려면 간단한 텍스트가 포함된 PPT를 직접 업로드하여 결과와 용어집 적용 여부를 확인하세요.
- 언어 자동 감지는 다양한 언어 텍스트를 슬라이드에 포함하여 감지 메시지를 확인하면 됩니다.
