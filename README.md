# PPT Translator

Streamlit 기반 PowerPoint 번역 프로토타입으로, LangChain과 OpenAI GPT-5 계열 모델을 활용해 슬라이드 텍스트를 고품질로 번역합니다. 원본 서식을 최대한 유지하면서 용어집, 자동 언어 감지, 상세 진행률 표시를 지원합니다.

## 주요 기능
- PPT/PPTX 업로드(최대 50MB) 및 메모리 내부 처리
- 소스/타겟 언어 자동 감지 및 추론 (`langdetect`)
- GPT-5, GPT-5-mini 두 모델만 사용하며 temperature 파라미터 미사용
- Excel 용어집 업로드 + 템플릿 다운로드 + 번역 전/후 용어 치환
- LangChain 기반 순차 배치 번역 (진행률 바 및 ETA 표시)
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
앱이 브라우저에서 열리면 PPT 파일을 업로드하고 번역 설정을 조정한 뒤 `🚀 번역 시작` 버튼을 눌러주세요.

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
- **LangChain 체인 구성**: `ChatOpenAI(model=model_name)`을 temperature 없이 초기화하고, PPT 컨텍스트/용어집/사용자 프롬프트를 주입한 뒤 번역 결과를 `|||` 구분자로 반환하도록 유도합니다.
- **배치 처리 흐름**: `helpers.chunk_paragraphs`가 문단을 순차 배치로 나누고, `translate_with_progress`가 각 배치를 재시도 로직과 함께 순차 호출합니다. LangChain의 `.batch()`는 사용하지 않습니다. 전체 문단 수 대비 약 5개 배치가 되도록 동적으로 크기를 조정해 진행률이 자주 업데이트되도록 했습니다.
- **진행률 표시**: `ProgressTracker`가 Streamlit progress bar와 상태 텍스트를 업데이트하며, 경과 시간과 ETA를 함께 제공합니다.
- **언어 감지 및 추론**: `LanguageDetector`가 langdetect 결과를 UI용 언어명으로 변환하고, 한국어/영어 우선 규칙으로 타겟 언어를 자동 추론합니다.
- **서식 유지 전략**: `PPTWriter`가 기존 run 길이를 기반으로 번역 텍스트를 분배해 가능한 한 원본 서식을 유지합니다.

## 테스트 방법
- 빠른 유닛 테스트는 표준 라이브러리 `unittest`로 제공됩니다.
  ```bash
  python -m unittest tests/test_translation.py
  ```
- 샘플 PPT로 번역을 검증하려면 간단한 텍스트가 포함된 PPT를 직접 업로드하여 결과와 용어집 적용 여부를 확인하세요.
- 언어 자동 감지는 다양한 언어 텍스트를 슬라이드에 포함하여 감지 메시지를 확인하면 됩니다.

## 알려진 제한 사항
- 이미지 최적화는 현재 플레이스홀더이며 추후 구현 예정입니다.
- 매우 복잡한 서식(문단 내 다수의 색상/강조)은 완벽히 재현되지 않을 수 있습니다.
