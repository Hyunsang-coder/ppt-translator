Patch notes 데이터를 최신 커밋 기반으로 업데이트합니다.

## 절차

1. `frontend/src/data/patch-notes.ts` 파일을 읽어 현재 패치 노트 데이터 확인
2. 마지막 패치 노트의 commitHash를 기준으로 `git log --oneline` 실행하여 이후 커밋 수집
3. 수집된 커밋 메시지를 분석하여 변경사항 분류:
   - **feature**: "Add", "Implement", "Support" 등으로 시작하는 새 기능
   - **fix**: "Fix", "Resolve", "Patch" 등으로 시작하는 버그 수정
   - **improvement**: 그 외 개선, 리팩토링, 성능 향상 등
4. 다음 커밋은 스킵:
   - CLAUDE.md 업데이트만 포함된 커밋
   - .gitignore 변경만 포함된 커밋
   - IP 주소/설정 변경만 포함된 커밋
   - Merge 커밋
   - 문서만 수정한 커밋 (README, docs 등)
5. 새로운 `PatchNote` 엔트리를 patchNotes 배열 맨 앞에 추가:
   - version: 오늘 날짜 (YYYY.MM.DD 형식). 같은 날짜가 이미 있으면 뒤에 `.2`, `.3` 등 suffix 추가
   - date: 오늘 날짜 (YYYY-MM-DD)
   - title: 주요 변경사항을 대표하는 한국어 제목
   - commitHash: 가장 최근 커밋의 short hash
   - changes: 분류된 변경사항 목록 (한국어 설명)

## 규칙

- 자동 커밋 금지. 변경사항을 사용자에게 보여주고 리뷰 후 저장
- description은 간결한 한국어로 작성 (한 줄)
- 유사한 변경사항은 하나로 합쳐도 됨
- feature → improvement → fix 순서로 정렬
