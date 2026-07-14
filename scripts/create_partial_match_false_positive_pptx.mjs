import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import { mkdirSync } from "node:fs";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..");
const require = createRequire(join(repoRoot, "frontend", "package.json"));
const pptxgen = require("pptxgenjs");
const output = join(repoRoot, "tests", "fixtures", "partial-match-false-positive.pptx");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "PPT 번역캣 regression fixture";
pptx.subject = "부분 일치 문구 오탐 회귀 테스트";
pptx.title = "부분 일치 오탐 스트레스 테스트";
pptx.company = "PPT 번역캣";
pptx.lang = "ko-KR";
pptx.theme = {
  headFontFace: "Apple SD Gothic Neo",
  bodyFontFace: "Apple SD Gothic Neo",
  lang: "ko-KR",
};

const C = {
  ink: "172033",
  muted: "667085",
  bg: "F7F8FC",
  card: "FFFFFF",
  border: "D8DDEA",
  blue: "4F67D8",
  blueSoft: "E9EDFF",
  red: "D64545",
  redSoft: "FDECEC",
  green: "238A63",
  greenSoft: "E7F6F0",
};

function addHeader(slide, number, title, subtitle) {
  slide.background = { color: C.bg };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: 13.333,
    h: 0.12,
    line: { color: C.blue, transparency: 100 },
    fill: { color: C.blue },
  });
  slide.addText(`CASE ${number}`, {
    x: 0.55,
    y: 0.36,
    w: 1.15,
    h: 0.28,
    margin: 0,
    fontFace: "Aptos",
    fontSize: 11,
    bold: true,
    color: C.blue,
    charSpacing: 1.4,
  });
  slide.addText(title, {
    x: 0.55,
    y: 0.7,
    w: 8.9,
    h: 0.5,
    margin: 0,
    fontSize: 27,
    bold: true,
    color: C.ink,
  });
  slide.addText(subtitle, {
    x: 0.55,
    y: 1.22,
    w: 11.8,
    h: 0.34,
    margin: 0,
    fontSize: 12,
    color: C.muted,
  });
  slide.addText("PPT 번역캣 · partial-match regression fixture", {
    x: 0.55,
    y: 7.12,
    w: 5.5,
    h: 0.18,
    margin: 0,
    fontSize: 9,
    color: C.muted,
  });
}

function addScenario(slide, text, { x, y, w, h = 0.92, kind = "candidate", label }) {
  const palette = kind === "edit"
    ? { fill: C.blueSoft, line: C.blue, label: C.blue }
    : kind === "valid"
      ? { fill: C.greenSoft, line: C.green, label: C.green }
      : { fill: C.card, line: C.border, label: C.muted };
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.06,
    fill: { color: palette.fill },
    line: { color: palette.line, width: kind === "candidate" ? 1 : 1.5 },
  });
  slide.addText(label, {
    x: x + 0.18,
    y: y + 0.12,
    w: w - 0.36,
    h: 0.18,
    margin: 0,
    fontSize: 10,
    bold: true,
    color: palette.label,
  });
  slide.addText(text, {
    x: x + 0.18,
    y: y + 0.36,
    w: w - 0.36,
    h: h - 0.46,
    margin: 0,
    fontSize: 15,
    bold: kind === "edit",
    color: C.ink,
    valign: "mid",
  });
}

{
  const slide = pptx.addSlide();
  addHeader(
    slide,
    "01",
    "한 글자 CJK 변경은 전파하지 않는다",
    "활성화 → 활성총 수정에서 추출되는 ‘화 → 총’이 무관한 단어를 오염시키는지 검사"
  );
  addScenario(slide, "활성화 기준", { x: 0.55, y: 1.85, w: 5.95, kind: "edit", label: "수정 대상 · 활성화 기준 → 활성총 기준" });
  addScenario(slide, "3개월 개발 + 6주 안정화, 11월 중순 베타 테스트", { x: 6.78, y: 1.85, w: 5.95, label: "추천되면 안 됨" });
  addScenario(slide, "Phase 2 범위는 활성화, 명확성, 매치메이킹에 초점", { x: 0.55, y: 3.02, w: 5.95, label: "추천되면 안 됨" });
  addScenario(slide, "원인이 방향성 실패인지 활성화 실패인지가 핵심 질문", { x: 6.78, y: 3.02, w: 5.95, label: "추천되면 안 됨" });
  addScenario(slide, "전화 회의 결과를 문서화하고 시각화 자료를 정리", { x: 0.55, y: 4.19, w: 5.95, label: "추천되면 안 됨" });
  addScenario(slide, "명확화 작업과 안정화 작업은 서로 다른 일정으로 관리", { x: 6.78, y: 4.19, w: 5.95, label: "추천되면 안 됨" });
  addTextSummary(slide, "기대 결과", "부분 일치 후보 0건", 0.55, 5.55, C.redSoft, C.red);
}

function addTextSummary(slide, label, value, x, y, fill, accent) {
  slide.addShape(pptx.ShapeType.rect, {
    x,
    y,
    w: 12.18,
    h: 0.92,
    fill: { color: fill },
    line: { color: accent, width: 1 },
  });
  slide.addText(label, {
    x: x + 0.22,
    y: y + 0.16,
    w: 1.2,
    h: 0.22,
    margin: 0,
    fontSize: 10,
    bold: true,
    color: accent,
  });
  slide.addText(value, {
    x: x + 1.5,
    y: y + 0.14,
    w: 10.2,
    h: 0.34,
    margin: 0,
    fontSize: 18,
    bold: true,
    color: C.ink,
  });
}

{
  const slide = pptx.addSlide();
  addHeader(
    slide,
    "02",
    "영문은 단어 경계를 지킨다",
    "CAT → DOG 변경이 CONCAT 같은 더 긴 단어 내부에 적용되지 않는지 검사"
  );
  addScenario(slide, "CAT policy", { x: 0.55, y: 1.85, w: 5.95, kind: "edit", label: "수정 대상 · CAT policy → DOG policy" });
  addScenario(slide, "CONCAT function", { x: 6.78, y: 1.85, w: 5.95, label: "추천되면 안 됨 · 단어 내부" });
  addScenario(slide, "The CAT schedule", { x: 0.55, y: 3.02, w: 5.95, kind: "valid", label: "추천되어야 함 · 독립 단어" });
  addScenario(slide, "SCATTER plot", { x: 6.78, y: 3.02, w: 5.95, label: "추천되면 안 됨 · 단어 내부" });
  addScenario(slide, "CAT-based workflow", { x: 0.55, y: 4.19, w: 5.95, kind: "valid", label: "추천되어야 함 · 구두점 경계" });
  addScenario(slide, "Category policy", { x: 6.78, y: 4.19, w: 5.95, label: "추천되면 안 됨 · 대소문자/부분 문자열" });
  addTextSummary(slide, "기대 결과", "후보 2건 · The CAT schedule / CAT-based workflow", 0.55, 5.55, C.greenSoft, C.green);
}

{
  const slide = pptx.addSlide();
  addHeader(
    slide,
    "03",
    "삭제 미리보기와 실제 적용은 같아야 한다",
    "활성화 삭제가 후보 화면과 적용 결과에서 동일하며 불필요한 공백을 남기지 않는지 검사"
  );
  addScenario(slide, "Phase 2 활성화 기준", { x: 0.55, y: 1.85, w: 5.95, kind: "edit", label: "수정 대상 · Phase 2 기준으로 변경" });
  addScenario(slide, "활성화 일정", { x: 6.78, y: 1.85, w: 5.95, kind: "valid", label: "추천되어야 함 · 결과: 일정" });
  addScenario(slide, "다음 단계 활성화", { x: 0.55, y: 3.02, w: 5.95, kind: "valid", label: "추천되어야 함 · 결과: 다음 단계" });
  addScenario(slide, "비활성화 정책", { x: 6.78, y: 3.02, w: 5.95, label: "추천되면 안 됨 · 한글 단어 내부" });
  addScenario(slide, "활성화", { x: 0.55, y: 4.19, w: 5.95, kind: "valid", label: "추천되어야 함 · 결과: 빈 문자열" });
  addScenario(slide, "활성화 여부와 활성화 일정", { x: 6.78, y: 4.19, w: 5.95, label: "추천되면 안 됨 · 한 문장에 2회" });
  addTextSummary(slide, "기대 결과", "후보와 적용 결과 일치 · 중복 위치는 후보 제외", 0.55, 5.55, C.greenSoft, C.green);
}

{
  const slide = pptx.addSlide();
  addHeader(
    slide,
    "04",
    "정상 후보와 완전 일치 전파를 구분한다",
    "의미 있는 다단어 치환은 추천하되 동일 원문 자동 전파 대상은 중복 후보로 내지 않는지 검사"
  );
  addScenario(slide, "Adjusted field drop rates", { x: 0.55, y: 1.85, w: 5.95, kind: "edit", label: "수정 대상 · field drop → World Spawn" });
  addScenario(slide, "The field drop table", { x: 6.78, y: 1.85, w: 5.95, kind: "valid", label: "추천되어야 함" });
  addScenario(slide, "field drop follows field drop rules", { x: 0.55, y: 3.02, w: 5.95, label: "추천되면 안 됨 · 한 문장에 2회" });
  addScenario(slide, "Unrelated text", { x: 6.78, y: 3.02, w: 5.95, label: "추천되면 안 됨" });
  addScenario(slide, "Phase 2 활성화 기준", { x: 0.55, y: 4.19, w: 5.95, kind: "edit", label: "완전 일치 원문 A" });
  addScenario(slide, "Phase 2 활성화 기준", { x: 6.78, y: 4.19, w: 5.95, kind: "edit", label: "완전 일치 원문 B · 부분 후보 중복 금지" });
  addTextSummary(slide, "기대 결과", "정상 부분 후보 1건 · 완전 전파 대상의 중복 후보 0건", 0.55, 5.55, C.greenSoft, C.green);
}

mkdirSync(dirname(output), { recursive: true });
await pptx.writeFile({ fileName: output });
console.log(output);
