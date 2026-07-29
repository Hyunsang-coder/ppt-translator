import type { FragmentFinding } from "@/types/api";

/**
 * Badge colour and plain-language label for a finding. The queue speaks about
 * one item at a time, so the label reads as a statement about this item rather
 * than as a category name.
 */
export function findingBadge(finding: FragmentFinding): { cls: string; label: string } {
  switch (finding.type) {
    case "terminology.violation":
      return { cls: "text-destructive bg-destructive/10", label: "용어집과 다름" };
    case "terminology.inconsistency":
    case "consistency.phrase":
      return { cls: "text-info bg-info/10", label: "표현이 오락가락함" };
    case "accuracy.omission":
      return { cls: "text-destructive bg-destructive/10", label: "번역이 빠짐" };
    case "fit.overflow":
      return { cls: "text-warning bg-warning/10", label: "박스에 넘침" };
    case "fit.length_limit":
      return { cls: "text-warning bg-warning/10", label: "길이 초과" };
    case "style.mapping_dropped":
      return { cls: "text-warning bg-warning/10", label: "색상 확인 필요" };
    default:
      return { cls: "text-muted-foreground bg-muted", label: finding.type };
  }
}

/**
 * What a style preview is actually showing. A fixed "색상 미리보기" label reads
 * as broken on a paragraph that has no colour: the dropped case is not a colour
 * preview at all, it is a warning that the original emphasis did not survive.
 */
export function stylePreviewNote(status: string): string | null {
  switch (status) {
    case "preserved": return "원문 색상 그대로";
    case "partial": return "원문 색상 일부만 확인됨";
    case "dropped": return "원문 강조를 잃고 첫 서식으로 통일됨";
    default: return null;
  }
}

export function styleStatusLabel(status: string): string {
  switch (status) {
    case "preserved": return "보존됨";
    case "partial": return "일부 확인 필요";
    case "dropped": return "단색 대체";
    default: return "단일 서식";
  }
}
