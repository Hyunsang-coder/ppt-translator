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

export function styleStatusLabel(status: string): string {
  switch (status) {
    case "preserved": return "보존됨";
    case "partial": return "일부 확인 필요";
    case "dropped": return "단색 대체";
    default: return "단일 서식";
  }
}
