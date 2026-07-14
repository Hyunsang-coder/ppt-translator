"use client";

import { useCallback, useEffect, useMemo, useState, type CSSProperties } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { apiClient } from "@/lib/api-client";
import type {
  FragmentItem,
  FragmentFinding,
  FragmentProposalResponse,
  PartialCandidate,
  StyleSegment,
} from "@/types/api";
import {
  X,
  Pencil,
  RefreshCw,
  Ban,
  Check,
  Download,
  Loader2,
  AlertTriangle,
  Undo2,
} from "lucide-react";
import { toast } from "sonner";

// Finding type -> badge style. Mirrors the LOG_TYPE_STYLES lookup pattern.
function badgeStyle(finding: FragmentFinding): { cls: string; label: string } {
  switch (finding.type) {
    case "terminology.violation":
      return { cls: "text-destructive bg-destructive/10", label: "용어집 위반" };
    case "terminology.inconsistency":
    case "consistency.phrase":
      return { cls: "text-info bg-info/10", label: "일관성" };
    case "accuracy.omission":
      return { cls: "text-destructive bg-destructive/10", label: "미번역" };
    case "fit.overflow":
      return { cls: "text-warning bg-warning/10", label: "공간 초과" };
    case "style.mapping_dropped":
      return { cls: "text-warning bg-warning/10", label: "색상 확인" };
    default:
      return { cls: "text-muted-foreground bg-muted", label: finding.type };
  }
}

interface ReviewPanelProps {
  jobId: string;
  onClose: () => void;
  onDownload: () => void;
}

type CardSpan = "short" | "med" | "long";

// 번역 길이로 타일이 차지할 그리드 폭을 정한다. 짧은 조각은 1칸, 중간은 2칸,
// 긴 문단·노트·편집 중인 조각은 전체 폭. 이렇게 해야 짧은 조각이 가로로
// 여러 개 채워져 스크롤이 짧아진다.
function cardSpan(frag: FragmentItem, editing: boolean): CardSpan {
  if (editing || frag.is_note) return "long";
  const n = frag.target.length;
  if (n > 80) return "long";
  if (n > 26) return "med";
  return "short";
}

function isOverflow(frag: FragmentItem): boolean {
  return (
    frag.length_budget !== null &&
    !frag.is_note &&
    frag.target.length > frag.length_budget
  );
}

type SlideGroup = {
  slide: number;
  title: string | null;
  items: FragmentItem[];
  flagged: number;
  edited: number;
};

export function ReviewPanel({ jobId, onClose, onDownload }: ReviewPanelProps) {
  const [fragments, setFragments] = useState<FragmentItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeSlide, setActiveSlide] = useState<number | null>(null);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editText, setEditText] = useState("");
  const [propagate, setPropagate] = useState(true);
  const [busyIndex, setBusyIndex] = useState<number | null>(null);
  const [revision, setRevision] = useState(0);
  const [committedRevision, setCommittedRevision] = useState(0);
  const [dirty, setDirty] = useState(false);
  const [proposal, setProposal] = useState<FragmentProposalResponse | null>(null);
  const [partialCandidates, setPartialCandidates] = useState<PartialCandidate[]>([]);
  const [selectedPartial, setSelectedPartial] = useState<Set<number>>(new Set());
  const [applyingPartial, setApplyingPartial] = useState(false);
  const [committing, setCommitting] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await apiClient.getJobFragments(jobId);
      setFragments(resp.fragments);
      setRevision(resp.revision);
      setCommittedRevision(resp.committed_revision);
      setDirty(resp.dirty);
      // 첫 로드 시 첫 슬라이드를 자동 선택.
      setActiveSlide((cur) =>
        cur !== null ? cur : (resp.fragments[0]?.slide ?? null)
      );
    } catch {
      setError("섹션 목록을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [jobId]);

  useEffect(() => {
    load();
  }, [load]);

  const flaggedCount = useMemo(
    () => fragments.filter((f) => f.findings.length > 0).length,
    [fragments]
  );
  const editedCount = useMemo(
    () => fragments.filter((f) => f.edited).length,
    [fragments]
  );

  // 슬라이드 단위로 그룹핑 — 사이드바 네비 + 본문 렌더 양쪽에서 쓴다.
  const slideGroups = useMemo<SlideGroup[]>(() => {
    const groups: SlideGroup[] = [];
    for (const f of fragments) {
      const last = groups[groups.length - 1];
      if (last && last.slide === f.slide) {
        last.items.push(f);
        if (f.findings.length > 0) last.flagged += 1;
        if (f.edited) last.edited += 1;
      } else {
        groups.push({
          slide: f.slide,
          title: f.slide_title,
          items: [f],
          flagged: f.findings.length > 0 ? 1 : 0,
          edited: f.edited ? 1 : 0,
        });
      }
    }
    return groups;
  }, [fragments]);

  const activeGroup = useMemo(
    () => slideGroups.find((g) => g.slide === activeSlide) ?? null,
    [slideGroups, activeSlide]
  );
  const proposalFragment = useMemo(
    () => fragments.find((fragment) => fragment.index === proposal?.index) ?? null,
    [fragments, proposal]
  );

  const startEdit = (frag: FragmentItem) => {
    setEditingIndex(frag.index);
    setEditText(frag.target);
  };

  const previewEdit = async (frag: FragmentItem) => {
    setBusyIndex(frag.index);
    try {
      const resp = await apiClient.proposeJobFragment(jobId, frag.index, {
        action: "edit",
        target: editText,
        propagate_identical: propagate,
      });
      setProposal(resp);
    } catch {
      toast.error("수정 미리보기를 만들지 못했습니다.");
    } finally {
      setBusyIndex(null);
    }
  };

  const retranslate = async (frag: FragmentItem, instruction?: string) => {
    setBusyIndex(frag.index);
    try {
      const resp = await apiClient.proposeJobFragment(jobId, frag.index, {
        action: "retranslate",
        instruction,
        propagate_identical: propagate,
      });
      setProposal(resp);
    } catch {
      toast.error("재번역에 실패했습니다.");
    } finally {
      setBusyIndex(null);
    }
  };

  const applyProposal = async () => {
    if (!proposal) return;
    setBusyIndex(proposal.index);
    try {
      const resp = await apiClient.applyJobFragmentProposal(
        jobId,
        proposal.proposal_id,
        revision
      );
      setRevision(resp.revision);
      setDirty(resp.dirty);
      setPartialCandidates(resp.partial_candidates);
      // 부분 일치는 문맥 검토가 필요한 보조 후보이므로 사용자가 직접 고른다.
      setSelectedPartial(new Set());
      setProposal(null);
      setEditingIndex(null);
      await load();
      toast.success(
        resp.changed_indices.length > 1
          ? `초안 ${resp.changed_indices.length}곳에 반영했습니다.`
          : "검토 초안에 반영했습니다."
      );
    } catch {
      toast.error("후보 적용에 실패했습니다. 목록을 새로 확인해주세요.");
      await load();
    } finally {
      setBusyIndex(null);
    }
  };

  const applySelectedPartial = async () => {
    if (
      applyingPartial ||
      selectedPartial.size === 0 ||
      partialCandidates.length === 0
    ) return;
    const first = partialCandidates[0];
    setApplyingPartial(true);
    try {
      await apiClient.applyPartialCandidates(jobId, {
        indices: Array.from(selectedPartial),
        old_phrase: first.old_phrase,
        new_phrase: first.new_phrase,
        expected_revision: revision,
      });
      setPartialCandidates([]);
      setSelectedPartial(new Set());
      await load();
      toast.success("선택한 부분 일치 문구를 초안에 반영했습니다.");
    } catch {
      toast.error("부분 일치 문구 적용에 실패했습니다.");
      await load();
    } finally {
      setApplyingPartial(false);
    }
  };

  const undo = async () => {
    try {
      await apiClient.undoReview(jobId, revision);
      setProposal(null);
      setPartialCandidates([]);
      await load();
      toast.success("마지막 초안 수정을 되돌렸습니다.");
    } catch {
      toast.error("되돌리기에 실패했습니다.");
    }
  };

  const commitAndDownload = async () => {
    setCommitting(true);
    try {
      const resp = await apiClient.commitReview(jobId, revision);
      setCommittedRevision(resp.committed_revision);
      setDirty(resp.dirty);
      await load();
      await onDownload();
      toast.success("검토 초안을 최종 PPT에 반영했습니다.");
    } catch {
      toast.error("최종 반영에 실패했습니다. 기존 결과 파일은 유지됩니다.");
    } finally {
      setCommitting(false);
    }
  };

  const ignore = async (frag: FragmentItem) => {
    setBusyIndex(frag.index);
    try {
      await apiClient.editJobFragment(jobId, frag.index, {
        action: "ignore",
        finding_type: frag.findings[0]?.type,
      });
      // 무시는 검출만 지운다 — 해당 조각의 findings를 비운다.
      setFragments((prev) =>
        prev.map((f) => (f.index === frag.index ? { ...f, findings: [] } : f))
      );
      toast.success("검출을 무시했습니다.");
    } catch {
      toast.error("처리에 실패했습니다.");
    } finally {
      setBusyIndex(null);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-background/95 backdrop-blur-sm flex flex-col">
      {/* header */}
      <div className="flex items-center justify-between gap-3 px-4 py-2.5 border-b bg-background/95 flex-wrap">
        <div>
          <h2 className="text-base font-bold leading-tight">번역 검토 &amp; 수정</h2>
          <p className="text-xs text-muted-foreground">
            {fragments.length}개 섹션 · 검출 {flaggedCount} · 수정됨 {editedCount}
            {dirty && ` · 최종 r${committedRevision} → 초안 r${revision}`}
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={undo}
            disabled={!dirty || committing}
            className="gap-2"
          >
            <Undo2 className="w-4 h-4" />
            되돌리기
          </Button>
          <Button size="sm" onClick={commitAndDownload} disabled={committing} className="gap-2">
            {committing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
            {dirty ? "최종 반영 후 저장" : "저장"}
          </Button>
          <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="닫기">
            <X className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {loading && (
        <div className="flex items-center justify-center flex-1 text-muted-foreground gap-2">
          <Loader2 className="w-5 h-5 animate-spin" />
          섹션을 불러오는 중...
        </div>
      )}

      {error && (
        <div className="m-4 p-3 rounded-lg border border-destructive/30 bg-destructive/10 text-sm text-destructive flex items-center gap-2">
          <AlertTriangle className="w-4 h-4" />
          {error}
        </div>
      )}

      {!loading && !error && (
        <div className="flex flex-1 min-h-0">
          {/* 사이드바: 슬라이드 목록 */}
          <nav className="w-40 shrink-0 border-r overflow-y-auto py-2">
            {slideGroups.map((g) => {
              const active = g.slide === activeSlide;
              return (
                <button
                  key={g.slide}
                  onClick={() => setActiveSlide(g.slide)}
                  className={`w-full text-left px-3 py-2 border-l-2 transition-colors ${
                    active
                      ? "border-primary bg-primary/10"
                      : "border-transparent hover:bg-muted/50"
                  }`}
                >
                  <div className="flex items-center gap-1.5">
                    <span
                      className={`text-xs font-bold ${
                        active ? "text-primary" : "text-muted-foreground"
                      }`}
                    >
                      S{g.slide}
                    </span>
                    {g.flagged > 0 && (
                      <span className="text-[10px] text-warning">⚠️{g.flagged}</span>
                    )}
                    {g.edited > 0 && (
                      <span className="text-[10px] text-success">✎{g.edited}</span>
                    )}
                  </div>
                  {g.title && (
                    <div className="text-[11px] text-muted-foreground truncate mt-0.5">
                      {g.title}
                    </div>
                  )}
                  <div className="text-[10px] text-muted-foreground/70">
                    섹션 {g.items.length}
                  </div>
                </button>
              );
            })}
          </nav>

          {/* 본문: 선택된 슬라이드의 조각만 — 반응형 그리드 (2~4열, 짝수) */}
          <div className="flex-1 overflow-y-auto px-4 py-3">
            {activeGroup && (
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xs font-bold text-primary bg-primary/10 rounded px-2 py-1">
                  S{activeGroup.slide}
                </span>
                {activeGroup.title && (
                  <h3 className="text-sm font-bold truncate">{activeGroup.title}</h3>
                )}
                <span className="text-xs text-muted-foreground">
                  섹션 {activeGroup.items.length}개
                </span>
              </div>
            )}

            <div className="review-grid">
              {activeGroup?.items.map((frag) => (
                <FragmentCard
                  key={frag.index}
                  frag={frag}
                  span={cardSpan(frag, editingIndex === frag.index)}
                  editing={editingIndex === frag.index}
                  busy={busyIndex === frag.index}
                  editText={editText}
                  propagate={propagate}
                  onEditTextChange={setEditText}
                  onPropagateChange={setPropagate}
                  onStartEdit={() => startEdit(frag)}
                  onCancelEdit={() => setEditingIndex(null)}
                  onPreviewManual={() => previewEdit(frag)}
                  onRetranslate={(instruction) => retranslate(frag, instruction)}
                  onIgnore={() => ignore(frag)}
                />
              ))}
            </div>

            {activeGroup && activeGroup.items.length === 0 && (
              <div className="text-center py-16 text-muted-foreground text-sm">
                이 슬라이드에 섹션이 없습니다.
              </div>
            )}
          </div>
        </div>
      )}

      {proposal && (
        <div className="absolute inset-0 z-20 bg-background/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-2xl rounded-xl border bg-card shadow-xl p-4 space-y-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h3 className="font-bold">수정 후보 비교</h3>
                <p className="text-xs text-muted-foreground">
                  색상과 강조를 확인한 뒤 초안에 적용하세요.
                </p>
              </div>
              <Button variant="ghost" size="icon-sm" onClick={() => setProposal(null)}>
                <X className="w-4 h-4" />
              </Button>
            </div>
            <div className="rounded-lg border bg-muted/30 p-3">
              <p className="text-[11px] font-bold text-muted-foreground mb-1">원문</p>
              <p className="text-sm leading-snug">{proposalFragment?.source ?? ""}</p>
            </div>
            <div className="grid sm:grid-cols-2 gap-3">
              <div className="rounded-lg bg-muted/50 p-3">
                <p className="text-[11px] font-bold text-muted-foreground mb-1">현재 번역</p>
                <p className="text-sm">{proposal.old_target}</p>
              </div>
              <div className="rounded-lg border border-primary/30 bg-primary/5 p-3">
                <p className="text-[11px] font-bold text-muted-foreground mb-1">수정 후보 · 색상 미리보기</p>
                <StyledText segments={proposal.style_segments} fallback={proposal.target} />
              </div>
            </div>
            <div className="flex flex-wrap gap-2 text-xs">
              <span className="rounded-full bg-muted px-2 py-1">
                서식 {styleStatusLabel(proposal.style_status)}
              </span>
              {proposal.changed_indices.length > 1 && (
                <span className="rounded-full bg-info/10 text-info px-2 py-1">
                  동일 문구 {proposal.changed_indices.length}곳
                </span>
              )}
              {proposal.over_budget && (
                <span className="rounded-full bg-destructive/10 text-destructive px-2 py-1">
                  예상 박스 용량 초과
                </span>
              )}
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setProposal(null)}>취소</Button>
              <Button onClick={applyProposal} disabled={busyIndex === proposal.index}>
                {busyIndex === proposal.index && <Loader2 className="w-4 h-4 animate-spin mr-2" />}
                적용
              </Button>
            </div>
          </div>
        </div>
      )}

      {partialCandidates.length > 0 && !proposal && (
        <div className="absolute bottom-4 left-1/2 z-10 w-[min(680px,calc(100%-2rem))] -translate-x-1/2 rounded-xl border bg-card shadow-xl p-4">
          <div className="flex items-start justify-between gap-3 mb-3">
            <div>
              <h3 className="text-sm font-bold">부분 일치 문구도 변경할까요?</h3>
              <p className="text-xs text-muted-foreground">
                문장 구조가 다른 위치는 원문을 확인하고 필요한 항목만 선택하세요.
              </p>
              {partialCandidates[0] && (
                <p className="mt-1 text-xs font-medium">
                  &ldquo;{partialCandidates[0].old_phrase}&rdquo; → &ldquo;
                  {partialCandidates[0].new_phrase || "(삭제)"}&rdquo;
                </p>
              )}
            </div>
            <Button
              variant="ghost"
              size="icon-sm"
              disabled={applyingPartial}
              onClick={() => setPartialCandidates([])}
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
          <div className="max-h-48 overflow-y-auto space-y-2">
            {partialCandidates.map((candidate) => (
              <label key={candidate.index} className="flex gap-2 rounded-lg bg-muted/50 p-2 text-xs">
                <input
                  type="checkbox"
                  checked={selectedPartial.has(candidate.index)}
                  disabled={applyingPartial}
                  onChange={(event) => {
                    setSelectedPartial((current) => {
                      const next = new Set(current);
                      if (event.target.checked) next.add(candidate.index);
                      else next.delete(candidate.index);
                      return next;
                    });
                  }}
                />
                <span className="min-w-0 space-y-0.5">
                  <span className="block text-muted-foreground">
                    <b className="text-foreground">S{candidate.slide}</b>
                    {candidate.is_note && " · 발표자 노트"} · 원문: {candidate.source}
                  </span>
                  <span className="block">{candidate.target}</span>
                  <span className="block text-primary">→ {candidate.proposed_target}</span>
                </span>
              </label>
            ))}
          </div>
          <div className="flex justify-end gap-2 mt-3">
            <Button
              variant="outline"
              size="sm"
              disabled={applyingPartial}
              onClick={() => setPartialCandidates([])}
            >
              건너뛰기
            </Button>
            <Button
              size="sm"
              onClick={applySelectedPartial}
              disabled={applyingPartial || selectedPartial.size === 0}
            >
              {applyingPartial && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {applyingPartial ? "적용 중" : `선택한 ${selectedPartial.size}건 적용`}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function styleStatusLabel(status: string): string {
  switch (status) {
    case "preserved": return "보존됨";
    case "partial": return "일부 확인 필요";
    case "dropped": return "단색 대체";
    default: return "단일 서식";
  }
}

const LIGHT_REVIEW_BACKGROUND_LUMINANCE = 0.98;
const DARK_REVIEW_BACKGROUND_LUMINANCE = 0.03;
const MIN_REVIEW_TEXT_CONTRAST = 4.5;

function relativeLuminance(color: string): number | null {
  const match = color.match(/^#([0-9a-f]{6})$/i);
  if (!match) return null;

  const channels = match[1].match(/.{2}/g)?.map((value) => {
    const srgb = Number.parseInt(value, 16) / 255;
    return srgb <= 0.04045
      ? srgb / 12.92
      : ((srgb + 0.055) / 1.055) ** 2.4;
  });
  if (!channels || channels.length !== 3) return null;

  return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
}

function contrastRatio(first: number, second: number): number {
  const lighter = Math.max(first, second);
  const darker = Math.min(first, second);
  return (lighter + 0.05) / (darker + 0.05);
}

function reviewColorContrast(color: string | null): {
  lowOnLight: boolean;
  lowOnDark: boolean;
} {
  if (!color) return { lowOnLight: false, lowOnDark: false };
  const luminance = relativeLuminance(color);
  if (luminance === null) return { lowOnLight: false, lowOnDark: false };

  return {
    lowOnLight:
      contrastRatio(luminance, LIGHT_REVIEW_BACKGROUND_LUMINANCE) <
      MIN_REVIEW_TEXT_CONTRAST,
    lowOnDark:
      contrastRatio(luminance, DARK_REVIEW_BACKGROUND_LUMINANCE) <
      MIN_REVIEW_TEXT_CONTRAST,
  };
}

function StyledText({ segments, fallback }: { segments: StyleSegment[]; fallback: string }) {
  if (segments.length === 0) return <span className="text-sm">{fallback}</span>;
  return (
    <span className="text-sm leading-snug">
      {segments.map((segment, index) => {
        const contrast = reviewColorContrast(segment.color);
        return (
          <span
            key={`${index}-${segment.group_index}`}
            className="review-style-color"
            data-low-contrast-light={contrast.lowOnLight || undefined}
            data-low-contrast-dark={contrast.lowOnDark || undefined}
            style={{
              "--review-original-color": segment.color ?? "var(--foreground)",
              fontWeight: segment.bold ? 700 : undefined,
              fontStyle: segment.italic ? "italic" : undefined,
            } as CSSProperties}
            title={segment.color ?? (segment.scheme ? `테마 색상: ${segment.scheme}` : undefined)}
          >
            {segment.text}
          </span>
        );
      })}
    </span>
  );
}

interface FragmentCardProps {
  frag: FragmentItem;
  span: CardSpan;
  editing: boolean;
  busy: boolean;
  editText: string;
  propagate: boolean;
  onEditTextChange: (v: string) => void;
  onPropagateChange: (v: boolean) => void;
  onStartEdit: () => void;
  onCancelEdit: () => void;
  onPreviewManual: () => void;
  onRetranslate: (instruction?: string) => void;
  onIgnore: () => void;
}

function FragmentCard({
  frag,
  span,
  editing,
  busy,
  editText,
  propagate,
  onEditTextChange,
  onPropagateChange,
  onStartEdit,
  onCancelEdit,
  onPreviewManual,
  onRetranslate,
  onIgnore,
}: FragmentCardProps) {
  const overflow = isOverflow(frag);
  // long/편집 = 원문|번역 2단 대조. short/med = 번역 크게 + 원문 흐리게.
  const twoCol = span === "long" || editing;

  const [editMode, setEditMode] = useState<"manual" | "ai">("manual");
  const [instruct, setInstruct] = useState("");

  useEffect(() => {
    if (!editing) {
      setEditMode("manual");
      setInstruct("");
    }
  }, [editing]);

  const submitRetranslate = () => {
    const trimmed = instruct.trim();
    onRetranslate(trimmed || (overflow ? "더 짧게" : undefined));
  };

  return (
    <div
      className={`group relative glass-card rounded-lg border p-2.5 transition-colors hover:border-primary ${
        span === "long"
          ? "review-span-full"
          : span === "med"
          ? "review-span-2"
          : ""
      }`}
    >
      {/* hover action panel */}
      {!editing && (
        <div className="absolute top-1 right-1.5 hidden group-hover:flex gap-0.5 bg-card border rounded-lg p-0.5 shadow-md z-10">
          <Button size="xs" variant="ghost" className="gap-1 h-6 px-1.5" onClick={onStartEdit}>
            <Pencil className="w-3 h-3" />
            수정
          </Button>
          {frag.findings.length > 0 && (
            <Button
              size="xs"
              variant="ghost"
              className="gap-1 h-6 px-1.5 text-muted-foreground"
              disabled={busy}
              onClick={onIgnore}
            >
              <Ban className="w-3 h-3" />
              무시
            </Button>
          )}
        </div>
      )}

      <div className="flex items-center gap-1.5 mb-1 text-[10px] font-semibold text-muted-foreground flex-wrap">
        <span>{frag.is_note ? "발표자 노트" : `섹션 ${frag.shape}`}</span>
        {frag.repeat_count > 1 && (
          <span className="bg-muted rounded-full px-1.5 py-0.5">×{frag.repeat_count}</span>
        )}
        {frag.edited && (
          <span className="text-success bg-success/10 rounded-full px-1.5 py-0.5">수정됨</span>
        )}
        {frag.style_status !== "single_style" && (
          <span className={`rounded-full px-1.5 py-0.5 ${
            frag.style_status === "preserved"
              ? "text-success bg-success/10"
              : "text-warning bg-warning/10"
          }`}>
            색상 {styleStatusLabel(frag.style_status)}
          </span>
        )}
        {frag.findings.map((finding, i) => {
          const s = badgeStyle(finding);
          return (
            <span key={i} className={`rounded-full px-1.5 py-0.5 ${s.cls}`}>
              {s.label}
            </span>
          );
        })}
      </div>

      {twoCol ? (
        // 2단 대조 (긴 문단·편집 중)
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            {editing && (
              <p className="text-[11px] font-bold text-muted-foreground mb-1">원문</p>
            )}
            <div className="text-sm text-foreground/70 leading-snug">{frag.source}</div>
          </div>
          {editing ? (
            <div className="min-w-0">
              <div className="flex gap-1 rounded-lg bg-muted p-1 mb-2" role="tablist" aria-label="수정 방식">
                <button
                  type="button"
                  role="tab"
                  aria-selected={editMode === "manual"}
                  disabled={busy}
                  onClick={() => setEditMode("manual")}
                  className={`flex-1 rounded-md px-2 py-1 text-xs font-medium transition-colors disabled:opacity-50 ${
                    editMode === "manual"
                      ? "bg-card text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  직접 수정
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={editMode === "ai"}
                  disabled={busy}
                  onClick={() => setEditMode("ai")}
                  className={`flex-1 rounded-md px-2 py-1 text-xs font-medium transition-colors disabled:opacity-50 ${
                    editMode === "ai"
                      ? "bg-card text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  AI 재번역
                </button>
              </div>

              {editMode === "manual" ? (
                <div role="tabpanel">
                  <Textarea
                    value={editText}
                    onChange={(e) => onEditTextChange(e.target.value)}
                    className="min-h-[72px] text-sm"
                    disabled={busy}
                    autoFocus
                  />
                  <p className="mt-1 text-[10px] text-muted-foreground">
                    번역문을 직접 고친 뒤 수정 후보를 확인합니다.
                  </p>
                </div>
              ) : (
                <div role="tabpanel" className="space-y-2">
                  <div className="rounded-md bg-muted/50 px-2.5 py-2">
                    <p className="text-[10px] font-bold text-muted-foreground mb-0.5">현재 번역</p>
                    <p className="text-xs leading-snug">{frag.target}</p>
                  </div>
                  <input
                    type="text"
                    value={instruct}
                    onChange={(e) => setInstruct(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !busy) submitRetranslate();
                    }}
                    placeholder="추가 요청사항 (선택) · 예: 더 격식있게, 존댓말로"
                    disabled={busy}
                    autoFocus
                    className="w-full text-xs bg-card border rounded px-2.5 py-2 outline-none focus:border-primary disabled:opacity-50"
                  />
                  <p className="text-[10px] text-muted-foreground">
                    요청사항을 비우면 원문을 기준으로 다시 번역합니다.
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm leading-snug">
              <StyledText segments={frag.style_segments} fallback={frag.target} />
            </div>
          )}
        </div>
      ) : (
        // 컴팩트 (짧은·중간): 번역 크게, 원문 흐리게 아래
        <div>
          <div className="text-sm leading-snug">
            <StyledText segments={frag.style_segments} fallback={frag.target} />
          </div>
          <div className="text-[11px] text-foreground/45 leading-snug mt-0.5">
            {frag.source}
          </div>
        </div>
      )}

      {frag.findings.map((finding, i) => (
        <div
          key={i}
          className="mt-1.5 text-[11px] text-muted-foreground bg-muted/50 rounded px-2 py-1"
        >
          {finding.description}
          {finding.suggested_fix && (
            <span className="text-foreground"> · 제안: {finding.suggested_fix}</span>
          )}
        </div>
      ))}

      {frag.length_budget !== null && !frag.is_note && (
        <div className="mt-1 text-[10px] text-muted-foreground">
          📐 {frag.length_budget}자 · {frag.target.length}자
          {overflow && <span className="text-destructive font-semibold"> (초과)</span>}
        </div>
      )}

      {/* 직접 수정과 AI 재번역이 공유하는 적용 옵션/액션 바 */}
      {editing && (
        <div className="mt-2 flex gap-1.5 flex-wrap items-center">
          <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground mr-auto">
            <input
              type="checkbox"
              checked={propagate}
              onChange={(e) => onPropagateChange(e.target.checked)}
              className="accent-primary"
            />
            동일 문구 함께 적용
          </label>
          <Button size="xs" variant="outline" disabled={busy} onClick={onCancelEdit}>
            취소
          </Button>
          <Button
            size="xs"
            className="gap-1"
            disabled={busy}
            onClick={editMode === "manual" ? onPreviewManual : submitRetranslate}
          >
            {busy ? (
              <Loader2 className="w-3 h-3 animate-spin" />
            ) : editMode === "manual" ? (
              <Check className="w-3 h-3" />
            ) : (
              <RefreshCw className="w-3 h-3" />
            )}
            {editMode === "manual" ? "확인" : "AI 번역"}
          </Button>
        </div>
      )}
    </div>
  );
}
