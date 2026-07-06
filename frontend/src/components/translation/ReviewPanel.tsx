"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { apiClient } from "@/lib/api-client";
import type { FragmentItem, FragmentFinding } from "@/types/api";
import {
  X,
  Pencil,
  RefreshCw,
  Ban,
  Check,
  Download,
  Loader2,
  AlertTriangle,
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

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await apiClient.getJobFragments(jobId);
      setFragments(resp.fragments);
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

  // 편집/재번역 결과를 전체 리로드 없이 해당 조각만 state에서 교체한다.
  // 이렇게 하면 리스트가 리렌더되지 않아 스크롤 위치가 유지된다.
  const patchFragments = useCallback(
    (changedIndices: number[], newTarget: string) => {
      const changed = new Set(changedIndices);
      setFragments((prev) =>
        prev.map((f) =>
          changed.has(f.index)
            ? { ...f, target: newTarget, edited: true }
            : f
        )
      );
    },
    []
  );

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

  const startEdit = (frag: FragmentItem) => {
    setEditingIndex(frag.index);
    setEditText(frag.target);
  };

  const applyEdit = async (frag: FragmentItem) => {
    setBusyIndex(frag.index);
    try {
      const resp = await apiClient.editJobFragment(jobId, frag.index, {
        action: "edit",
        target: editText,
        propagate_identical: propagate,
      });
      patchFragments(resp.changed_indices, resp.target);
      const propagated = resp.changed_indices.length - 1;
      toast.success(
        propagated > 0
          ? `수정이 ${resp.changed_indices.length}곳에 반영됐습니다.`
          : "수정이 반영됐습니다."
      );
      setEditingIndex(null);
    } catch {
      toast.error("수정 반영에 실패했습니다.");
    } finally {
      setBusyIndex(null);
    }
  };

  const retranslate = async (frag: FragmentItem, instruction?: string) => {
    setBusyIndex(frag.index);
    try {
      const resp = await apiClient.editJobFragment(jobId, frag.index, {
        action: "retranslate",
        instruction,
        propagate_identical: propagate,
      });
      patchFragments(resp.changed_indices, resp.target);
      toast.success("재번역이 반영됐습니다.");
    } catch {
      toast.error("재번역에 실패했습니다.");
    } finally {
      setBusyIndex(null);
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
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={onDownload} className="gap-2">
            <Download className="w-4 h-4" />
            수정 반영해 저장
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
                  onApplyEdit={() => applyEdit(frag)}
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
    </div>
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
  onApplyEdit: () => void;
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
  onApplyEdit,
  onRetranslate,
  onIgnore,
}: FragmentCardProps) {
  const overflow = isOverflow(frag);
  // long/편집 = 원문|번역 2단 대조. short/med = 번역 크게 + 원문 흐리게.
  const twoCol = span === "long" || editing;

  // 재번역 추가 요청 입력 토글. 카드 로컬 상태 — 카드마다 독립적으로 열린다.
  const [showInstruct, setShowInstruct] = useState(false);
  const [instruct, setInstruct] = useState("");
  const submitRetranslate = () => {
    const trimmed = instruct.trim();
    onRetranslate(trimmed || (overflow ? "더 짧게" : undefined));
    setShowInstruct(false);
    setInstruct("");
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
          <Button
            size="xs"
            variant="ghost"
            className="gap-1 h-6 px-1.5"
            disabled={busy}
            onClick={() => setShowInstruct((v) => !v)}
            title="추가 요청사항을 적어 재번역 (비워도 됨)"
          >
            {busy ? (
              <Loader2 className="w-3 h-3 animate-spin" />
            ) : (
              <RefreshCw className="w-3 h-3" />
            )}
            재번역
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
          <div className="text-sm text-foreground/70 leading-snug">{frag.source}</div>
          {editing ? (
            <Textarea
              value={editText}
              onChange={(e) => onEditTextChange(e.target.value)}
              className="min-h-[52px] text-sm"
              autoFocus
            />
          ) : (
            <div className="text-sm leading-snug">{frag.target}</div>
          )}
        </div>
      ) : (
        // 컴팩트 (짧은·중간): 번역 크게, 원문 흐리게 아래
        <div>
          <div className="text-sm leading-snug">{frag.target}</div>
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

      {/* 재번역: 추가 요청사항(선택) 입력 후 실행 */}
      {showInstruct && !editing && (
        <div className="mt-2 flex gap-1.5 items-center">
          <input
            type="text"
            value={instruct}
            onChange={(e) => setInstruct(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") submitRetranslate();
              if (e.key === "Escape") {
                setShowInstruct(false);
                setInstruct("");
              }
            }}
            placeholder="추가 요청사항 (선택) · 예: 더 격식있게, 존댓말로"
            autoFocus
            disabled={busy}
            className="flex-1 min-w-0 text-xs bg-card border rounded px-2 py-1 outline-none focus:border-primary"
          />
          <Button
            size="xs"
            variant="outline"
            className="shrink-0"
            disabled={busy}
            onClick={() => {
              setShowInstruct(false);
              setInstruct("");
            }}
          >
            취소
          </Button>
          <Button size="xs" className="gap-1 shrink-0" disabled={busy} onClick={submitRetranslate}>
            {busy ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
            재번역
          </Button>
        </div>
      )}

      {/* 편집 중 저장/취소 바 */}
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
          <Button size="xs" variant="outline" onClick={onCancelEdit}>
            취소
          </Button>
          <Button size="xs" className="gap-1" disabled={busy} onClick={onApplyEdit}>
            {busy ? <Loader2 className="w-3 h-3 animate-spin" /> : <Check className="w-3 h-3" />}
            저장
          </Button>
        </div>
      )}
    </div>
  );
}
