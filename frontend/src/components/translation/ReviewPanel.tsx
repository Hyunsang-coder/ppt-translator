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

type FilterKey = "all" | "flagged" | "edited";

export function ReviewPanel({ jobId, onClose, onDownload }: ReviewPanelProps) {
  const [fragments, setFragments] = useState<FragmentItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<FilterKey>("all");
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
    } catch {
      setError("조각 목록을 불러오지 못했습니다.");
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

  const visible = useMemo(() => {
    if (filter === "flagged") return fragments.filter((f) => f.findings.length > 0);
    if (filter === "edited") return fragments.filter((f) => f.edited);
    return fragments;
  }, [fragments, filter]);

  // Group visible fragments by slide, preserving order.
  const bySlide = useMemo(() => {
    const groups: { slide: number; title: string | null; items: FragmentItem[] }[] = [];
    for (const f of visible) {
      const last = groups[groups.length - 1];
      if (last && last.slide === f.slide) {
        last.items.push(f);
      } else {
        groups.push({ slide: f.slide, title: f.slide_title, items: [f] });
      }
    }
    return groups;
  }, [visible]);

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
      const propagated = resp.changed_indices.length - 1;
      toast.success(
        propagated > 0
          ? `수정이 ${resp.changed_indices.length}곳에 반영됐습니다.`
          : "수정이 반영됐습니다."
      );
      setEditingIndex(null);
      await load();
    } catch {
      toast.error("수정 반영에 실패했습니다.");
    } finally {
      setBusyIndex(null);
    }
  };

  const retranslate = async (frag: FragmentItem, instruction?: string) => {
    setBusyIndex(frag.index);
    try {
      await apiClient.editJobFragment(jobId, frag.index, {
        action: "retranslate",
        instruction,
        propagate_identical: propagate,
      });
      toast.success("재번역이 반영됐습니다.");
      await load();
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
      toast.success("검출을 무시했습니다.");
      await load();
    } catch {
      toast.error("처리에 실패했습니다.");
    } finally {
      setBusyIndex(null);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-background/95 backdrop-blur-sm overflow-y-auto">
      <div className="max-w-4xl mx-auto px-4 py-6">
        {/* header */}
        <div className="sticky top-0 z-10 bg-background/95 backdrop-blur-sm pb-3 mb-2 border-b">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <div>
              <h2 className="text-lg font-bold">번역 검토 &amp; 수정</h2>
              <p className="text-sm text-muted-foreground">
                {fragments.length}개 조각 · 검출 {flaggedCount} · 수정됨 {editedCount}
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
          {/* filters */}
          <div className="flex gap-2 mt-3">
            {(
              [
                ["all", `전체 ${fragments.length}`],
                ["flagged", `⚠️ 확인 필요 ${flaggedCount}`],
                ["edited", `수정됨 ${editedCount}`],
              ] as [FilterKey, string][]
            ).map(([key, label]) => (
              <button
                key={key}
                onClick={() => setFilter(key)}
                className={`text-xs font-semibold px-3 py-1.5 rounded-full border transition-colors ${
                  filter === key
                    ? "bg-foreground text-background border-foreground"
                    : "bg-card text-muted-foreground border-border hover:border-foreground/40"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {loading && (
          <div className="flex items-center justify-center py-16 text-muted-foreground gap-2">
            <Loader2 className="w-5 h-5 animate-spin" />
            조각을 불러오는 중...
          </div>
        )}

        {error && (
          <div className="p-3 rounded-lg border border-destructive/30 bg-destructive/10 text-sm text-destructive flex items-center gap-2">
            <AlertTriangle className="w-4 h-4" />
            {error}
          </div>
        )}

        {!loading && !error && bySlide.map((group) => (
          <div key={group.slide} className="mb-5">
            <div className="flex items-center gap-2 mb-2 mt-4">
              <span className="text-xs font-bold text-primary bg-primary/10 rounded px-2 py-1">
                S{group.slide}
              </span>
              {group.title && (
                <h3 className="text-sm font-bold truncate">{group.title}</h3>
              )}
              <span className="text-xs text-muted-foreground">조각 {group.items.length}개</span>
            </div>

            {group.items.map((frag) => (
              <div
                key={frag.index}
                className="glass-card rounded-xl border p-4 mb-2.5"
              >
                <div className="flex items-center gap-2 mb-2 text-xs font-semibold text-muted-foreground flex-wrap">
                  <span>
                    Slide {frag.slide} · {frag.is_note ? "발표자 노트" : `도형 ${frag.shape}`}
                  </span>
                  {frag.repeat_count > 1 && (
                    <span className="text-muted-foreground bg-muted rounded-full px-2 py-0.5">
                      반복 ×{frag.repeat_count}
                    </span>
                  )}
                  {frag.edited && (
                    <span className="text-success bg-success/10 rounded-full px-2 py-0.5">
                      수정됨
                    </span>
                  )}
                  {frag.findings.map((finding, i) => {
                    const s = badgeStyle(finding);
                    return (
                      <span key={i} className={`rounded-full px-2 py-0.5 ${s.cls}`}>
                        {s.label}
                      </span>
                    );
                  })}
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <div className="text-[11px] font-bold text-muted-foreground tracking-wide mb-1">
                      원문
                    </div>
                    <div className="text-sm text-foreground/80">{frag.source}</div>
                  </div>
                  <div>
                    <div className="text-[11px] font-bold text-muted-foreground tracking-wide mb-1">
                      번역
                    </div>
                    {editingIndex === frag.index ? (
                      <Textarea
                        value={editText}
                        onChange={(e) => setEditText(e.target.value)}
                        className="min-h-[60px]"
                        autoFocus
                      />
                    ) : (
                      <div className="text-sm">{frag.target}</div>
                    )}
                  </div>
                </div>

                {frag.findings.map((finding, i) => (
                  <div
                    key={i}
                    className="mt-2 text-xs text-muted-foreground bg-muted/50 rounded-lg px-3 py-2"
                  >
                    {finding.description}
                    {finding.suggested_fix && (
                      <span className="text-foreground"> · 제안: {finding.suggested_fix}</span>
                    )}
                  </div>
                ))}

                {frag.length_budget !== null && !frag.is_note && (
                  <div className="mt-1.5 text-[11px] text-muted-foreground">
                    📐 길이 예산 약 {frag.length_budget}자 · 현재 {frag.target.length}자
                    {frag.target.length > frag.length_budget && (
                      <span className="text-destructive font-semibold"> (초과)</span>
                    )}
                  </div>
                )}

                {/* actions */}
                <div className="mt-3 flex gap-2 flex-wrap">
                  {editingIndex === frag.index ? (
                    <>
                      <label className="flex items-center gap-1.5 text-xs text-muted-foreground mr-auto">
                        <input
                          type="checkbox"
                          checked={propagate}
                          onChange={(e) => setPropagate(e.target.checked)}
                          className="accent-primary"
                        />
                        동일 문구에 함께 적용
                      </label>
                      <Button
                        size="xs"
                        variant="outline"
                        onClick={() => setEditingIndex(null)}
                      >
                        취소
                      </Button>
                      <Button
                        size="xs"
                        className="gap-1"
                        disabled={busyIndex === frag.index}
                        onClick={() => applyEdit(frag)}
                      >
                        {busyIndex === frag.index ? (
                          <Loader2 className="w-3 h-3 animate-spin" />
                        ) : (
                          <Check className="w-3 h-3" />
                        )}
                        저장
                      </Button>
                    </>
                  ) : (
                    <>
                      <Button size="xs" variant="outline" className="gap-1" onClick={() => startEdit(frag)}>
                        <Pencil className="w-3 h-3" />
                        직접 수정
                      </Button>
                      <Button
                        size="xs"
                        variant="outline"
                        className="gap-1"
                        disabled={busyIndex === frag.index}
                        onClick={() => retranslate(frag, "더 짧게")}
                      >
                        <RefreshCw className="w-3 h-3" />
                        재번역
                      </Button>
                      {frag.findings.length > 0 && (
                        <Button
                          size="xs"
                          variant="ghost"
                          className="gap-1 text-muted-foreground"
                          disabled={busyIndex === frag.index}
                          onClick={() => ignore(frag)}
                        >
                          <Ban className="w-3 h-3" />
                          무시
                        </Button>
                      )}
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        ))}

        {!loading && !error && visible.length === 0 && (
          <div className="text-center py-16 text-muted-foreground text-sm">
            표시할 조각이 없습니다.
          </div>
        )}
      </div>
    </div>
  );
}
