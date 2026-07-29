"use client";

import { Button } from "@/components/ui/button";
import { findingBadge } from "@/components/translation/review/finding-labels";
import { blockFindings, primaryFinding, type ReviewBlock } from "@/lib/review-queue";
import { ArrowLeft, Pencil } from "lucide-react";

interface FragmentListProps {
  /** Every block in the deck, in deck order — not just the flagged ones. */
  blocks: ReviewBlock[];
  resolved: Readonly<Record<string, unknown>>;
  onOpen: (key: string) => void;
  onBack: () => void;
}

/**
 * The queue only carries what needs a decision. This is the other half: read
 * the whole deck side by side, and pull any line into the queue to edit it —
 * including the ones nothing was flagged on.
 */
export function FragmentList({ blocks, resolved, onOpen, onBack }: FragmentListProps) {
  const bySlide = new Map<number, ReviewBlock[]>();
  for (const block of blocks) {
    const group = bySlide.get(block.slide);
    if (group) group.push(block);
    else bySlide.set(block.slide, [block]);
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center gap-2.5 px-7 pt-3.5">
        <Button variant="ghost" size="sm" className="h-[30px] gap-1.5" onClick={onBack}>
          <ArrowLeft className="size-3.5" />
          검토 큐로 돌아가기
        </Button>
        <span className="text-[13px] text-muted-foreground">
          전체 {blocks.length}개 문구
        </span>
      </div>

      <div className="flex-1 space-y-5 overflow-y-auto px-7 pb-5 pt-4">
        {[...bySlide.entries()].map(([slide, group]) => (
          <section key={slide}>
            <p className="mb-2.5 text-xs font-semibold text-muted-foreground">
              슬라이드 {slide}
              {group[0].items[0].slide_title && ` · ${group[0].items[0].slide_title}`}
            </p>
            <div className="overflow-hidden rounded-[10px] border border-border">
              {group.map((block, order) => {
                const primary = primaryFinding(block);
                const badge = primary ? findingBadge(primary.finding) : null;
                return (
                  <div
                    key={block.key}
                    className={`group grid grid-cols-2 gap-px bg-border ${
                      order > 0 ? "border-t border-border" : ""
                    }`}
                  >
                    <div className="bg-card px-[13px] py-2.5 text-[13px] text-foreground/65">
                      {block.items.map((item) => (
                        <p key={item.index}>{item.source}</p>
                      ))}
                    </div>
                    <div className="flex items-start gap-2 bg-card px-[13px] py-2.5 text-[13px]">
                      <span className="min-w-0 flex-1">
                        {block.items.map((item) => (
                          <span key={item.index} className="block">{item.target}</span>
                        ))}
                        {badge && (
                          <span
                            className={`mt-1 inline-block rounded-full px-2 py-0.5 text-[11px] font-bold ${badge.cls}`}
                          >
                            {badge.label}
                          </span>
                        )}
                        {!badge && blockFindings(block).length === 0 && block.key in resolved && (
                          <span className="mt-1 inline-block text-[11px] font-semibold text-success">
                            처리함
                          </span>
                        )}
                      </span>
                      <Button
                        variant="outline"
                        size="xs"
                        className="hidden gap-1 group-hover:inline-flex"
                        onClick={() => onOpen(block.key)}
                      >
                        <Pencil className="size-3" />
                        수정
                      </Button>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
