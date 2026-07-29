"use client";

import { Check } from "lucide-react";

export interface SlideProgress {
  slide: number;
  title: string | null;
  remaining: number;
}

interface SlideRailProps {
  resolved: number;
  total: number;
  /** Slides with work left, in deck order. Finished ones are summarised below. */
  slides: SlideProgress[];
  doneSlides: number;
  activeSlide: number | null;
  /** Total blocks in the deck, for the link into the full list. */
  allCount: number;
  onSelectSlide: (slide: number) => void;
  onShowAll: () => void;
}

/** How far the queue has come and where the remaining work sits. */
export function SlideRail({
  resolved,
  total,
  slides,
  doneSlides,
  activeSlide,
  allCount,
  onSelectSlide,
  onShowAll,
}: SlideRailProps) {
  const percent = total > 0 ? (resolved / total) * 100 : 0;

  return (
    <nav className="flex w-[230px] shrink-0 flex-col border-r border-border bg-card">
      <div className="border-b border-border px-4 pb-3 pt-4">
        <p className="mb-2 text-xs font-semibold text-muted-foreground">검토 진행</p>
        <p className="flex items-baseline gap-1.5">
          <span className="text-[30px] font-bold leading-none tracking-[-0.03em]">
            {resolved}
          </span>
          <span className="text-[15px] text-muted-foreground">/ {total}건</span>
        </p>
        <div className="mt-2.5 h-1.5 rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-primary transition-all duration-500 ease-out"
            style={{ width: `${percent}%` }}
          />
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-0.5 overflow-y-auto px-2.5 pb-3.5 pt-2.5">
        <p className="px-2 pb-1.5 pt-1 text-[11px] font-semibold text-muted-foreground">
          슬라이드별 남은 항목
        </p>
        {slides.map((entry) => {
          const active = entry.slide === activeSlide;
          return (
            <button
              key={entry.slide}
              type="button"
              onClick={() => onSelectSlide(entry.slide)}
              className={`flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors ${
                active ? "bg-primary/12" : "hover:bg-muted/60"
              }`}
            >
              <span
                className={`w-5 text-xs font-bold ${
                  active ? "text-primary" : "text-muted-foreground"
                }`}
              >
                {entry.slide}
              </span>
              <span
                className={`flex-1 truncate text-[13px] ${
                  active ? "font-semibold text-foreground" : "text-muted-foreground"
                }`}
              >
                {entry.title || `슬라이드 ${entry.slide}`}
              </span>
              <span
                className={`inline-flex size-5 items-center justify-center rounded-full text-[11px] font-bold ${
                  active ? "bg-primary text-primary-foreground" : "bg-muted"
                }`}
              >
                {entry.remaining}
              </span>
            </button>
          );
        })}

        <div className="mx-2 my-2.5 h-px bg-border" />
        {doneSlides > 0 && (
          <p className="flex items-center gap-2 px-2.5 py-1.5 text-xs text-muted-foreground">
            <Check className="size-3.5 text-success" />
            확인 끝난 슬라이드 {doneSlides}개
          </p>
        )}
        <button
          type="button"
          onClick={onShowAll}
          className="px-2.5 py-1.5 text-left text-xs font-medium text-primary hover:underline"
        >
          전체 {allCount}개 문구 보기
        </button>
      </div>
    </nav>
  );
}
