"use client";

import type { CSSProperties } from "react";
import type { StyleSegment } from "@/types/api";

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

export function reviewColorContrast(color: string | null): {
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

// Size and leading come from the block around it: the queue scales its body
// text to the fragment's length, and a size of its own would undo that.
export function StyledText({ segments, fallback }: { segments: StyleSegment[]; fallback: string }) {
  if (segments.length === 0) return <span>{fallback}</span>;
  return (
    <span>
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
