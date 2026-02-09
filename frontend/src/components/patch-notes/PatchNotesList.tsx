"use client";

import Link from "next/link";
import { patchNotes } from "@/data/patch-notes";
import { PatchNoteCard } from "./PatchNoteCard";
import { Button } from "@/components/ui/button";

export function PatchNotesList() {
  return (
    <div className="max-w-3xl mx-auto space-y-12">
      {/* Hero */}
      <div className="text-center space-y-3 animate-slide-up">
        <h2 className="text-3xl sm:text-4xl font-bold brand-gradient-text">
          패치 노트
        </h2>
        <p className="text-muted-foreground text-base sm:text-lg">
          PPT 번역캣의 업데이트 내역을 확인하세요
        </p>
      </div>

      {/* Timeline */}
      <div className="relative space-y-6">
        {/* Vertical line - desktop only */}
        <div className="hidden md:block absolute left-7 top-7 bottom-7 w-px bg-border" />

        {patchNotes.map((note, index) => (
          <PatchNoteCard key={note.version} note={note} index={index} />
        ))}
      </div>

      {/* CTA */}
      <div
        className="text-center pt-4 animate-slide-up opacity-0"
        style={{ animationDelay: "0.6s", animationFillMode: "forwards" }}
      >
        <Button
          asChild
          className="btn-gradient px-8 py-3 text-base font-semibold rounded-xl"
        >
          <Link href="/translate">번역 시작하기</Link>
        </Button>
      </div>
    </div>
  );
}
