"use client";

import { Header } from "@/components/shared/Header";
import { PatchNotesList } from "@/components/patch-notes/PatchNotesList";

export default function PatchNotesPage() {
  return (
    <div className="min-h-screen animated-gradient-bg flex flex-col">
      <Header activeTab="patch-notes" />

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 flex-1">
        <PatchNotesList />
      </main>

      {/* Footer */}
      <footer className="border-t border-border/50 mt-auto">
        <div className="container mx-auto px-4 py-4 text-center text-sm text-muted-foreground">
          <span className="font-medium text-foreground">PPT 번역캣</span>
          <span className="mx-2">-</span>
          <span>created by Hyunsang Joo</span>
        </div>
      </footer>
    </div>
  );
}
