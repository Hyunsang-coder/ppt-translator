"use client";

import { Header } from "@/components/shared/Header";
import { ExtractionForm } from "@/components/extraction/ExtractionForm";

export default function ExtractPage() {
  return (
    <div className="min-h-screen animated-gradient-bg flex flex-col">
      <Header activeTab="extract" />

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 flex-1">
        <ExtractionForm />
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
