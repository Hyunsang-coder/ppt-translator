"use client";

import { Header } from "@/components/shared/Header";
import { TranslationForm } from "@/components/translation/TranslationForm";

export default function TranslatePage() {
  return (
    <div className="min-h-screen animated-gradient-bg flex flex-col">
      <Header activeTab="translate" />

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 flex-1">
        <TranslationForm />
      </main>

      {/* Footer */}
      <footer className="border-t border-border/50 mt-auto">
        <div className="container mx-auto px-4 py-4 text-center text-sm text-muted-foreground">
          <span className="font-medium text-foreground">PPT 번역캣</span>
          <span className="mx-2">-</span>
          <span>OpenAI GPT & Anthropic Claude 기반 번역 도구</span>
        </div>
      </footer>
    </div>
  );
}
