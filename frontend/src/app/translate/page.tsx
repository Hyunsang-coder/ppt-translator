"use client";

import Link from "next/link";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TranslationForm } from "@/components/translation/TranslationForm";

export default function TranslatePage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold">PPT 번역캣</h1>
            <Tabs defaultValue="translate">
              <TabsList>
                <TabsTrigger value="translate" asChild>
                  <Link href="/translate">번역</Link>
                </TabsTrigger>
                <TabsTrigger value="extract" asChild>
                  <Link href="/extract">텍스트 추출</Link>
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <TranslationForm />
      </main>

      {/* Footer */}
      <footer className="border-t mt-auto">
        <div className="container mx-auto px-4 py-4 text-center text-sm text-muted-foreground">
          PPT 번역캣 - OpenAI GPT & Anthropic Claude 기반 번역 도구
        </div>
      </footer>
    </div>
  );
}
