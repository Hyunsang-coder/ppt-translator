"use client";

import Image from "next/image";
import Link from "next/link";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ExtractionForm } from "@/components/extraction/ExtractionForm";

export default function ExtractPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Image
                src="/cat-logo.png"
                alt="번역캣"
                width={40}
                height={40}
                className="rounded"
              />
              <h1 className="text-2xl font-bold">PPT 번역캣</h1>
            </div>
            <Tabs defaultValue="extract">
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
        <ExtractionForm />
      </main>

      {/* Footer */}
      <footer className="border-t mt-auto">
        <div className="container mx-auto px-4 py-4 text-center text-sm text-muted-foreground">
          PPT 번역캣 - PowerPoint 텍스트 추출 도구
        </div>
      </footer>
    </div>
  );
}
