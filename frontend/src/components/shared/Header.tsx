"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { Moon, Sun, Languages, FileText, Info, ScrollText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface HeaderProps {
  activeTab: "translate" | "extract" | "how-it-works" | "patch-notes";
}

export function Header({ activeTab }: HeaderProps) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark");
  };

  return (
    <header className="sticky top-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-xl text-foreground">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          {/* Logo and Title */}
          <Link href="/translate" className="flex items-center gap-3 group">
            <div className="relative">
              <div className="absolute inset-0 bg-primary/20 rounded-xl blur-lg opacity-50 group-hover:opacity-75 transition-opacity" />
              <Image
                src="/cat-logo.png"
                alt="번역캣"
                width={44}
                height={44}
                className="relative rounded-xl shadow-lg group-hover:scale-105 transition-transform"
              />
            </div>
            <div className="flex flex-col">
              <h1 className="text-xl font-bold text-foreground">
                PPT 번역캣
              </h1>
              <span className="text-xs text-muted-foreground hidden sm:block">
                AI 기반 PowerPoint 번역
              </span>
            </div>
          </Link>

          {/* Navigation and Actions */}
          <div className="flex items-center gap-4">
            {/* Tab Navigation */}
            <Tabs value={activeTab} className="hidden sm:block">
              <TabsList className="bg-muted/60 border border-border/50">
                <TabsTrigger value="translate" asChild className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground text-muted-foreground">
                  <Link href="/translate" className="flex items-center gap-2">
                    <Languages className="w-4 h-4" />
                    <span>번역</span>
                  </Link>
                </TabsTrigger>
                <TabsTrigger value="extract" asChild className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground text-muted-foreground">
                  <Link href="/extract" className="flex items-center gap-2">
                    <FileText className="w-4 h-4" />
                    <span>텍스트 추출</span>
                  </Link>
                </TabsTrigger>
                <TabsTrigger value="how-it-works" asChild className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground text-muted-foreground">
                  <Link href="/how-it-works" className="flex items-center gap-2">
                    <Info className="w-4 h-4" />
                    <span>작동 원리</span>
                  </Link>
                </TabsTrigger>
                <TabsTrigger value="patch-notes" asChild className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground text-muted-foreground">
                  <Link href="/patch-notes" className="flex items-center gap-2">
                    <ScrollText className="w-4 h-4" />
                    <span>패치 노트</span>
                  </Link>
                </TabsTrigger>
              </TabsList>
            </Tabs>

            {/* Mobile Navigation */}
            <div className="flex sm:hidden">
              <Tabs value={activeTab}>
                <TabsList className="bg-muted/60 border border-border/50">
                  <TabsTrigger value="translate" asChild className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground text-muted-foreground">
                    <Link href="/translate" className="px-3">
                      <Languages className="w-4 h-4" />
                    </Link>
                  </TabsTrigger>
                  <TabsTrigger value="extract" asChild className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground text-muted-foreground">
                    <Link href="/extract" className="px-3">
                      <FileText className="w-4 h-4" />
                    </Link>
                  </TabsTrigger>
                  <TabsTrigger value="how-it-works" asChild className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground text-muted-foreground">
                    <Link href="/how-it-works" className="px-3">
                      <Info className="w-4 h-4" />
                    </Link>
                  </TabsTrigger>
                  <TabsTrigger value="patch-notes" asChild className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground text-muted-foreground">
                    <Link href="/patch-notes" className="px-3">
                      <ScrollText className="w-4 h-4" />
                    </Link>
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            {/* Theme Toggle */}
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
              className="relative overflow-hidden hover:bg-accent"
              aria-label="테마 변경"
            >
              {mounted ? (
                theme === "dark" ? (
                  <Sun className="w-5 h-5 text-foreground animate-scale-in" />
                ) : (
                  <Moon className="w-5 h-5 text-foreground animate-scale-in" />
                )
              ) : (
                <div className="w-5 h-5" />
              )}
            </Button>
          </div>
        </div>
      </div>
    </header>
  );
}
