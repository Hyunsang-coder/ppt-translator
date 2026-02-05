"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { Moon, Sun, Languages, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface HeaderProps {
  activeTab: "translate" | "extract";
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
    <header className="sticky top-0 z-50 border-b border-border bg-foreground text-background">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          {/* Logo and Title */}
          <Link href="/translate" className="flex items-center gap-3 group">
            <div className="relative">
              <div className="absolute inset-0 brand-gradient rounded-xl blur-lg opacity-50 group-hover:opacity-75 transition-opacity" />
              <Image
                src="/cat-logo.png"
                alt="번역캣"
                width={44}
                height={44}
                className="relative rounded-xl shadow-lg group-hover:scale-105 transition-transform"
              />
            </div>
            <div className="flex flex-col">
              <h1 className="text-xl font-bold text-background">
                PPT 번역캣
              </h1>
              <span className="text-xs text-background/60 hidden sm:block">
                AI 기반 PowerPoint 번역
              </span>
            </div>
          </Link>

          {/* Navigation and Actions */}
          <div className="flex items-center gap-4">
            {/* Tab Navigation */}
            <Tabs value={activeTab} className="hidden sm:block">
              <TabsList className="bg-background/10 border-0">
                <TabsTrigger value="translate" asChild className="data-[state=active]:bg-background data-[state=active]:text-foreground text-background/70">
                  <Link href="/translate" className="flex items-center gap-2">
                    <Languages className="w-4 h-4" />
                    <span>번역</span>
                  </Link>
                </TabsTrigger>
                <TabsTrigger value="extract" asChild className="data-[state=active]:bg-background data-[state=active]:text-foreground text-background/70">
                  <Link href="/extract" className="flex items-center gap-2">
                    <FileText className="w-4 h-4" />
                    <span>텍스트 추출</span>
                  </Link>
                </TabsTrigger>
              </TabsList>
            </Tabs>

            {/* Mobile Navigation */}
            <div className="flex sm:hidden">
              <Tabs value={activeTab}>
                <TabsList className="bg-background/10 border-0">
                  <TabsTrigger value="translate" asChild className="data-[state=active]:bg-background data-[state=active]:text-foreground text-background/70">
                    <Link href="/translate" className="px-3">
                      <Languages className="w-4 h-4" />
                    </Link>
                  </TabsTrigger>
                  <TabsTrigger value="extract" asChild className="data-[state=active]:bg-background data-[state=active]:text-foreground text-background/70">
                    <Link href="/extract" className="px-3">
                      <FileText className="w-4 h-4" />
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
              className="relative overflow-hidden hover:bg-background/10"
              aria-label="테마 변경"
            >
              {mounted ? (
                theme === "dark" ? (
                  <Sun className="w-5 h-5 text-background animate-scale-in" />
                ) : (
                  <Moon className="w-5 h-5 text-background animate-scale-in" />
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
