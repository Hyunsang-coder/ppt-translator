"use client";

import Link from "next/link";
import {
  ArrowRight,
  CheckCircle2,
  FileSearch,
  Languages,
  Palette,
  SearchCheck,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { DesktopShell } from "@/components/desktop-shell";
import { Header } from "@/components/shared/Header";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";

const workflow = [
  {
    icon: FileSearch,
    title: "PPT를 먼저 분석합니다",
    description:
      "슬라이드의 문장, 줄바꿈, 색상, 글자 스타일을 읽고 같은 문장이 반복되는 곳을 정리합니다.",
  },
  {
    icon: Languages,
    title: "문맥에 맞게 번역합니다",
    description:
      "앞뒤 슬라이드의 흐름, 용어집, 사용자가 입력한 톤 지침을 함께 참고해서 번역합니다.",
  },
  {
    icon: Palette,
    title: "강조 색상을 의미에 맞춥니다",
    description:
      "색이 섞인 문단은 색상 위치를 그대로 베끼지 않고, 원래 강조된 말이 번역문에서 어떤 표현이 되었는지 다시 찾습니다.",
  },
  {
    icon: ShieldCheck,
    title: "확실할 때만 적용합니다",
    description:
      "강조 구간이 번역문과 정확히 맞을 때만 색을 입힙니다. 애매하면 엉뚱한 곳에 색칠하지 않도록 기본 스타일로 둡니다.",
  },
];

const checks = [
  "번역된 조각들을 이어 붙였을 때 전체 문장과 정확히 같은지 확인합니다.",
  "빨간색, 파란색, 초록색 같은 강조 색상이 원문의 의미와 맞는지 우선합니다.",
  "한국어와 영어의 어순이 달라도 단순 위치가 아니라 표현의 뜻을 기준으로 봅니다.",
  "확신이 낮으면 색이 빠질 수 있지만, 잘못된 위치에 강조가 붙는 것보다 안전합니다.",
];

const caveats = [
  {
    title: "어순이 크게 바뀌는 문장",
    description:
      "한국어의 뒤쪽 표현이 영어에서는 앞쪽으로 이동할 수 있어 색상 위치도 달라질 수 있습니다.",
  },
  {
    title: "한 단어가 여러 단어로 늘어나는 경우",
    description:
      "예를 들어 짧은 원문 표현이 번역문에서는 설명형 문구가 되면 강조 범위가 넓어질 수 있습니다.",
  },
  {
    title: "강조 의미가 애매한 경우",
    description:
      "원문 색상이 장식인지, 특정 단어 강조인지 분명하지 않으면 보수적으로 처리합니다.",
  },
];

export default function HowItWorksPage() {
  return (
    <DesktopShell>
      <div className="min-h-screen animated-gradient-bg flex flex-col">
        <Header activeTab="how-it-works" />

        <main className="container mx-auto px-4 py-8 flex-1">
          <div className="mx-auto max-w-5xl space-y-8">
            <section className="space-y-4">
              <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1 text-sm text-muted-foreground">
                <Sparkles className="h-4 w-4 text-primary" />
                번역 품질과 서식 보존을 함께 봅니다
              </div>
              <div className="max-w-3xl space-y-3">
                <h1 className="text-3xl font-bold tracking-normal text-foreground sm:text-4xl">
                  PPT 번역캣은 이렇게 작업합니다
                </h1>
                <p className="text-base leading-7 text-muted-foreground sm:text-lg">
                  문장을 먼저 자연스럽게 옮기고, 원본의 색상과 강조는 번역문에서
                  같은 의미를 가진 표현에 다시 붙입니다. 단순히 원본의 위치를
                  따라 색을 칠하지 않습니다.
                </p>
              </div>
            </section>

            <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              {workflow.map((item, index) => {
                const Icon = item.icon;
                return (
                  <Card key={item.title} className="rounded-lg">
                    <CardHeader className="gap-3">
                      <div className="flex items-center justify-between gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-md bg-primary/10 text-primary">
                          <Icon className="h-5 w-5" />
                        </div>
                        <span className="text-sm font-medium text-muted-foreground">
                          {index + 1}
                        </span>
                      </div>
                      <CardTitle className="text-base leading-6">
                        {item.title}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm leading-6 text-muted-foreground">
                        {item.description}
                      </p>
                    </CardContent>
                  </Card>
                );
              })}
            </section>

            <section className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
              <Card className="rounded-lg">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <SearchCheck className="h-5 w-5 text-primary" />
                    <CardTitle className="text-xl">색상 보정 방식</CardTitle>
                  </div>
                  <CardDescription>
                    색이 여러 개 들어간 문단은 번역과 색상 매칭을 따로 보지 않고
                    한 번에 판단합니다.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {checks.map((check) => (
                    <div key={check} className="flex gap-3">
                      <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-success" />
                      <p className="text-sm leading-6 text-foreground">
                        {check}
                      </p>
                    </div>
                  ))}
                </CardContent>
              </Card>

              <Card className="rounded-lg">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="h-5 w-5 text-primary" />
                    <CardTitle className="text-xl">안전한 실패</CardTitle>
                  </div>
                  <CardDescription>
                    색상 판단이 애매할 때는 보기 좋은 결과보다 틀리지 않는 결과를
                    우선합니다.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="rounded-md border border-border bg-muted/40 p-4">
                    <p className="text-sm font-medium text-foreground">
                      좋은 경우
                    </p>
                    <p className="mt-2 text-sm leading-6 text-muted-foreground">
                      원문의 빨간색 “1 HP 아래로”가 번역문에서도 “below 1 HP”에만
                      붙습니다.
                    </p>
                  </div>
                  <div className="rounded-md border border-border bg-muted/40 p-4">
                    <p className="text-sm font-medium text-foreground">
                      피하려는 경우
                    </p>
                    <p className="mt-2 text-sm leading-6 text-muted-foreground">
                      원문에서 뒤쪽이 빨갛다는 이유만으로 번역문의 아무 뒤쪽
                      단어에 색이 붙는 상황입니다.
                    </p>
                  </div>
                </CardContent>
              </Card>
            </section>

            <section className="space-y-4">
              <div className="flex items-center gap-2">
                <Palette className="h-5 w-5 text-primary" />
                <h2 className="text-xl font-semibold text-foreground">
                  결과가 달라 보일 수 있는 이유
                </h2>
              </div>
              <div className="grid gap-4 md:grid-cols-3">
                {caveats.map((item) => (
                  <div
                    key={item.title}
                    className="rounded-lg border border-border bg-card p-5"
                  >
                    <h3 className="text-base font-semibold text-foreground">
                      {item.title}
                    </h3>
                    <p className="mt-2 text-sm leading-6 text-muted-foreground">
                      {item.description}
                    </p>
                  </div>
                ))}
              </div>
            </section>

            <section className="flex flex-col gap-4 rounded-lg border border-border bg-card p-6 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-lg font-semibold text-foreground">
                  번역 결과를 확인할 때
                </h2>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">
                  색이 들어간 문구를 먼저 훑어보세요. 색이 빠진 경우보다 엉뚱한
                  단어에 색이 붙은 경우가 더 중요합니다.
                </p>
              </div>
              <Button asChild className="gap-2 sm:shrink-0">
                <Link href="/translate">
                  번역하러 가기
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </section>
          </div>
        </main>

        <footer className="border-t border-border/50 mt-auto">
          <div className="container mx-auto px-4 py-4 text-center text-sm text-muted-foreground">
            <span className="font-medium text-foreground">PPT 번역캣</span>
            <span className="mx-2">-</span>
            <span>created by Hyunsang Joo</span>
          </div>
        </footer>
      </div>
    </DesktopShell>
  );
}
