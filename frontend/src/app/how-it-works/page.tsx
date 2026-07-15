"use client";

import Link from "next/link";
import {
  ArrowRight,
  CheckCircle2,
  Download,
  FileSearch,
  FileText,
  Languages,
  ListChecks,
  Palette,
  SearchCheck,
  Settings2,
  ShieldCheck,
  Sparkles,
  Upload,
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
    icon: Upload,
    title: "파일을 올리고 설정을 고릅니다",
    description:
      "PPT를 선택한 뒤 원문·번역 언어, AI 모델, 용어집, 글상자 맞춤 같은 옵션을 정합니다.",
  },
  {
    icon: Languages,
    title: "슬라이드를 읽고 번역합니다",
    description:
      "문장과 서식을 분석한 뒤, 앞뒤 슬라이드 문맥과 용어집을 참고해 자연스럽게 옮깁니다.",
  },
  {
    icon: ListChecks,
    title: "검토 화면에서 다듬습니다",
    description:
      "번역이 끝나면 검토 & 수정이 열립니다. 표시된 항목을 고치거나 일부만 다시 번역할 수 있습니다.",
  },
  {
    icon: Download,
    title: "확정하고 저장합니다",
    description:
      "마음에 들면 최종 PPT로 반영해 다운로드합니다. 원본 서식은 가능한 한 그대로 유지됩니다.",
  },
];

const helpers = [
  {
    icon: FileSearch,
    title: "용어집",
    description:
      "고유 명사·게임 용어처럼 정해 둔 표현을 등록하면 번역에 우선 반영합니다. Excel/CSV로도 가져올 수 있습니다.",
  },
  {
    icon: Settings2,
    title: "글상자 맞춤",
    description:
      "번역문이 길어지면 글자 크기를 줄이거나 상자를 넓혀 넘치지 않게 맞춥니다. 글자 수 가이드도 켤 수 있습니다.",
  },
  {
    icon: Palette,
    title: "색상·강조 보존",
    description:
      "빨강·파랑처럼 강조된 말은 번역문에서 같은 뜻의 표현을 찾아 다시 칠합니다. 위치가 아니라 의미를 봅니다.",
  },
  {
    icon: FileText,
    title: "텍스트 추출",
    description:
      "번역이 아니라 Markdown으로 뽑아 두고 싶을 때는 텍스트 추출 탭을 사용합니다.",
  },
];

const checks = [
  "번역이 끝나면 바로 파일이 내려받아지지 않고, 검토 화면에서 한 번 더 확인할 수 있습니다.",
  "용어가 어긋났거나, 문장이 넘치거나, 강조 색이 빠진 항목은 검토 화면에서 표시됩니다.",
  "같은 문구가 여러 곳에 있으면, 한 곳만 고친 뒤 같은 표현에 함께 반영할 수 있습니다.",
  "강조 색은 확실할 때만 입힙니다. 애매하면 잘못된 위치에 칠하지 않고 기본 스타일로 둡니다.",
];

const caveats = [
  {
    title: "어순이 크게 바뀌는 문장",
    description:
      "한국어의 뒤쪽 표현이 영어에서는 앞쪽으로 이동할 수 있어, 강조 색의 위치도 달라 보일 수 있습니다.",
  },
  {
    title: "짧은 말이 길어지는 경우",
    description:
      "원문의 짧은 표현이 번역문에서는 설명형으로 늘어나면, 강조 범위가 넓어지거나 글상자가 넘칠 수 있습니다.",
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
                번역 → 검토 → 저장까지 한 흐름
              </div>
              <div className="max-w-3xl space-y-3">
                <h1 className="text-3xl font-bold tracking-normal text-foreground sm:text-4xl">
                  PPT 번역캣은 이렇게 작업합니다
                </h1>
                <p className="text-base leading-7 text-muted-foreground sm:text-lg">
                  파일을 올리면 슬라이드를 읽고 번역한 뒤, 검토 화면에서 다듬고
                  저장합니다. 서식은 최대한 유지하고, 용어집·글상자 맞춤·강조
                  색 보정으로 결과 품질을 높입니다.
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

            <section className="space-y-4">
              <div className="flex items-center gap-2">
                <Settings2 className="h-5 w-5 text-primary" />
                <h2 className="text-xl font-semibold text-foreground">
                  함께 쓰는 기능
                </h2>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                {helpers.map((item) => {
                  const Icon = item.icon;
                  return (
                    <div
                      key={item.title}
                      className="rounded-lg border border-border bg-card p-5"
                    >
                      <div className="flex items-center gap-2">
                        <Icon className="h-4 w-4 text-primary" />
                        <h3 className="text-base font-semibold text-foreground">
                          {item.title}
                        </h3>
                      </div>
                      <p className="mt-2 text-sm leading-6 text-muted-foreground">
                        {item.description}
                      </p>
                    </div>
                  );
                })}
              </div>
            </section>

            <section className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
              <Card className="rounded-lg">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <SearchCheck className="h-5 w-5 text-primary" />
                    <CardTitle className="text-xl">검토할 때 보면 좋은 점</CardTitle>
                  </div>
                  <CardDescription>
                    번역이 끝난 뒤에도 바로 확정하지 않고, 문제 있는 항목부터
                    확인할 수 있습니다.
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
                  검토 화면의 표시 항목과 색이 들어간 문구를 먼저 훑어보세요.
                  색이 빠진 경우보다 엉뚱한 단어에 색이 붙은 경우가 더 중요합니다.
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
