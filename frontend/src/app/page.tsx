import Image from "next/image";
import {
  Apple,
  BookMarked,
  Download,
  Github,
  Languages,
  MonitorDown,
  Wand2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  formatReleaseDate,
  getLatestRelease,
} from "@/lib/github-release";

const releaseBase =
  "https://github.com/Hyunsang-coder/ppt-translator/releases/latest/download";

const downloads = [
  {
    label: "macOS Apple Silicon",
    href: `${releaseBase}/ppt-translator-macos-arm64.dmg`,
    icon: Apple,
  },
  {
    label: "Windows",
    href: `${releaseBase}/ppt-translator-windows-x64-setup.exe`,
    icon: MonitorDown,
  },
];

const features = [
  {
    icon: Wand2,
    title: "서식 그대로",
    body: "글꼴·색상·강조를 흐트러뜨리지 않고 원본 레이아웃을 유지합니다.",
  },
  {
    icon: BookMarked,
    title: "용어는 일관되게",
    body: "용어집과 문맥을 반영해 슬라이드마다 같은 용어로 번역합니다.",
  },
  {
    icon: Languages,
    title: "언어는 알아서",
    body: "원문 언어를 자동으로 감지해 바로 번역을 시작합니다.",
  },
];

export default async function Home() {
  const latestRelease = await getLatestRelease();

  return (
    <main className="min-h-screen bg-background text-foreground">
      <section className="mx-auto flex min-h-screen max-w-4xl flex-col items-center justify-center px-6 py-16 text-center sm:py-20">
        {/* Hero: identity */}
        <Image
          src="/cat-logo.png"
          alt="PPT 번역캣"
          width={88}
          height={88}
          priority
          className="mb-6 rounded-2xl shadow-lg"
        />
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
          PPT 번역캣
        </h1>
        <p className="mt-4 max-w-xl text-lg text-muted-foreground sm:text-xl">
          슬라이드 서식은 그대로, 용어는 일관되게.
          <br className="hidden sm:inline" /> 데스크톱 앱으로 PowerPoint를 번역하세요.
        </p>

        {latestRelease ? (
          <p className="mt-5 inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/40 px-3.5 py-1.5 text-sm text-muted-foreground">
            최신 버전{" "}
            <span className="font-semibold text-foreground">
              {latestRelease.tagName}
            </span>
            <span aria-hidden="true" className="text-border">
              ·
            </span>
            <time dateTime={latestRelease.publishedAt}>
              {formatReleaseDate(latestRelease.publishedAt)} 배포
            </time>
          </p>
        ) : (
          <p className="mt-5 text-sm text-muted-foreground">
            최신 릴리스 정보를 불러오지 못했습니다.{" "}
            <a
              href="https://github.com/Hyunsang-coder/ppt-translator/releases"
              className="underline underline-offset-4 hover:text-foreground"
            >
              GitHub Releases
            </a>
            에서 확인해 주세요.
          </p>
        )}

        {/* Value: three things it does */}
        <ul className="mt-12 grid w-full gap-4 text-left sm:grid-cols-3">
          {features.map(({ icon: Icon, title, body }) => (
            <li
              key={title}
              className="rounded-xl border border-border bg-card/60 p-5"
            >
              <span className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
                <Icon className="h-5 w-5" />
              </span>
              <h2 className="mt-3 text-base font-semibold">{title}</h2>
              <p className="mt-1.5 text-sm leading-6 text-muted-foreground">
                {body}
              </p>
            </li>
          ))}
        </ul>

        {/* Action: download */}
        <div className="mt-12 flex w-full max-w-md flex-col gap-3 sm:flex-row sm:justify-center">
          {downloads.map(({ label, href, icon: Icon }) => (
            <Button
              key={href}
              asChild
              size="lg"
              className="flex-1 gap-2"
            >
              <a href={href}>
                <Icon className="h-4 w-4" />
                {label}
              </a>
            </Button>
          ))}
        </div>

        <div className="mt-4 flex items-center gap-1 text-sm text-muted-foreground">
          <Button asChild variant="link" size="sm" className="gap-1.5 text-muted-foreground hover:text-foreground">
            <a href="https://github.com/Hyunsang-coder/ppt-translator">
              <Github className="h-4 w-4" />
              GitHub
            </a>
          </Button>
          <span aria-hidden="true" className="text-border">
            ·
          </span>
          <Button asChild variant="link" size="sm" className="gap-1.5 text-muted-foreground hover:text-foreground">
            <a
              href={
                latestRelease?.htmlUrl ??
                "https://github.com/Hyunsang-coder/ppt-translator/releases"
              }
            >
              <Download className="h-4 w-4" />
              릴리스 노트
            </a>
          </Button>
        </div>
      </section>
    </main>
  );
}
