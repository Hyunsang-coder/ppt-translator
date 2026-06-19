import Image from "next/image";
import { Apple, Download, Github, MonitorDown } from "lucide-react";
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

export default async function Home() {
  const latestRelease = await getLatestRelease();

  return (
    <main className="min-h-screen bg-background text-foreground">
      <section className="mx-auto flex min-h-screen max-w-3xl flex-col items-center justify-center px-6 py-12 text-center">
        <Image
          src="/cat-logo.png"
          alt="PPT 번역캣"
          width={96}
          height={96}
          priority
          className="mb-6 rounded-2xl shadow-lg"
        />
        <p className="mb-3 text-sm font-medium text-muted-foreground">
          데스크톱 앱으로 제공됩니다
        </p>
        <h1 className="text-4xl font-bold tracking-normal sm:text-5xl">
          PPT 번역캣
        </h1>
        {latestRelease ? (
          <p className="mt-4 rounded-full border border-border bg-muted/40 px-4 py-1.5 text-sm text-muted-foreground">
            최신 버전{" "}
            <span className="font-semibold text-foreground">
              {latestRelease.tagName}
            </span>
            <span aria-hidden="true"> · </span>
            <time dateTime={latestRelease.publishedAt}>
              {formatReleaseDate(latestRelease.publishedAt)} 배포
            </time>
          </p>
        ) : (
          <p className="mt-4 text-sm text-muted-foreground">
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
        <p className="mt-5 max-w-2xl text-base leading-7 text-muted-foreground sm:text-lg">
          슬라이드 번역 후 글꼴·색상·강조가 깨지는 문제, 용어가 슬라이드마다
          달라지는 문제를 줄입니다. PPT 번역캣은 문맥과 용어집을 반영해
          번역하고, 원본 서식을 최대한 유지한 채 결과 파일을 만듭니다. 아래에서
          macOS 또는 Windows용 앱을 설치하세요.
        </p>
        <div className="mt-8 grid w-full gap-3 sm:grid-cols-3">
          {downloads.map(({ label, href, icon: Icon }) => (
            <Button key={href} asChild size="lg" className="gap-2">
              <a href={href}>
                <Icon className="h-4 w-4" />
                {label}
              </a>
            </Button>
          ))}
        </div>
        <div className="mt-4 flex flex-col gap-3 sm:flex-row">
          <Button asChild variant="outline" size="lg" className="gap-2">
            <a href="https://github.com/Hyunsang-coder/ppt-translator">
              <Github className="h-4 w-4" />
              GitHub
            </a>
          </Button>
          <Button asChild variant="outline" size="lg" className="gap-2">
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
