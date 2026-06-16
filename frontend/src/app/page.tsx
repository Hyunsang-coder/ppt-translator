import Image from "next/image";
import { Apple, Download, Github, MonitorDown } from "lucide-react";
import { Button } from "@/components/ui/button";

const releaseBase =
  "https://github.com/Hyunsang-coder/ppt-translator/releases/latest/download";

const downloads = [
  {
    label: "macOS Apple Silicon",
    href: `${releaseBase}/ppt-translator-macos-arm64.dmg`,
    icon: Apple,
  },
  {
    label: "macOS Intel",
    href: `${releaseBase}/ppt-translator-macos-x64.dmg`,
    icon: Apple,
  },
  {
    label: "Windows",
    href: `${releaseBase}/ppt-translator-windows-x64-setup.exe`,
    icon: MonitorDown,
  },
];

export default function Home() {
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
        <p className="mt-5 max-w-2xl text-base leading-7 text-muted-foreground sm:text-lg">
          웹 번역 서비스는 종료되었습니다. 최신 버전은 macOS와 Windows용
          데스크톱 앱으로 배포됩니다.
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
            <a href="https://github.com/Hyunsang-coder/ppt-translator/releases">
              <Download className="h-4 w-4" />
              전체 릴리스
            </a>
          </Button>
        </div>
      </section>
    </main>
  );
}
