"use client";

import Link from "next/link";
import {
  FileSearch,
  Globe,
  BookOpen,
  Copy,
  Languages,
  Layers,
  Brain,
  Expand,
  BookCheck,
  Palette,
  FileOutput,
} from "lucide-react";
import { PipelineStep, type PipelinePhase } from "./PipelineStep";
import { Button } from "@/components/ui/button";

const phases: PipelinePhase[] = [
  {
    number: 1,
    name: "PPT 파싱",
    nameEn: "PPT Parsing",
    description: "PPTX 파일에서 모든 도형, 표, 그룹의 텍스트를 추출합니다",
    icon: FileSearch,
    percentRange: "0-2%",
    group: "preparation",
  },
  {
    number: 2,
    name: "글로벌 컨텍스트",
    nameEn: "Global Context",
    description: "프레젠테이션 전체 맥락을 분석하여 일관된 번역을 준비합니다",
    icon: Globe,
    percentRange: "2-5%",
    group: "preparation",
  },
  {
    number: 3,
    name: "용어집 적용",
    nameEn: "Glossary Application",
    description: "용어집에 등록된 전문 용어를 번역 전에 미리 적용합니다",
    icon: BookOpen,
    percentRange: "5-8%",
    group: "preparation",
  },
  {
    number: 4,
    name: "중복 제거",
    nameEn: "Deduplication",
    description: "동일한 문장을 하나로 통합하여 API 호출 횟수를 줄입니다",
    icon: Copy,
    percentRange: "8%",
    group: "translation",
  },
  {
    number: 5,
    name: "언어 감지",
    nameEn: "Language Detection",
    description: "소스 언어를 자동으로 감지합니다",
    icon: Languages,
    percentRange: "5%",
    group: "translation",
  },
  {
    number: 6,
    name: "배치 분할",
    nameEn: "Batch Splitting",
    description: "문단을 적절한 크기의 배치로 나누어 병렬 처리를 준비합니다",
    icon: Layers,
    percentRange: "8%",
    group: "translation",
  },
  {
    number: 7,
    name: "LLM 번역",
    nameEn: "LLM Translation",
    description: "GPT/Claude에게 구조화된 출력으로 번역을 요청합니다",
    icon: Brain,
    percentRange: "10-80%",
    group: "translation",
  },
  {
    number: 8,
    name: "중복 복원",
    nameEn: "Dedup Expansion",
    description: "통합했던 중복 문장에 번역 결과를 다시 매핑합니다",
    icon: Expand,
    percentRange: "80%",
    group: "translation",
  },
  {
    number: 9,
    name: "용어집 보정",
    nameEn: "Glossary Correction",
    description: "번역 후 용어집 준수 여부를 재확인하고 보정합니다",
    icon: BookCheck,
    percentRange: "80-90%",
    group: "finalization",
  },
  {
    number: 10,
    name: "색상 분배",
    nameEn: "Color Distribution",
    description: "다색 서식의 원본 스타일을 번역문에 정확히 매핑합니다",
    icon: Palette,
    percentRange: "80-90%",
    group: "finalization",
  },
  {
    number: 11,
    name: "PPT 적용",
    nameEn: "PPT Write-back",
    description: "번역문을 원본 서식 그대로 PPT에 기록하고 텍스트 피팅을 적용합니다",
    icon: FileOutput,
    percentRange: "90-100%",
    group: "finalization",
  },
];

const groups = [
  { key: "preparation" as const, label: "준비", range: "Phase 1-3" },
  { key: "translation" as const, label: "번역", range: "Phase 4-8" },
  { key: "finalization" as const, label: "마무리", range: "Phase 9-11" },
];

export function PipelineTimeline() {
  return (
    <div className="max-w-3xl mx-auto space-y-12">
      {/* Hero */}
      <div className="text-center space-y-3 animate-slide-up">
        <h2 className="text-3xl sm:text-4xl font-bold brand-gradient-text">
          번역 파이프라인
        </h2>
        <p className="text-muted-foreground text-base sm:text-lg">
          PPT 번역캣이 파일을 번역하는 11단계 과정을 살펴보세요
        </p>
      </div>

      {/* Groups */}
      {groups.map((group) => {
        const groupPhases = phases.filter((p) => p.group === group.key);
        return (
          <section key={group.key} className="space-y-4">
            {/* Group header */}
            <div className="glass-card px-4 py-2 inline-flex items-center gap-2">
              <span className="font-semibold text-foreground">{group.label}</span>
              <span className="text-xs text-muted-foreground">{group.range}</span>
            </div>

            {/* Timeline */}
            <div className="relative space-y-4">
              {/* Vertical line - desktop only */}
              <div className="hidden md:block absolute left-6 top-6 bottom-6 w-px bg-border" />

              {groupPhases.map((phase, i) => (
                <PipelineStep
                  key={phase.number}
                  phase={phase}
                  index={phases.indexOf(phase)}
                />
              ))}
            </div>
          </section>
        );
      })}

      {/* CTA */}
      <div className="text-center pt-4 animate-slide-up" style={{ animationDelay: "1.2s", animationFillMode: "forwards", opacity: 0 }}>
        <Button asChild className="btn-gradient px-8 py-3 text-base font-semibold rounded-xl">
          <Link href="/translate">번역 시작하기</Link>
        </Button>
      </div>
    </div>
  );
}
