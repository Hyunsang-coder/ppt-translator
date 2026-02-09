"use client";

import { type LucideIcon } from "lucide-react";

export interface PipelinePhase {
  number: number;
  name: string;
  nameEn: string;
  description: string;
  icon: LucideIcon;
  percentRange: string;
  group: "preparation" | "translation" | "finalization";
}

interface PipelineStepProps {
  phase: PipelinePhase;
  index: number;
}

export function PipelineStep({ phase, index }: PipelineStepProps) {
  const Icon = phase.icon;

  return (
    <div
      className="relative flex items-start gap-4 md:gap-6 animate-slide-up opacity-0"
      style={{ animationDelay: `${index * 0.1}s`, animationFillMode: "forwards" }}
    >
      {/* Number circle - desktop only */}
      <div className="hidden md:flex w-12 h-12 rounded-full brand-gradient items-center justify-center text-white dark:text-black font-bold text-lg shrink-0 shadow-md z-10">
        {phase.number}
      </div>

      {/* Card */}
      <div className="glass-card p-4 flex-1">
        <div className="flex items-start gap-3">
          {/* Mobile number badge */}
          <div className="flex md:hidden w-8 h-8 rounded-full brand-gradient items-center justify-center text-white dark:text-black font-bold text-sm shrink-0">
            {phase.number}
          </div>

          {/* Icon */}
          <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
            <Icon className="w-5 h-5 text-primary" />
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-baseline gap-2 flex-wrap">
              <h3 className="font-semibold text-foreground">{phase.name}</h3>
              <span className="text-xs font-mono text-muted-foreground">{phase.percentRange}</span>
            </div>
            <p className="text-sm text-muted-foreground mt-1">{phase.description}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
