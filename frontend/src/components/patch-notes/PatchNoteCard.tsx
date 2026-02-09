"use client";

import {
  type PatchNote,
  type ChangeType,
  changeTypeConfig,
} from "@/data/patch-notes";

interface PatchNoteCardProps {
  note: PatchNote;
  index: number;
}

const typeOrder: ChangeType[] = ["feature", "improvement", "fix"];

export function PatchNoteCard({ note, index }: PatchNoteCardProps) {
  const grouped = typeOrder
    .map((type) => ({
      type,
      config: changeTypeConfig[type],
      items: note.changes.filter((c) => c.type === type),
    }))
    .filter((g) => g.items.length > 0);

  const formattedDate = new Date(note.date + "T00:00:00").toLocaleDateString("ko-KR", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  return (
    <div
      className="relative flex gap-4 md:gap-6 animate-slide-up opacity-0"
      style={{ animationDelay: `${index * 0.1}s`, animationFillMode: "forwards" }}
    >
      {/* Version circle - desktop */}
      <div className="hidden md:flex shrink-0 w-14 h-14 rounded-full brand-gradient items-center justify-center z-10 flex-col">
        <span className="text-[10px] font-bold text-primary-foreground leading-none">{note.version.slice(0, 4)}</span>
        <span className="text-xs font-bold text-primary-foreground leading-none">{note.version.slice(4)}</span>
      </div>

      {/* Card */}
      <div className="flex-1 glass-card p-5 space-y-4">
        {/* Header */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Version pill - mobile */}
          <span className="md:hidden inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold brand-gradient text-primary-foreground">
            {note.version}
          </span>
          <h3 className="text-lg font-semibold text-foreground">{note.title}</h3>
        </div>
        <div className="flex items-center gap-3 text-sm text-muted-foreground">
          <span>{formattedDate}</span>
          <span className="font-mono text-xs bg-muted px-2 py-0.5 rounded">
            {note.commitHash}
          </span>
        </div>

        {/* Changes grouped by type */}
        <div className="space-y-3">
          {grouped.map(({ type, config, items }) => {
            const Icon = config.icon;
            return (
              <div key={type} className="space-y-1.5">
                <div className="flex items-center gap-1.5">
                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium ${config.colorClass} ${config.bgClass}`}
                  >
                    <Icon className="w-3 h-3" />
                    {config.label}
                  </span>
                </div>
                <ul className="space-y-1 pl-1">
                  {items.map((item, i) => (
                    <li
                      key={i}
                      className="text-sm text-muted-foreground flex items-start gap-2"
                    >
                      <span className="mt-2 w-1 h-1 rounded-full bg-muted-foreground/40 shrink-0" />
                      {item.description}
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
