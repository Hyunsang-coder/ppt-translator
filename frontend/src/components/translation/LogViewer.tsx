"use client";

import { useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import type { LogEntry } from "@/stores/translation-store";

interface LogViewerProps {
  logs: LogEntry[];
  onClear?: () => void;
}

const LOG_TYPE_COLORS: Record<LogEntry["type"], string> = {
  info: "text-foreground",
  success: "text-green-600 dark:text-green-400",
  error: "text-destructive",
  warning: "text-yellow-600 dark:text-yellow-400",
};

const LOG_TYPE_ICONS: Record<LogEntry["type"], string> = {
  info: "ℹ️",
  success: "✅",
  error: "❌",
  warning: "⚠️",
};

export function LogViewer({ logs, onClear }: LogViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new logs
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  const formatTime = (date: Date): string => {
    return date.toLocaleTimeString("ko-KR", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  return (
    <Card>
      <CardHeader className="pb-2 flex flex-row items-center justify-between">
        <CardTitle className="text-lg">로그</CardTitle>
        {onClear && logs.length > 0 && (
          <Button variant="ghost" size="sm" onClick={onClear}>
            지우기
          </Button>
        )}
      </CardHeader>
      <CardContent>
        <div
          ref={containerRef}
          className="h-48 overflow-y-auto font-mono text-xs bg-muted/30 rounded p-2 space-y-1"
        >
          {logs.length === 0 ? (
            <p className="text-muted-foreground text-center py-4">로그가 없습니다.</p>
          ) : (
            logs.map((log) => (
              <div key={log.id} className={`flex gap-2 ${LOG_TYPE_COLORS[log.type]}`}>
                <span className="flex-shrink-0">{LOG_TYPE_ICONS[log.type]}</span>
                <span className="text-muted-foreground flex-shrink-0">
                  [{formatTime(log.timestamp)}]
                </span>
                <span className="break-all">{log.message}</span>
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
}
