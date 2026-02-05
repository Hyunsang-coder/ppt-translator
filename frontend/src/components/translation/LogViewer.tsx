"use client";

import { useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollText, Trash2, Info, CheckCircle, XCircle, AlertTriangle } from "lucide-react";
import type { LogEntry } from "@/stores/translation-store";

interface LogViewerProps {
  logs: LogEntry[];
  onClear?: () => void;
}

const LOG_TYPE_STYLES: Record<LogEntry["type"], {
  textColor: string;
  borderColor: string;
  bgColor: string;
  icon: React.ReactNode;
}> = {
  info: {
    textColor: "text-foreground",
    borderColor: "border-l-foreground",
    bgColor: "",
    icon: <Info className="w-3.5 h-3.5 text-foreground" />,
  },
  success: {
    textColor: "text-foreground",
    borderColor: "border-l-foreground",
    bgColor: "",
    icon: <CheckCircle className="w-3.5 h-3.5 text-foreground" />,
  },
  error: {
    textColor: "text-destructive",
    borderColor: "border-l-destructive",
    bgColor: "",
    icon: <XCircle className="w-3.5 h-3.5 text-destructive" />,
  },
  warning: {
    textColor: "text-foreground",
    borderColor: "border-l-foreground",
    bgColor: "",
    icon: <AlertTriangle className="w-3.5 h-3.5 text-foreground" />,
  },
};

export function LogViewer({ logs, onClear }: LogViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

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
    <Card className="border-border overflow-hidden">
      <CardHeader className="pb-2 flex flex-row items-center justify-between">
        <CardTitle className="text-lg flex items-center gap-2">
          <ScrollText className="w-5 h-5 text-foreground" />
          <span>로그</span>
          {logs.length > 0 && (
            <span className="text-xs font-normal text-foreground/60 ml-2">
              ({logs.length})
            </span>
          )}
        </CardTitle>
        {onClear && logs.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onClear}
            className="h-8 px-2 hover:bg-destructive/10 hover:text-destructive"
          >
            <Trash2 className="w-4 h-4 mr-1" />
            지우기
          </Button>
        )}
      </CardHeader>
      <CardContent>
        <div
          ref={containerRef}
          className="h-48 overflow-y-auto font-mono text-xs rounded-lg border border-border p-2 space-y-1"
        >
          {logs.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-foreground/60">
              <ScrollText className="w-8 h-8 mb-2 opacity-50" />
              <p>로그가 없습니다.</p>
            </div>
          ) : (
            logs.map((log, index) => {
              const style = LOG_TYPE_STYLES[log.type];
              return (
                <div
                  key={log.id}
                  className={`
                    log-entry flex items-start gap-2 p-2 rounded-md border-l-2
                    ${style.borderColor} ${style.bgColor}
                  `}
                  style={{ animationDelay: `${Math.min(index * 0.05, 0.3)}s` }}
                >
                  <span className="flex-shrink-0 mt-0.5">{style.icon}</span>
                  <span className="text-foreground/60 flex-shrink-0 tabular-nums">
                    {formatTime(log.timestamp)}
                  </span>
                  <span className={`break-all ${style.textColor}`}>{log.message}</span>
                </div>
              );
            })
          )}
        </div>
      </CardContent>
    </Card>
  );
}
