"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface MarkdownPreviewProps {
  markdown: string | null;
}

export function MarkdownPreview({ markdown }: MarkdownPreviewProps) {
  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>미리보기</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[500px] overflow-auto bg-muted/30 rounded p-4">
          {markdown ? (
            <pre className="whitespace-pre-wrap font-mono text-sm">{markdown}</pre>
          ) : (
            <p className="text-muted-foreground text-center py-8">
              PPT 파일을 업로드하고 변환 버튼을 클릭하세요.
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
