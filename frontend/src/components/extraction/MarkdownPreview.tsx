"use client";

interface MarkdownPreviewProps {
  markdown: string | null;
}

export function MarkdownPreview({ markdown }: MarkdownPreviewProps) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">미리보기</label>
      <div className="max-h-[300px] overflow-auto bg-muted/30 border border-border rounded-lg p-3">
        {markdown ? (
          <pre className="whitespace-pre-wrap font-mono text-sm">{markdown}</pre>
        ) : (
          <p className="text-muted-foreground text-center py-4 text-sm">
            PPT 파일을 업로드하고 변환 버튼을 클릭하세요.
          </p>
        )}
      </div>
    </div>
  );
}
