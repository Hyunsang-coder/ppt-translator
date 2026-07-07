"use client";

interface MarkdownPreviewProps {
  markdown: string | null;
}

export function MarkdownPreview({ markdown }: MarkdownPreviewProps) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">미리보기</label>
      {/* 이 페이지의 주인공은 추출 결과: 고정 300px 대신 화면 잔여 공간을 채운다 */}
      <div className="h-[calc(100vh-400px)] min-h-[320px] overflow-auto bg-muted/30 border border-border rounded-lg p-3">
        {markdown ? (
          <pre className="whitespace-pre-wrap font-mono text-sm">{markdown}</pre>
        ) : (
          <div className="h-full flex items-center justify-center">
            <p className="text-muted-foreground text-center text-sm">
              PPT 파일을 업로드하고 변환 버튼을 클릭하세요.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
