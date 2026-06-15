"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { isTauri } from "@/lib/api-base";
import {
  deleteApiKey,
  hasApiKey,
  saveApiKey,
  type Provider,
} from "@/lib/keychain";

const PROVIDERS: { id: Provider; label: string; placeholder: string }[] = [
  { id: "openai", label: "OpenAI API Key", placeholder: "sk-..." },
  { id: "anthropic", label: "Anthropic API Key", placeholder: "sk-ant-..." },
];

export default function SettingsPage() {
  const [values, setValues] = useState<Record<Provider, string>>({
    openai: "",
    anthropic: "",
  });
  const [saved, setSaved] = useState<Record<Provider, boolean>>({
    openai: false,
    anthropic: false,
  });

  useEffect(() => {
    if (!isTauri()) return;
    (async () => {
      setSaved({
        openai: await hasApiKey("openai"),
        anthropic: await hasApiKey("anthropic"),
      });
    })();
  }, []);

  if (!isTauri()) {
    return (
      <main className="mx-auto max-w-xl p-8">
        <h1 className="mb-4 text-2xl font-semibold">설정</h1>
        <p className="text-muted-foreground">
          API 키 설정은 데스크톱 앱에서만 사용할 수 있습니다.
        </p>
      </main>
    );
  }

  const handleSave = async (provider: Provider) => {
    const key = values[provider].trim();
    if (!key) {
      toast.error("키를 입력해주세요.");
      return;
    }
    try {
      await saveApiKey(provider, key);
      setSaved((s) => ({ ...s, [provider]: true }));
      setValues((v) => ({ ...v, [provider]: "" }));
      toast.success("키를 저장했습니다. 적용하려면 앱을 재시작해주세요.");
    } catch (e) {
      toast.error(`저장 실패: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const handleDelete = async (provider: Provider) => {
    try {
      await deleteApiKey(provider);
      setSaved((s) => ({ ...s, [provider]: false }));
      toast.success("키를 삭제했습니다. 적용하려면 앱을 재시작해주세요.");
    } catch (e) {
      toast.error(`삭제 실패: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  return (
    <main className="mx-auto max-w-xl space-y-6 p-8">
      <div>
        <h1 className="text-2xl font-semibold">설정</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          API 키는 이 기기의 보안 저장소(macOS Keychain / Windows 자격 증명 관리자)에
          암호화되어 저장됩니다. 키 변경 후에는 앱을 재시작해야 적용됩니다.
        </p>
      </div>

      {PROVIDERS.map((p) => (
        <Card key={p.id}>
          <CardHeader>
            <CardTitle className="text-base">{p.label}</CardTitle>
            <CardDescription>
              {saved[p.id] ? "저장됨 ✓" : "저장된 키 없음"}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="space-y-1.5">
              <Label htmlFor={p.id}>새 키 입력</Label>
              <Input
                id={p.id}
                type="password"
                autoComplete="off"
                placeholder={p.placeholder}
                value={values[p.id]}
                onChange={(e) =>
                  setValues((v) => ({ ...v, [p.id]: e.target.value }))
                }
              />
            </div>
            <div className="flex gap-2">
              <Button onClick={() => handleSave(p.id)}>저장</Button>
              {saved[p.id] && (
                <Button variant="outline" onClick={() => handleDelete(p.id)}>
                  삭제
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      ))}
    </main>
  );
}
