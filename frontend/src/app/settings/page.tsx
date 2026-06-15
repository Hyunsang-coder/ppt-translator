"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
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
  restartSidecar,
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

  const [busy, setBusy] = useState(false);

  const handleSave = async (provider: Provider) => {
    const key = values[provider].trim();
    if (!key) {
      toast.error("키를 입력해주세요.");
      return;
    }
    setBusy(true);
    const tid = toast.loading("키를 저장하고 번역 엔진에 적용하는 중…");
    try {
      await saveApiKey(provider, key);
      setSaved((s) => ({ ...s, [provider]: true }));
      setValues((v) => ({ ...v, [provider]: "" }));
      await restartSidecar();
      toast.success("키를 저장하고 적용했습니다. 이제 번역할 수 있습니다.", { id: tid });
    } catch (e) {
      toast.error(`저장 실패: ${e instanceof Error ? e.message : String(e)}`, { id: tid });
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (provider: Provider) => {
    setBusy(true);
    const tid = toast.loading("키를 삭제하고 적용하는 중…");
    try {
      await deleteApiKey(provider);
      setSaved((s) => ({ ...s, [provider]: false }));
      await restartSidecar();
      toast.success("키를 삭제하고 적용했습니다.", { id: tid });
    } catch (e) {
      toast.error(`삭제 실패: ${e instanceof Error ? e.message : String(e)}`, { id: tid });
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="mx-auto max-w-xl space-y-6 p-8">
      <Link
        href="/translate"
        className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" />
        번역으로 돌아가기
      </Link>
      <div>
        <h1 className="text-2xl font-semibold">설정</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          API 키는 이 기기의 보안 저장소(macOS Keychain / Windows 자격 증명 관리자)에
          암호화되어 저장됩니다. 저장하면 번역 엔진에 자동으로 적용됩니다.
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
                type="text"
                autoComplete="off"
                autoCorrect="off"
                autoCapitalize="off"
                spellCheck={false}
                data-1p-ignore
                data-lpignore="true"
                name={`apikey-${p.id}`}
                placeholder={p.placeholder}
                value={values[p.id]}
                onChange={(e) =>
                  setValues((v) => ({ ...v, [p.id]: e.target.value }))
                }
              />
              {values[p.id] && !values[p.id].startsWith("sk-") && (
                <p className="text-xs text-amber-600">
                  키가 보통 &quot;sk-&quot;로 시작합니다. 첫 글자가 빠지지
                  않았는지 확인해주세요.
                </p>
              )}
            </div>
            <div className="flex gap-2">
              <Button onClick={() => handleSave(p.id)} disabled={busy}>
                저장
              </Button>
              {saved[p.id] && (
                <Button
                  variant="outline"
                  onClick={() => handleDelete(p.id)}
                  disabled={busy}
                >
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
