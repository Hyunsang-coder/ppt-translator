"use client";

import { useEffect } from "react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { FileUploader } from "@/components/shared/FileUploader";
import { useConfig } from "@/hooks/useConfig";
import type { TranslationSettings } from "@/types/api";

interface SettingsPanelProps {
  settings: TranslationSettings;
  onSettingsChange: (settings: Partial<TranslationSettings>) => void;
  glossaryFile: File | null;
  onGlossaryFileChange: (file: File | null) => void;
  disabled?: boolean;
}

export function SettingsPanel({
  settings,
  onSettingsChange,
  glossaryFile,
  onGlossaryFileChange,
  disabled = false,
}: SettingsPanelProps) {
  const { languages, getModelsForProvider, isLoading, error } = useConfig();

  const providerModels = getModelsForProvider(settings.provider);

  // Reset model when provider changes
  useEffect(() => {
    if (providerModels.length > 0 && !providerModels.find((m) => m.id === settings.model)) {
      onSettingsChange({ model: providerModels[0].id });
    }
  }, [settings.provider, settings.model, providerModels, onSettingsChange]);

  if (error) {
    return (
      <div className="p-4 text-sm text-destructive bg-destructive/10 rounded-lg">
        설정을 불러오는데 실패했습니다: {error}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        {/* Source Language */}
        <div className="space-y-2">
          <Label htmlFor="source-lang">소스 언어</Label>
          <Select
            value={settings.sourceLang}
            onValueChange={(value) => onSettingsChange({ sourceLang: value })}
            disabled={disabled || isLoading}
          >
            <SelectTrigger id="source-lang">
              <SelectValue placeholder="언어 선택" />
            </SelectTrigger>
            <SelectContent>
              {languages.map((lang) => (
                <SelectItem key={lang.code} value={lang.code}>
                  {lang.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Target Language */}
        <div className="space-y-2">
          <Label htmlFor="target-lang">타겟 언어</Label>
          <Select
            value={settings.targetLang}
            onValueChange={(value) => onSettingsChange({ targetLang: value })}
            disabled={disabled || isLoading}
          >
            <SelectTrigger id="target-lang">
              <SelectValue placeholder="언어 선택" />
            </SelectTrigger>
            <SelectContent>
              {languages.map((lang) => (
                <SelectItem key={lang.code} value={lang.code}>
                  {lang.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {/* Provider */}
        <div className="space-y-2">
          <Label htmlFor="provider">Provider</Label>
          <Select
            value={settings.provider}
            onValueChange={(value) => onSettingsChange({ provider: value })}
            disabled={disabled || isLoading}
          >
            <SelectTrigger id="provider">
              <SelectValue placeholder="Provider 선택" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="openai">OpenAI</SelectItem>
              <SelectItem value="anthropic">Anthropic</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Model */}
        <div className="space-y-2">
          <Label htmlFor="model">모델</Label>
          <Select
            value={settings.model}
            onValueChange={(value) => onSettingsChange({ model: value })}
            disabled={disabled || isLoading || providerModels.length === 0}
          >
            <SelectTrigger id="model">
              <SelectValue placeholder="모델 선택" />
            </SelectTrigger>
            <SelectContent>
              {providerModels.map((model) => (
                <SelectItem key={model.id} value={model.id}>
                  {model.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Preprocess Repetitions */}
      <div className="flex items-center space-x-2">
        <Checkbox
          id="preprocess"
          checked={settings.preprocessRepetitions}
          onCheckedChange={(checked) =>
            onSettingsChange({ preprocessRepetitions: checked === true })
          }
          disabled={disabled}
        />
        <Label htmlFor="preprocess" className="text-sm font-normal cursor-pointer">
          반복 문구 전처리 (동일 텍스트 중복 번역 방지)
        </Label>
      </div>

      {/* User Prompt */}
      <div className="space-y-1.5">
        <Label htmlFor="user-prompt">사용자 프롬프트 (선택)</Label>
        <Textarea
          id="user-prompt"
          placeholder="번역 시 참고할 추가 지침을 입력하세요..."
          value={settings.userPrompt}
          onChange={(e) => onSettingsChange({ userPrompt: e.target.value })}
          disabled={disabled}
          rows={2}
          className="resize-none"
        />
      </div>

      {/* Glossary File */}
      <FileUploader
        label="용어집 파일 (선택)"
        description="Excel 파일 (.xlsx, .xls)"
        accept={{
          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
          "application/vnd.ms-excel": [".xls"],
        }}
        maxSizeMB={10}
        selectedFile={glossaryFile}
        onFileSelect={onGlossaryFileChange}
        disabled={disabled}
      />
    </div>
  );
}
