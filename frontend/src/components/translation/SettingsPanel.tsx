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
import { Button } from "@/components/ui/button";
import { FileUploader } from "@/components/shared/FileUploader";
import { useConfig } from "@/hooks/useConfig";
import type { TranslationSettings, FilenameSettings } from "@/types/api";
import { Sparkles, Loader2, FileText } from "lucide-react";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";

interface SettingsPanelProps {
  settings: TranslationSettings;
  onSettingsChange: (settings: Partial<TranslationSettings>) => void;
  glossaryFile: File | null;
  onGlossaryFileChange: (file: File | null) => void;
  pptFile: File | null;
  generatedContext: string;
  generatedInstructions: string;
  isGeneratingContext: boolean;
  isGeneratingInstructions: boolean;
  onGenerateContext: () => void;
  onGenerateInstructions: () => void;
  onContextChange: (context: string) => void;
  onInstructionsChange: (instructions: string) => void;
  disabled?: boolean;
}

export function SettingsPanel({
  settings,
  onSettingsChange,
  glossaryFile,
  onGlossaryFileChange,
  pptFile,
  generatedContext,
  generatedInstructions,
  isGeneratingContext,
  isGeneratingInstructions,
  onGenerateContext,
  onGenerateInstructions,
  onContextChange,
  onInstructionsChange,
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

        {/* Target Language - Auto 제외 (타겟 언어는 필수 선택) */}
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
              {languages
                .filter((lang) => lang.code !== "Auto")
                .map((lang) => (
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

      {/* Translate Speaker Notes */}
      <div className="flex items-center space-x-2">
        <Checkbox
          id="translate-notes"
          checked={settings.translateNotes}
          onCheckedChange={(checked) =>
            onSettingsChange({ translateNotes: checked === true })
          }
          disabled={disabled}
        />
        <Label htmlFor="translate-notes" className="text-sm font-normal cursor-pointer">
          슬라이드 노트 번역
        </Label>
      </div>

      {/* Context (Background Information) */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label htmlFor="context">컨텍스트 (배경 정보)</Label>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-muted-foreground">
              using {settings.provider === "openai" ? "GPT-5 Mini" : "Haiku 4.5"}
            </span>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onGenerateContext}
              disabled={disabled || isGeneratingContext || !pptFile}
              className="h-7 text-xs gap-1.5"
            >
              {isGeneratingContext ? (
                <>
                  <Loader2 className="w-3 h-3 animate-spin" />
                  생성 중...
                </>
              ) : (
                <>
                  <Sparkles className="w-3 h-3" />
                  자동 생성
                </>
              )}
            </Button>
          </div>
        </div>
        <Textarea
          id="context"
          placeholder="문서의 주제, 도메인, 대상 독자 등 배경 정보를 입력하세요."
          value={generatedContext || settings.context}
          onChange={(e) => {
            if (generatedContext) {
              onContextChange(e.target.value);
            } else {
              onSettingsChange({ context: e.target.value });
            }
          }}
          disabled={disabled || isGeneratingContext}
          rows={3}
          className="resize-none"
        />
        {generatedContext && (
          <p className="text-xs text-muted-foreground">
            * 자동 생성된 컨텍스트입니다. 필요에 따라 수정할 수 있습니다.
          </p>
        )}
      </div>

      {/* Instructions (Style/Tone) */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label htmlFor="instructions">번역 지침 (스타일/톤)</Label>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-muted-foreground">
              using {settings.provider === "openai" ? "GPT-5 Mini" : "Haiku 4.5"}
            </span>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onGenerateInstructions}
              disabled={disabled || isGeneratingInstructions || !pptFile || !settings.targetLang}
              className="h-7 text-xs gap-1.5"
            >
              {isGeneratingInstructions ? (
                <>
                  <Loader2 className="w-3 h-3 animate-spin" />
                  생성 중...
                </>
              ) : (
                <>
                  <Sparkles className="w-3 h-3" />
                  자동 생성
                </>
              )}
            </Button>
          </div>
        </div>
        <Textarea
          id="instructions"
          placeholder="격식체/비격식체, 직역/의역 선호, 특정 용어 처리 방식 등을 입력하세요."
          value={generatedInstructions || settings.instructions}
          onChange={(e) => {
            if (generatedInstructions) {
              onInstructionsChange(e.target.value);
            } else {
              onSettingsChange({ instructions: e.target.value });
            }
          }}
          disabled={disabled || isGeneratingInstructions}
          rows={3}
          className="resize-none"
        />
        {generatedInstructions && (
          <p className="text-xs text-muted-foreground">
            * 자동 생성된 지침입니다. 필요에 따라 수정할 수 있습니다.
          </p>
        )}
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

export interface FilenameSettingsSectionProps {
  settings: FilenameSettings;
  onChange: (settings: FilenameSettings) => void;
  pptFile: File | null;
  targetLang: string;
  modelName: string;
  disabled?: boolean;
}

// Language code mapping for filenames
const LANGUAGE_CODE_MAP: Record<string, string> = {
  "한국어": "KR",
  "영어": "EN",
  "일본어": "JP",
  "중국어": "CN",
  "스페인어": "ES",
  "프랑스어": "FR",
  "독일어": "DE",
};

export function FilenameSettingsSection({
  settings,
  onChange,
  pptFile,
  targetLang,
  modelName,
  disabled = false,
}: FilenameSettingsSectionProps) {
  const originalName = pptFile?.name.replace(/\.[^/.]+$/, "") || "presentation";
  const today = new Date().toISOString().slice(0, 10).replace(/-/g, "");

  const generatePreview = () => {
    if (settings.mode === "custom") {
      const customName = settings.customName.trim() || "파일명을 입력하세요";
      return `${customName}.pptx`;
    }

    const parts: string[] = [];
    if (settings.includeLanguage && targetLang) {
      const langCode = LANGUAGE_CODE_MAP[targetLang] || targetLang;
      parts.push(langCode);
    }
    if (settings.includeOriginalName) parts.push(originalName);
    if (settings.includeModel) parts.push(modelName);
    if (settings.includeDate) parts.push(today);

    if (parts.length === 0) {
      return "translated.pptx";
    }
    return `${parts.join("_")}.pptx`;
  };

  const updateSetting = <K extends keyof FilenameSettings>(
    key: K,
    value: FilenameSettings[K]
  ) => {
    onChange({ ...settings, [key]: value });
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <FileText className="w-4 h-4 text-muted-foreground" />
        <Label>출력 파일명 설정</Label>
      </div>

      <RadioGroup
        value={settings.mode}
        onValueChange={(value) => updateSetting("mode", value as "auto" | "custom")}
        disabled={disabled}
        className="space-y-3"
      >
        {/* Auto mode */}
        <div className="space-y-2">
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="auto" id="filename-auto" />
            <Label htmlFor="filename-auto" className="font-normal cursor-pointer">
              자동 생성
            </Label>
          </div>

          {settings.mode === "auto" && (
            <div className="ml-6 space-y-2">
              <div className="grid grid-cols-2 gap-2">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="include-language"
                    checked={settings.includeLanguage}
                    onCheckedChange={(checked) =>
                      updateSetting("includeLanguage", checked === true)
                    }
                    disabled={disabled}
                  />
                  <Label htmlFor="include-language" className="text-sm font-normal cursor-pointer">
                    대상 언어
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="include-original"
                    checked={settings.includeOriginalName}
                    onCheckedChange={(checked) =>
                      updateSetting("includeOriginalName", checked === true)
                    }
                    disabled={disabled}
                  />
                  <Label htmlFor="include-original" className="text-sm font-normal cursor-pointer">
                    원본 파일명
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="include-model"
                    checked={settings.includeModel}
                    onCheckedChange={(checked) =>
                      updateSetting("includeModel", checked === true)
                    }
                    disabled={disabled}
                  />
                  <Label htmlFor="include-model" className="text-sm font-normal cursor-pointer">
                    모델명
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="include-date"
                    checked={settings.includeDate}
                    onCheckedChange={(checked) =>
                      updateSetting("includeDate", checked === true)
                    }
                    disabled={disabled}
                  />
                  <Label htmlFor="include-date" className="text-sm font-normal cursor-pointer">
                    날짜
                  </Label>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Custom mode */}
        <div className="space-y-2">
          <div className="flex items-center space-x-2">
            <RadioGroupItem value="custom" id="filename-custom" />
            <Label htmlFor="filename-custom" className="font-normal cursor-pointer">
              직접 입력
            </Label>
          </div>

          {settings.mode === "custom" && (
            <div className="ml-6">
              <Input
                placeholder="파일명 입력 (확장자 제외)"
                value={settings.customName}
                onChange={(e) => updateSetting("customName", e.target.value)}
                disabled={disabled}
                className="h-8 text-sm"
              />
            </div>
          )}
        </div>
      </RadioGroup>

      {/* Preview */}
      <div className="p-2 bg-muted rounded-md border border-border">
        <p className="text-xs text-muted-foreground">
          미리보기: <span className="font-mono text-foreground">{generatePreview()}</span>
        </p>
      </div>
    </div>
  );
}
