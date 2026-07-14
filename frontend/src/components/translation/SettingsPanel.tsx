"use client";

import { useEffect, useState } from "react";
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
import type { TranslationSettings, FilenameSettings, TextFitMode, ImageCompression, LengthLimit } from "@/types/api";
import { FileText, Type, ImageDown, ChevronDown, Info, Settings2 } from "lucide-react";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";

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
  const { languages, config, getModelsForProvider, isLoading, error } = useConfig();
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const providerModels = getModelsForProvider(settings.provider);

  // Pick a valid model whenever the current one isn't in the provider's list
  // (initial empty model, or after a provider switch). A-2: the store no longer
  // hard-codes a default model — prefer the backend's default_model, else the
  // first model the backend advertises for this provider.
  useEffect(() => {
    if (providerModels.length > 0 && !providerModels.find((m) => m.id === settings.model)) {
      const backendDefault = providerModels.find((m) => m.id === config?.default_model);
      onSettingsChange({ model: (backendDefault ?? providerModels[0]).id });
    }
  }, [settings.provider, settings.model, providerModels, config, onSettingsChange]);

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

        {/* Target Language - Auto 제외 (대상 언어는 필수 선택) */}
        <div className="space-y-2">
          <Label htmlFor="target-lang">
            대상 언어 <span className="text-destructive">*</span>
          </Label>
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
          <Label htmlFor="provider">AI 제공자</Label>
          <Select
            value={settings.provider}
            onValueChange={(value) => onSettingsChange({ provider: value })}
            disabled={disabled || isLoading}
          >
            <SelectTrigger id="provider">
              <SelectValue placeholder="제공자 선택" />
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

      {/* Advanced Options - 선택 설정은 아코디언으로 접어 첫 화면을 가볍게 유지 */}
      <div className="rounded-xl border border-border">
        <button
          type="button"
          onClick={() => setAdvancedOpen((open) => !open)}
          aria-expanded={advancedOpen}
          className="w-full flex items-center gap-2 px-4 py-3 text-sm font-medium cursor-pointer"
        >
          <Settings2 className="w-4 h-4 text-muted-foreground" />
          고급 옵션
          <span className="text-xs font-normal text-muted-foreground">
            번역 · 스타일 · 이미지 압축 · 컨텍스트
          </span>
          <ChevronDown
            className={`w-4 h-4 ml-auto text-muted-foreground transition-transform ${
              advancedOpen ? "rotate-180" : ""
            }`}
          />
        </button>
        {advancedOpen && (
          <TooltipProvider delayDuration={300}>
            <div className="px-4 pb-4 space-y-4">
      {/* Translation Options */}
      <div className="space-y-3">
        <Label className="text-sm font-medium">번역 옵션</Label>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
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
              반복 문구 전처리
            </Label>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent>
                동일 텍스트를 한 번만 번역해 속도를 높이고 용어 일관성을 유지합니다.
              </TooltipContent>
            </Tooltip>
          </div>

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
        </div>
      </div>

      {/* Style Options */}
      <div className="space-y-3">
        <Label className="text-sm font-medium">스타일 옵션</Label>
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
            <div className="flex items-center gap-3">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="text-fit-shrink"
                  checked={settings.textFitMode === "auto_shrink" || settings.textFitMode === "shrink_then_expand"}
                  onCheckedChange={(checked) => {
                    const shrink = checked === true;
                    const expand = settings.textFitMode === "expand_box" || settings.textFitMode === "shrink_then_expand";
                    const mode: TextFitMode = shrink && expand ? "shrink_then_expand" : shrink ? "auto_shrink" : expand ? "expand_box" : "none";
                    onSettingsChange({ textFitMode: mode });
                  }}
                  disabled={disabled}
                />
                <Label htmlFor="text-fit-shrink" className="text-sm font-normal cursor-pointer">
                  폰트 자동 축소
                </Label>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Info className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent>
                    번역문이 길어지면 글자 크기를 줄여 텍스트 박스 안에 맞춥니다.
                  </TooltipContent>
                </Tooltip>
              </div>
              {(settings.textFitMode === "auto_shrink" || settings.textFitMode === "shrink_then_expand") && (
                <Select
                  value={String(settings.minFontRatio)}
                  onValueChange={(value) => onSettingsChange({ minFontRatio: Number(value) })}
                  disabled={disabled}
                >
                  <SelectTrigger id="min-font-ratio" className="w-[140px] h-8 text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="90">90%</SelectItem>
                    <SelectItem value="80">80% (기본)</SelectItem>
                    <SelectItem value="70">70%</SelectItem>
                  </SelectContent>
                </Select>
              )}
            </div>
            <div className="flex items-center space-x-2">
                  <Checkbox
                    id="text-fit-expand"
                    checked={settings.textFitMode === "expand_box" || settings.textFitMode === "shrink_then_expand"}
                    onCheckedChange={(checked) => {
                      const expand = checked === true;
                      const shrink = settings.textFitMode === "auto_shrink" || settings.textFitMode === "shrink_then_expand";
                      const mode: TextFitMode = shrink && expand ? "shrink_then_expand" : shrink ? "auto_shrink" : expand ? "expand_box" : "none";
                      onSettingsChange({ textFitMode: mode });
                    }}
                    disabled={disabled}
                  />
                  <Label htmlFor="text-fit-expand" className="text-sm font-normal cursor-pointer">
                    텍스트 박스 확장
                  </Label>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Info className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>
                      번역문이 길어지면 박스 너비를 넓혀 텍스트가 잘리지 않게 합니다.
                    </TooltipContent>
                  </Tooltip>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="length-limit"
                  checked={settings.lengthLimit !== null}
                  onCheckedChange={(checked) => {
                    onSettingsChange({ lengthLimit: checked === true ? 130 : null });
                  }}
                  disabled={disabled}
                />
                <Label htmlFor="length-limit" className="text-sm font-normal cursor-pointer">
                  번역 길이 가이드
                </Label>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Info className="w-3.5 h-3.5 text-muted-foreground cursor-help" />
                  </TooltipTrigger>
                  <TooltipContent>
                    AI에 항목별 최대 글자 수를 지시하고, 초과 결과는 검토 화면에 표시합니다. 문장을 강제로 잘라내지는 않습니다.
                  </TooltipContent>
                </Tooltip>
              </div>
              {settings.lengthLimit !== null && (
                <Select
                  value={String(settings.lengthLimit)}
                  onValueChange={(value) => onSettingsChange({ lengthLimit: Number(value) as LengthLimit })}
                  disabled={disabled}
                >
                  <SelectTrigger id="length-limit-value" className="w-[140px] h-8 text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="110">110%</SelectItem>
                    <SelectItem value="130">130% (기본)</SelectItem>
                    <SelectItem value="150">150%</SelectItem>
                  </SelectContent>
                </Select>
              )}
            </div>
          </div>
      </div>

      {/* Image Compression */}
      <div className="space-y-3">
        <Label className="text-sm font-medium">이미지 압축</Label>
        <div className="flex items-center gap-3">
          <div className="flex items-center space-x-2">
            <Checkbox
              id="compress-images"
              checked={settings.imageCompression !== "none"}
              onCheckedChange={(checked) =>
                onSettingsChange({ imageCompression: checked ? "medium" : "none" })
              }
              disabled={disabled}
            />
            <Label htmlFor="compress-images" className="text-sm font-normal cursor-pointer">
              압축 사용 (대용량 파일 최적화)
            </Label>
          </div>
          {settings.imageCompression !== "none" && (
            <Select
              value={settings.imageCompression}
              onValueChange={(value) =>
                onSettingsChange({ imageCompression: value as ImageCompression })
              }
              disabled={disabled}
            >
              <SelectTrigger className="w-[130px] h-8 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="high">높음 (85%)</SelectItem>
                <SelectItem value="medium">보통 (70%)</SelectItem>
                <SelectItem value="low">낮음 (50%)</SelectItem>
              </SelectContent>
            </Select>
          )}
        </div>
        <p className="text-xs text-muted-foreground">
          이미지 품질을 낮춰 메모리 사용을 줄입니다. 번역 품질에는 영향 없습니다.
        </p>
      </div>

      {/* Context (Background Information) */}
      <div className="space-y-2">
        <Label htmlFor="context">컨텍스트 (배경 정보)</Label>
        <Textarea
          id="context"
          placeholder="문서의 주제, 도메인, 대상 독자 등 배경 정보를 입력하세요."
          value={settings.context}
          onChange={(e) => onSettingsChange({ context: e.target.value })}
          disabled={disabled}
          rows={3}
          className="h-28 max-h-28 resize-none overflow-y-auto [field-sizing:fixed]"
        />
      </div>

      {/* Instructions (Style/Tone) */}
      <div className="space-y-2">
        <Label htmlFor="instructions">번역 지침 (스타일/톤)</Label>
        <Textarea
          id="instructions"
          placeholder="격식체/비격식체, 직역/의역 선호, 특정 용어 처리 방식 등을 입력하세요."
          value={settings.instructions}
          onChange={(e) => onSettingsChange({ instructions: e.target.value })}
          disabled={disabled}
          rows={3}
          className="h-28 max-h-28 resize-none overflow-y-auto [field-sizing:fixed]"
        />
      </div>
            </div>
          </TooltipProvider>
        )}
      </div>

      {/* Glossary File - PPT 드롭존과 혼동하지 않도록 컴팩트 버튼형 */}
      <FileUploader
        variant="compact"
        label="용어집 추가"
        description="선택 · Excel (.xlsx, .xls)"
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

type FilenamePartKey = "language" | "originalName" | "model" | "date";
type FilenameBooleanSettingKey =
  | "includeLanguage"
  | "includeOriginalName"
  | "includeModel"
  | "includeDate";

const FILENAME_PART_ORDER: FilenamePartKey[] = ["language", "originalName", "model", "date"];
const FILENAME_SETTING_KEY_MAP: Record<FilenamePartKey, FilenameBooleanSettingKey> = {
  language: "includeLanguage",
  originalName: "includeOriginalName",
  model: "includeModel",
  date: "includeDate",
};

const FILENAME_PART_LABELS: Record<FilenamePartKey, string> = {
  language: "대상 언어",
  originalName: "원본 파일명",
  model: "모델명",
  date: "날짜",
};

// 칩과 미리보기 세그먼트를 같은 색으로 매핑해 구성 요소를 시각적으로 연결한다
const FILENAME_PART_COLORS: Record<FilenamePartKey, string> = {
  language: "bg-blue-100 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300",
  originalName: "bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300",
  model: "bg-pink-100 text-pink-800 dark:bg-pink-950/60 dark:text-pink-300",
  date: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300",
};

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

function isFilenamePartKey(value: string): value is FilenamePartKey {
  return FILENAME_PART_ORDER.includes(value as FilenamePartKey);
}

function getEnabledFilenamePartOrder(settings: FilenameSettings): FilenamePartKey[] {
  const seen = new Set<FilenamePartKey>();
  const normalizedOrder: FilenamePartKey[] = [];
  const configuredOrder = settings.componentOrder || [];

  for (const part of configuredOrder) {
    if (isFilenamePartKey(part) && !seen.has(part)) {
      normalizedOrder.push(part);
      seen.add(part);
    }
  }
  for (const part of FILENAME_PART_ORDER) {
    if (!seen.has(part)) {
      normalizedOrder.push(part);
    }
  }

  return normalizedOrder.filter((part) => settings[FILENAME_SETTING_KEY_MAP[part]]);
}

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

  const enabledOrder = getEnabledFilenamePartOrder(settings);
  const partValues: Record<FilenamePartKey, string> = {
    language: targetLang ? LANGUAGE_CODE_MAP[targetLang] || targetLang : "",
    originalName,
    model: modelName,
    date: today,
  };
  const previewSegments = enabledOrder
    .map((part) => ({ part, text: partValues[part] }))
    .filter((segment) => segment.text);

  const updateSetting = <K extends keyof FilenameSettings>(
    key: K,
    value: FilenameSettings[K]
  ) => {
    onChange({ ...settings, [key]: value });
  };

  const updatePartSelection = (part: FilenamePartKey, checked: boolean) => {
    const settingKey = FILENAME_SETTING_KEY_MAP[part];
    const withoutPart = (settings.componentOrder || []).filter((item) => item !== part);
    const nextOrder = checked ? [...withoutPart, part] : withoutPart;

    onChange({
      ...settings,
      [settingKey]: checked,
      componentOrder: nextOrder,
    });
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
              {/* 순서 번호 칩: 선택한 순서가 곧 파일명 순서임을 드러낸다 */}
              <div className="flex flex-wrap gap-1.5">
                {FILENAME_PART_ORDER.map((part) => {
                  const enabled = settings[FILENAME_SETTING_KEY_MAP[part]];
                  const orderIndex = enabledOrder.indexOf(part);
                  return (
                    <button
                      key={part}
                      type="button"
                      disabled={disabled}
                      aria-pressed={enabled}
                      onClick={() => updatePartSelection(part, !enabled)}
                      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                        enabled
                          ? `border-transparent font-medium ${FILENAME_PART_COLORS[part]}`
                          : "border-border text-muted-foreground hover:border-primary/50 hover:text-foreground"
                      }`}
                    >
                      {enabled && orderIndex >= 0 && (
                        <span className="inline-flex h-4 w-4 items-center justify-center rounded-full bg-foreground/10 text-[10px] font-semibold">
                          {orderIndex + 1}
                        </span>
                      )}
                      {FILENAME_PART_LABELS[part]}
                    </button>
                  );
                })}
              </div>
              <p className="text-xs text-muted-foreground">
                선택한 순서대로 파일명이 조합됩니다.
              </p>
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

      {/* Preview - 자동 생성 모드에서는 구성 요소를 칩과 같은 색으로 표시 */}
      <div className="p-2 bg-muted rounded-md border border-border">
        <p className="text-xs text-muted-foreground">
          미리보기:{" "}
          <span className="font-mono text-foreground">
            {settings.mode === "custom" ? (
              `${settings.customName.trim() || "파일명을 입력하세요"}.pptx`
            ) : previewSegments.length === 0 ? (
              "translated.pptx"
            ) : (
              <>
                {previewSegments.map((segment, index) => (
                  <span key={segment.part}>
                    {index > 0 && "_"}
                    <span className={`rounded px-0.5 ${FILENAME_PART_COLORS[segment.part]}`}>
                      {segment.text}
                    </span>
                  </span>
                ))}
                .pptx
              </>
            )}
          </span>
        </p>
      </div>
    </div>
  );
}
