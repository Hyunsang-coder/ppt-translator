"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import {
  ArrowDown,
  ArrowRight,
  ArrowUp,
  BookOpen,
  Check,
  Download,
  FileSpreadsheet,
  FileText,
  Pencil,
  Plus,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { apiClient } from "@/lib/api-client";
import {
  glossaryToCsv,
  parseGlossaryCsv,
  safeGlossaryFilename,
  type GlossaryEntry,
} from "@/lib/glossary";
import { saveBlob } from "@/lib/save-file";
import { useGlossaryStore } from "@/stores/glossary-store";

interface GlossaryManagerModalProps {
  open: boolean;
  onClose: () => void;
  disabled?: boolean;
}

const INITIAL_VISIBLE_ENTRIES = 200;

export function GlossaryManagerModal({
  open,
  onClose,
  disabled = false,
}: GlossaryManagerModalProps) {
  const glossaries = useGlossaryStore((state) => state.glossaries);
  const activeGlossaryIds = useGlossaryStore((state) => state.activeGlossaryIds);
  const hydrated = useGlossaryStore((state) => state.hydrated);
  const hydrate = useGlossaryStore((state) => state.hydrate);
  const createGlossary = useGlossaryStore((state) => state.createGlossary);
  const renameGlossary = useGlossaryStore((state) => state.renameGlossary);
  const deleteGlossary = useGlossaryStore((state) => state.deleteGlossary);
  const toggleGlossaryActive = useGlossaryStore((state) => state.toggleGlossaryActive);
  const moveActiveGlossary = useGlossaryStore((state) => state.moveActiveGlossary);
  const addEntry = useGlossaryStore((state) => state.addEntry);
  const updateEntry = useGlossaryStore((state) => state.updateEntry);
  const deleteEntry = useGlossaryStore((state) => state.deleteEntry);
  const importEntries = useGlossaryStore((state) => state.importEntries);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showNew, setShowNew] = useState(false);
  const [newName, setNewName] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [nameDraft, setNameDraft] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [notes, setNotes] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [visibleLimit, setVisibleLimit] = useState(INITIAL_VISIBLE_ENTRIES);

  const csvInputRef = useRef<HTMLInputElement>(null);
  const excelInputRef = useRef<HTMLInputElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    hydrate();
    const state = useGlossaryStore.getState();
    setSelectedId((current) => (
      current && state.glossaries.some((glossary) => glossary.id === current)
        ? current
        : state.activeGlossaryIds[0] ?? state.glossaries[0]?.id ?? null
    ));
    setShowNew(false);
    setRenaming(false);
    setSearchQuery("");
    setVisibleLimit(INITIAL_VISIBLE_ENTRIES);
    resetEntryForm();
  }, [hydrate, open]);

  useEffect(() => {
    if (!open) return;
    const previouslyFocused = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    modalRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
        return;
      }
      if (event.key !== "Tab" || !modalRef.current) return;
      const focusable = Array.from(modalRef.current.querySelectorAll<HTMLElement>(
        'button:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
      ));
      if (focusable.length === 0) {
        event.preventDefault();
        modalRef.current.focus();
        return;
      }
      const first = focusable[0]!;
      const last = focusable[focusable.length - 1]!;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      previouslyFocused?.focus();
    };
  }, [onClose, open]);

  const selected = glossaries.find((glossary) => glossary.id === selectedId) ?? null;
  const entries = selected?.entries ?? [];
  const activeIndex = selectedId ? activeGlossaryIds.indexOf(selectedId) : -1;
  const isActive = activeIndex >= 0;

  const filteredEntries = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    if (!query) return entries;
    return entries.filter((entry) => (
      entry.source.toLowerCase().includes(query)
      || entry.target.toLowerCase().includes(query)
      || entry.notes?.toLowerCase().includes(query)
    ));
  }, [entries, searchQuery]);
  const visibleEntries = filteredEntries.slice(0, visibleLimit);

  if (!open) return null;

  function resetEntryForm() {
    setSource("");
    setTarget("");
    setNotes("");
    setEditingId(null);
  }

  function selectGlossary(id: string) {
    setSelectedId(id);
    setRenaming(false);
    setSearchQuery("");
    setVisibleLimit(INITIAL_VISIBLE_ENTRIES);
    resetEntryForm();
  }

  function notifyError(error: unknown, fallback: string) {
    toast.error(error instanceof Error ? error.message : fallback);
  }

  function handleCreateGlossary() {
    if (disabled || !newName.trim()) return;
    try {
      const id = createGlossary(newName);
      setNewName("");
      setShowNew(false);
      selectGlossary(id);
      toast.success("용어집을 만들었습니다. 사용할 용어집은 별도로 선택할 수 있습니다.");
    } catch (error) {
      notifyError(error, "용어집 생성에 실패했습니다.");
    }
  }

  function handleRename() {
    if (disabled || !selected || !nameDraft.trim()) return;
    try {
      renameGlossary(selected.id, nameDraft);
      setRenaming(false);
      toast.success("이름을 변경했습니다.");
    } catch (error) {
      notifyError(error, "이름 변경에 실패했습니다.");
    }
  }

  function handleDeleteGlossary() {
    if (disabled || !selected) return;
    if (!window.confirm(`「${selected.name}」용어집과 ${selected.entries.length}개 용어를 삭제할까요?`)) {
      return;
    }
    const selectedIndex = glossaries.findIndex((glossary) => glossary.id === selected.id);
    const nextSelected = glossaries[selectedIndex + 1] ?? glossaries[selectedIndex - 1] ?? null;
    try {
      deleteGlossary(selected.id);
      setSelectedId(nextSelected?.id ?? null);
      resetEntryForm();
      toast.success("용어집을 삭제했습니다.");
    } catch (error) {
      notifyError(error, "용어집 삭제에 실패했습니다.");
    }
  }

  function handleToggleActive() {
    if (disabled || !selected) return;
    try {
      toggleGlossaryActive(selected.id);
      toast.success(isActive ? "이번 번역에서 제외했습니다." : "이번 번역에 사용할 용어집으로 추가했습니다.");
    } catch (error) {
      notifyError(error, "용어집 사용 설정을 변경하지 못했습니다.");
    }
  }

  function handleMove(direction: -1 | 1) {
    if (disabled || !selected || !isActive) return;
    try {
      moveActiveGlossary(selected.id, direction);
    } catch (error) {
      notifyError(error, "용어집 우선순위를 변경하지 못했습니다.");
    }
  }

  function handleSaveEntry() {
    if (disabled || !selected) return;
    try {
      if (editingId) {
        updateEntry(selected.id, editingId, { source, target, notes });
        toast.success("용어를 수정했습니다.");
      } else {
        const result = addEntry(selected.id, source, target, notes);
        toast.success(
          result.activated
            ? "용어를 추가하고 이번 번역에 사용할 용어집으로 연결했습니다."
            : "용어를 추가했습니다."
        );
      }
      resetEntryForm();
    } catch (error) {
      notifyError(error, "용어 저장에 실패했습니다.");
    }
  }

  function startEdit(entry: GlossaryEntry) {
    setEditingId(entry.id);
    setSource(entry.source);
    setTarget(entry.target);
    setNotes(entry.notes ?? "");
  }

  async function handleCsvFile(file: File | null) {
    if (!file || !selected || disabled) return;
    if (file.size > 10 * 1024 * 1024) {
      toast.error("파일이 10MB를 초과합니다.");
      return;
    }
    setImporting(true);
    try {
      const parsed = parseGlossaryCsv(await file.text());
      if (parsed.length === 0) throw new Error("유효한 용어를 찾지 못했습니다.");
      const result = importEntries(selected.id, parsed);
      toast.success(`가져오기 완료: ${result.inserted}개 추가, ${result.updated}개 갱신`);
    } catch (error) {
      notifyError(error, "CSV 가져오기에 실패했습니다.");
    } finally {
      setImporting(false);
      if (csvInputRef.current) csvInputRef.current.value = "";
    }
  }

  async function handleExcelFile(file: File | null) {
    if (!file || !selected || disabled) return;
    if (file.size > 10 * 1024 * 1024) {
      toast.error("파일이 10MB를 초과합니다.");
      return;
    }
    setImporting(true);
    try {
      const { entries: parsed } = await apiClient.parseGlossaryFile(file);
      if (parsed.length === 0) throw new Error("유효한 용어를 찾지 못했습니다.");
      const result = importEntries(selected.id, parsed);
      toast.success(`가져오기 완료: ${result.inserted}개 추가, ${result.updated}개 갱신`);
    } catch (error) {
      notifyError(error, "Excel 가져오기에 실패했습니다.");
    } finally {
      setImporting(false);
      if (excelInputRef.current) excelInputRef.current.value = "";
    }
  }

  async function handleExport(format: "csv" | "excel") {
    if (!selected || disabled || exporting) return;
    setExporting(true);
    try {
      const basename = safeGlossaryFilename(selected.name);
      if (format === "csv") {
        const blob = new Blob([glossaryToCsv(selected.entries)], { type: "text/csv;charset=utf-8" });
        await saveBlob(blob, `${basename}.csv`);
      } else {
        const blob = await apiClient.exportGlossaryFile(selected.name, selected.entries, "excel");
        await saveBlob(blob, `${basename}.xlsx`);
      }
      toast.success("용어집을 내보냈습니다.");
    } catch (error) {
      notifyError(error, "용어집 내보내기에 실패했습니다.");
    } finally {
      setExporting(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-2 sm:p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="glossary-manager-title"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={modalRef}
        tabIndex={-1}
        className="flex h-[min(760px,94vh)] w-full max-w-5xl overflow-hidden rounded-xl border border-border bg-background shadow-xl outline-none max-sm:flex-col"
      >
        <aside className="flex w-60 shrink-0 flex-col border-r border-border bg-muted/30 max-sm:max-h-44 max-sm:w-full max-sm:border-b max-sm:border-r-0">
          <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-3">
            <div>
              <h2 id="glossary-manager-title" className="text-sm font-semibold">용어집</h2>
              <p className="text-[10px] text-muted-foreground">내 용어집 라이브러리</p>
            </div>
            <Button
              type="button"
              variant="ghost"
              size="icon-xs"
              disabled={disabled}
              onClick={() => setShowNew(true)}
              aria-label="새 용어집"
            >
              <Plus />
            </Button>
          </div>

          {showNew && (
            <div className="flex gap-1 border-b border-border p-2">
              <Input
                value={newName}
                onChange={(event) => setNewName(event.target.value)}
                placeholder="새 용어집 이름"
                className="h-8 text-sm"
                disabled={disabled}
                onKeyDown={(event) => {
                  if (event.key === "Enter") handleCreateGlossary();
                  if (event.key === "Escape") setShowNew(false);
                }}
                autoFocus
              />
              <Button
                type="button"
                size="icon-sm"
                disabled={disabled || !newName.trim()}
                onClick={handleCreateGlossary}
                aria-label="용어집 만들기"
              >
                <Check />
              </Button>
            </div>
          )}

          <ul className="flex-1 space-y-1 overflow-y-auto p-2">
            {!hydrated ? (
              <li className="p-4 text-center text-xs text-muted-foreground">불러오는 중...</li>
            ) : glossaries.length === 0 ? (
              <li>
                <button
                  type="button"
                  onClick={() => setShowNew(true)}
                  className="w-full rounded-md border border-dashed border-border p-4 text-left text-xs text-muted-foreground hover:border-primary/50 hover:text-foreground"
                >
                  저장된 용어집이 없습니다. 새 용어집을 만들어주세요.
                </button>
              </li>
            ) : glossaries.map((glossary) => {
              const selectedItem = glossary.id === selectedId;
              const priority = activeGlossaryIds.indexOf(glossary.id);
              return (
                <li key={glossary.id}>
                  <button
                    type="button"
                    onClick={() => selectGlossary(glossary.id)}
                    aria-pressed={selectedItem}
                    className={`flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-sm transition-colors ${
                      selectedItem
                        ? "bg-primary/10 font-medium text-foreground"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground"
                    }`}
                  >
                    <span className={`h-2 w-2 shrink-0 rounded-full ${
                      priority >= 0 ? "bg-primary" : "border border-muted-foreground"
                    }`} />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate">{glossary.name}</span>
                      <span className="block text-[11px] opacity-70">{glossary.entries.length}개 용어</span>
                    </span>
                    {priority >= 0 && (
                      <span className="text-[10px] font-semibold text-primary">{priority + 1}</span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
        </aside>

        <main className="flex min-h-0 min-w-0 flex-1 flex-col">
          {!selected ? (
            <div className="relative flex flex-1 items-center justify-center p-8 text-center text-sm text-muted-foreground">
              용어집을 선택하거나 새로 만들어주세요.
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                className="absolute right-3 top-3"
                onClick={onClose}
                aria-label="닫기"
              >
                <X />
              </Button>
            </div>
          ) : (
            <>
              <header className="shrink-0 border-b border-border px-4 py-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    {renaming ? (
                      <div className="flex max-w-sm gap-2">
                        <Input
                          value={nameDraft}
                          onChange={(event) => setNameDraft(event.target.value)}
                          className="h-8 text-sm"
                          onKeyDown={(event) => {
                            if (event.key === "Enter") handleRename();
                            if (event.key === "Escape") setRenaming(false);
                          }}
                          autoFocus
                        />
                        <Button type="button" size="sm" className="h-8" onClick={handleRename}>저장</Button>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2">
                        <BookOpen className="h-4 w-4 text-muted-foreground" />
                        <h3 className="truncate text-sm font-semibold">{selected.name}</h3>
                        <button
                          type="button"
                          className="text-muted-foreground hover:text-foreground"
                          disabled={disabled}
                          onClick={() => {
                            setNameDraft(selected.name);
                            setRenaming(true);
                          }}
                          aria-label="이름 변경"
                        >
                          <Pencil className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    )}
                    <p className="mt-1 text-xs text-muted-foreground">
                      용어집을 편집하는 것과 이번 번역에 사용하는 것은 별도로 관리됩니다.
                    </p>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon-sm"
                      className="text-destructive hover:text-destructive"
                      disabled={disabled}
                      onClick={handleDeleteGlossary}
                      aria-label="용어집 삭제"
                    >
                      <Trash2 />
                    </Button>
                    <Button type="button" variant="ghost" size="icon-sm" onClick={onClose} aria-label="닫기">
                      <X />
                    </Button>
                  </div>
                </div>

                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <Button
                    type="button"
                    variant={isActive ? "secondary" : "outline"}
                    size="sm"
                    className="h-8"
                    disabled={disabled}
                    onClick={handleToggleActive}
                  >
                    {isActive ? `사용 중 · 우선순위 ${activeIndex + 1}` : "이번 번역에 사용"}
                  </Button>
                  {isActive && activeGlossaryIds.length > 1 && (
                    <>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-sm"
                        disabled={disabled || activeIndex === 0}
                        onClick={() => handleMove(-1)}
                        aria-label="우선순위 올리기"
                      >
                        <ArrowUp />
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon-sm"
                        disabled={disabled || activeIndex === activeGlossaryIds.length - 1}
                        onClick={() => handleMove(1)}
                        aria-label="우선순위 내리기"
                      >
                        <ArrowDown />
                      </Button>
                    </>
                  )}
                </div>
              </header>

              <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border px-4 py-2">
                <input
                  ref={csvInputRef}
                  type="file"
                  accept=".csv,.tsv,text/csv,text/tab-separated-values"
                  className="hidden"
                  disabled={disabled || importing}
                  onChange={(event) => void handleCsvFile(event.target.files?.[0] ?? null)}
                />
                <input
                  ref={excelInputRef}
                  type="file"
                  accept=".xlsx,.xls,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel"
                  className="hidden"
                  disabled={disabled || importing}
                  onChange={(event) => void handleExcelFile(event.target.files?.[0] ?? null)}
                />
                <Button type="button" variant="outline" size="sm" className="h-8" disabled={disabled || importing} onClick={() => csvInputRef.current?.click()}>
                  <FileText /> CSV 가져오기
                </Button>
                <Button type="button" variant="outline" size="sm" className="h-8" disabled={disabled || importing} onClick={() => excelInputRef.current?.click()}>
                  <FileSpreadsheet /> Excel 가져오기
                </Button>
                <Button type="button" variant="ghost" size="sm" className="h-8" disabled={disabled || exporting} onClick={() => void handleExport("csv")}>
                  <Download /> CSV
                </Button>
                <Button type="button" variant="ghost" size="sm" className="h-8" disabled={disabled || exporting} onClick={() => void handleExport("excel")}>
                  <Download /> Excel
                </Button>
                <div className="relative ml-auto min-w-[150px] max-w-[220px] flex-1">
                  <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={searchQuery}
                    onChange={(event) => {
                      setSearchQuery(event.target.value);
                      setVisibleLimit(INITIAL_VISIBLE_ENTRIES);
                    }}
                    placeholder="용어 검색"
                    className="h-8 pl-7 text-sm"
                  />
                </div>
              </div>

              <div className="shrink-0 space-y-2 border-b border-border px-4 py-3">
                <div className="flex flex-wrap items-end gap-2">
                  <div className="min-w-[120px] flex-1 space-y-1">
                    <Label className="text-xs text-muted-foreground">원문</Label>
                    <Input value={source} onChange={(event) => setSource(event.target.value)} disabled={disabled} className="h-8 text-sm" placeholder="Source" />
                  </div>
                  <ArrowRight className="mb-2 h-4 w-4 text-muted-foreground" />
                  <div className="min-w-[120px] flex-1 space-y-1">
                    <Label className="text-xs text-muted-foreground">번역</Label>
                    <Input
                      value={target}
                      onChange={(event) => setTarget(event.target.value)}
                      disabled={disabled}
                      className="h-8 text-sm"
                      placeholder="Target"
                      onKeyDown={(event) => {
                        if (event.key === "Enter") handleSaveEntry();
                      }}
                    />
                  </div>
                  <div className="min-w-[100px] flex-1 space-y-1">
                    <Label className="text-xs text-muted-foreground">메모 (선택)</Label>
                    <Input value={notes} onChange={(event) => setNotes(event.target.value)} disabled={disabled} className="h-8 text-sm" placeholder="Notes" />
                  </div>
                  <Button type="button" size="sm" className="h-8" disabled={disabled || !source.trim() || !target.trim()} onClick={handleSaveEntry}>
                    {editingId ? "수정" : "추가"}
                  </Button>
                  {editingId && (
                    <Button type="button" variant="ghost" size="sm" className="h-8" onClick={resetEntryForm}>취소</Button>
                  )}
                </div>
              </div>

              <div className="min-h-0 flex-1 overflow-y-auto p-2">
                {visibleEntries.length === 0 ? (
                  <p className="px-2 py-8 text-center text-sm text-muted-foreground">
                    {entries.length === 0 ? "용어가 없습니다. 위 입력란이나 파일 가져오기로 추가하세요." : "검색 결과가 없습니다."}
                  </p>
                ) : (
                  <ul>
                    {visibleEntries.map((entry) => (
                      <li key={entry.id} className="group flex items-center gap-2 rounded-md px-2 py-1.5 hover:bg-muted/60">
                        <span className="min-w-0 flex-1 truncate text-sm">
                          <span className="font-medium">{entry.source}</span>
                          <span className="mx-1.5 text-muted-foreground">→</span>
                          <span>{entry.target}</span>
                          {entry.notes && <span className="ml-2 text-xs text-muted-foreground">({entry.notes})</span>}
                        </span>
                        <button type="button" className="text-muted-foreground opacity-0 hover:text-foreground group-hover:opacity-100 focus-visible:opacity-100" disabled={disabled} onClick={() => startEdit(entry)} aria-label={`${entry.source} 수정`}>
                          <Pencil className="h-3.5 w-3.5" />
                        </button>
                        <button
                          type="button"
                          className="text-muted-foreground opacity-0 hover:text-destructive group-hover:opacity-100 focus-visible:opacity-100"
                          disabled={disabled}
                          onClick={() => {
                            try {
                              deleteEntry(selected.id, entry.id);
                              if (editingId === entry.id) resetEntryForm();
                            } catch (error) {
                              notifyError(error, "용어 삭제에 실패했습니다.");
                            }
                          }}
                          aria-label={`${entry.source} 삭제`}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
                {visibleEntries.length < filteredEntries.length && (
                  <div className="flex justify-center py-3">
                    <Button type="button" variant="ghost" size="sm" onClick={() => setVisibleLimit((limit) => limit + INITIAL_VISIBLE_ENTRIES)}>
                      더 보기 ({filteredEntries.length - visibleEntries.length}개 남음)
                    </Button>
                  </div>
                )}
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
