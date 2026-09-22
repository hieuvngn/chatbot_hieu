import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";

const HEADING_RE = /^(#{1,4})\s+(.+?)\s*$/;

function renderMarkdown(text: string): React.ReactNode[] {
  const lines = text.split("\n");
  const out: React.ReactNode[] = [];
  let buffer: string[] = [];
  const flush = (key: number) => {
    if (buffer.length === 0) return;
    out.push(
      <p key={`p-${key}`} className="text-sm leading-relaxed text-zinc-700 dark:text-zinc-200">
        {buffer.join(" ")}
      </p>,
    );
    buffer = [];
  };
  lines.forEach((line, i) => {
    const m = line.match(HEADING_RE);
    if (m !== null) {
      flush(i);
      const level = m[1].length;
      const cls =
        level === 1
          ? "mt-4 text-base font-semibold first:mt-0"
          : level === 2
            ? "mt-3 text-sm font-semibold first:mt-0"
            : "mt-2 text-sm font-medium first:mt-0";
      out.push(
        <h3 key={`h-${i}`} className={cls}>
          {m[2]}
        </h3>,
      );
    } else if (line.trim() === "") {
      flush(i);
    } else {
      buffer.push(line);
    }
  });
  flush(lines.length);
  return out;
}

export default function AttachmentViewer({
  attachment,
  focusChapter,
  onClose,
}: {
  attachment: { id: string; filename: string; file_kind: string };
  /** Chapter to scroll to and highlight once content has loaded. */
  focusChapter?: string | null;
  onClose: () => void;
}) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["attachment-content", attachment.id],
    queryFn: () => api.attachmentContent(attachment.id),
  });
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (data === undefined || focusChapter == null) return;
    const container = scrollRef.current;
    if (container === null) return;
    const match = Array.from(
      container.querySelectorAll<HTMLElement>("[data-chapter]"),
    ).find((s) => s.dataset.chapter === focusChapter);
    if (match === undefined) return;
    match.scrollIntoView({ behavior: "smooth", block: "start" });
    const HIGHLIGHT = [
      "bg-indigo-50",
      "dark:bg-indigo-950/60",
      "ring-1",
      "ring-indigo-200",
      "dark:ring-indigo-800",
    ];
    match.classList.add(...HIGHLIGHT);
    return () => match.classList.remove(...HIGHLIGHT);
  }, [data, focusChapter]);

  return (
    <aside className="flex h-full w-full flex-col border-l bg-white dark:bg-zinc-950">
      <header className="flex items-center gap-2 border-b px-4 py-3">
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          aria-label="Đóng viewer"
          className="h-8 w-8"
        >
          <ChevronLeft size={16} />
        </Button>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold">{attachment.filename}</p>
          <p className="text-[11px] uppercase tracking-wide text-zinc-400">
            {attachment.file_kind}
          </p>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          aria-label="Đóng viewer"
          className="h-8 w-8"
        >
          <X size={16} />
        </Button>
      </header>
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
        {isLoading && <p className="text-sm text-zinc-400">Đang tải nội dung…</p>}
        {error !== null && error !== undefined && (
          <p className="text-sm text-red-600">
            Không tải được nội dung: {String(error)}
          </p>
        )}
        {data !== undefined && data.sections.length === 0 && (
          <p className="text-sm text-zinc-400">Tài liệu trống.</p>
        )}
        {data !== undefined && data.sections.map((section, i) => (
          <section key={i} data-chapter={section.chapter} className="mb-4 scroll-mt-4 rounded-xl p-1 last:mb-0">
            {section.chapter !== "" && (
              <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-zinc-500">
                {section.chapter}
              </h4>
            )}
            {attachment.file_kind === "md" ? (
              <div>{renderMarkdown(section.text)}</div>
            ) : (
              <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-zinc-700 dark:text-zinc-200">
                {section.text}
              </pre>
            )}
          </section>
        ))}
      </div>
    </aside>
  );
}
