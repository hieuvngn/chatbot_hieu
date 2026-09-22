import { AlertTriangle, Info, Puzzle } from "lucide-react";
import CitationCard from "./CitationCard";
import Markdown from "./Markdown";
import WebSearchResults from "./WebSearchResults";
import type { Turn } from "@/lib/types";

function isFallbackNotice(text: string): boolean {
  return text.startsWith("Lưu ý:") || text.startsWith("Note:");
}

export default function ChatMessage({ turn }: { turn: Turn }) {
  if (turn.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl bg-indigo-100 px-4 py-2.5 text-sm text-indigo-950 dark:bg-indigo-900/60 dark:text-indigo-50">
          {turn.text}
        </div>
      </div>
    );
  }

  if (turn.refused) {
    return (
      <div className="rounded-2xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
        <p className="flex items-center gap-2 font-medium">
          <AlertTriangle size={15} />
          Không tìm đủ tài liệu hỗ trợ để trả lời.
        </p>
        {turn.rephrase_suggestion !== "" && (
          <p className="mt-1 text-xs opacity-80">Gợi ý viết lại: {turn.rephrase_suggestion}</p>
        )}
      </div>
    );
  }

  const fallback = isFallbackNotice(turn.text) && turn.citations.length === 0;
  const kbCitations = turn.citations.filter((c) => c.source.kind !== "web");

  return (
    <div className="space-y-3">
      {fallback && (
        <p className="flex items-start gap-1.5 text-xs text-blue-700 dark:text-blue-300">
          <Info size={13} className="mt-0.5 shrink-0" />
          Không tìm thấy tài liệu liên quan trong kho — trả lời theo hiểu biết chung (không có trích dẫn).
        </p>
      )}
      <div className="text-sm leading-relaxed text-zinc-800 dark:text-zinc-100">
        <Markdown text={turn.text} />
        {turn.skills_applied.length > 0 && (
          <p className="mt-2 flex flex-wrap items-center gap-1 text-[11px] text-zinc-400">
            <Puzzle size={11} />
            {turn.skills_applied.join(", ")}
          </p>
        )}
      </div>
      <WebSearchResults citations={turn.citations} />
      {kbCitations.length > 0 && (
        <section className="space-y-1.5">
          <header className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
            <Puzzle size={11} />
            <span>Tài liệu tham khảo ({kbCitations.length})</span>
          </header>
          <ul className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {kbCitations.map((citation) => (
              <li key={citation.marker}>
                <CitationCard citation={citation} />
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}