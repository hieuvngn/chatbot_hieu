import { AlertTriangle, Info, Puzzle } from "lucide-react";
import CitationCard from "./CitationCard";
import Markdown from "./Markdown";
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

  return (
    <div className="space-y-2">
      {fallback && (
        <p className="flex items-start gap-1.5 text-xs text-blue-700 dark:text-blue-300">
          <Info size={13} className="mt-0.5 shrink-0" />
          Không tìm thấy tài liệu liên quan trong kho — trả lời theo hiểu biết chung (không có trích dẫn).
        </p>
      )}
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start">
        <div className="min-w-0 flex-1 text-sm leading-relaxed text-zinc-800 dark:text-zinc-100">
          <Markdown text={turn.text} />
          {turn.skills_applied.length > 0 && (
            <p className="mt-2 flex flex-wrap items-center gap-1 text-[11px] text-zinc-400">
              <Puzzle size={11} />
              {turn.skills_applied.join(", ")}
            </p>
          )}
        </div>
        {turn.citations.length > 0 && (
          <div className="flex w-full shrink-0 flex-row gap-2 overflow-x-auto lg:w-56 lg:flex-col lg:overflow-visible">
            {turn.citations.map((citation) => (
              <div key={citation.marker} className="w-48 shrink-0 lg:w-full">
                <CitationCard citation={citation} />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
