import Markdown from "./Markdown";
import type { Turn } from "@/lib/types";

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
  return (
    <div className="text-sm leading-relaxed text-zinc-800 dark:text-zinc-100">
      <Markdown text={turn.text} />
    </div>
  );
}
