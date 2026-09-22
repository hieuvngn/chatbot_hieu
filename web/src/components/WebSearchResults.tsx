import { ExternalLink, Globe } from "lucide-react";
import type { Citation } from "@/lib/types";

/**
 * "Search results" panel — surfaces the actual web URLs the pipeline
 * retrieved (via Firecrawl) and the LLM may or may not have cited inline.
 * Renders only citations whose source.kind === "web".
 */
export default function WebSearchResults({ citations }: { citations: Citation[] }) {
  const web = citations.filter((c) => c.source.kind === "web");
  if (web.length === 0) return null;
  return (
    <section className="space-y-1.5">
      <header className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
        <Globe size={11} />
        <span>Web search results ({web.length})</span>
      </header>
      <ul className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
        {web.map((citation, idx) => {
          const url = citation.source.chapter;
          const title = citation.source.document_title || url;
          const host = safeHost(url);
          return (
            <li key={`${url}-${idx}`}>
              <a
                href={url}
                target="_blank"
                rel="noreferrer"
                className="flex items-start gap-2 rounded-lg border border-indigo-100 bg-indigo-50/60 px-2.5 py-1.5 text-xs transition hover:border-indigo-300 hover:bg-indigo-50 dark:border-indigo-900 dark:bg-indigo-950/40 dark:hover:bg-indigo-950"
              >
                <ExternalLink size={12} className="mt-0.5 shrink-0 text-indigo-600 dark:text-indigo-400" />
                <div className="min-w-0">
                  <div className="line-clamp-1 font-medium text-zinc-800 dark:text-zinc-100">{title}</div>
                  {host !== "" && (
                    <div className="line-clamp-1 text-[10px] text-zinc-500">{host}</div>
                  )}
                </div>
              </a>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function safeHost(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return "";
  }
}