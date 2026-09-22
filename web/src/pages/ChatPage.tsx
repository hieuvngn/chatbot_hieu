import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUp, Globe } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import ChatMessage from "@/components/ChatMessage";
import { Button } from "@/components/ui/button";
import { api, streamChat } from "@/lib/api";
import type { Turn } from "@/lib/types";
import { useWebToggle } from "@/lib/web-toggle-context";

const SUGGESTIONS = [
  "Giải thích bảng băm là gì?",
  "Nên học môn nào trước khi học Trí tuệ nhân tạo?",
  "So sánh danh sách liên kết và mảng.",
];

function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-2 py-2 text-xs text-zinc-400">
      <div className="flex gap-1">
        {[0, 150, 300].map((delay) => (
          <span
            key={delay}
            className="h-1.5 w-1.5 animate-bounce rounded-full bg-zinc-400"
            style={{ animationDelay: `${delay}ms` }}
          />
        ))}
      </div>
      <span>Đang suy nghĩ…</span>
    </div>
  );
}

function emptyAssistantTurn(skills: string[]): Turn {
  return {
    role: "assistant",
    text: "",
    citations: [],
    refused: false,
    rephrase_suggestion: "",
    skills_applied: skills,
  };
}

export default function ChatPage() {
  const { conversationId } = useParams();
  const queryClient = useQueryClient();
  const { useWeb, setUseWeb } = useWebToggle();
  const { data: features } = useQuery({ queryKey: ["features"], queryFn: api.features });
  const [input, setInput] = useState("");
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [streamingTurn, setStreamingTurn] = useState<Turn | null>(null);
  const [sendError, setSendError] = useState<string | null>(null);
  const sendingRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const { data: turns = [], isLoading } = useQuery({
    queryKey: ["messages", conversationId],
    queryFn: () => api.messages(conversationId!),
    enabled: conversationId !== undefined,
  });
  const hasLive = pendingQuestion !== null;
  const streamingText = streamingTurn?.text ?? "";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns.length, hasLive, streamingText]);

  useEffect(() => {
    setInput("");
    setPendingQuestion(null);
    setStreamingTurn(null);
    setSendError(null);
    sendingRef.current = false;
    abortRef.current?.abort();
    abortRef.current = null;
    if (textareaRef.current !== null) textareaRef.current.style.height = "auto";
  }, [conversationId]);

  async function submit(text: string): Promise<void> {
    const trimmed = text.trim();
    if (trimmed.length === 0 || sendingRef.current || conversationId === undefined) return;
    sendingRef.current = true;
    setPendingQuestion(trimmed);
    setSendError(null);
    setInput("");
    if (textareaRef.current !== null) textareaRef.current.style.height = "auto";
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamChat(conversationId, trimmed, useWeb, {
        onStart: (meta) => {
          setStreamingTurn(emptyAssistantTurn(meta.skills_applied));
        },
        onDelta: (delta) => {
          setStreamingTurn((prev) =>
            prev === null
              ? emptyAssistantTurn([])
              : { ...prev, text: prev.text + delta },
          );
        },
        onDone: (result) => {
          setStreamingTurn({
            role: "assistant",
            text: result.answer,
            citations: result.citations.map((c) => ({
              marker: c.marker,
              source: c.source,
            })),
            refused: result.refused,
            rephrase_suggestion: result.rephrase_suggestion,
            skills_applied: result.skills_applied,
          });
          setPendingQuestion(null);
          setStreamingTurn(null);
          void queryClient.invalidateQueries({
            queryKey: ["messages", conversationId],
          });
          void queryClient.invalidateQueries({ queryKey: ["conversations"] });
        },
        onRefused: (rephraseSuggestion) => {
          setStreamingTurn({
            role: "assistant",
            text: "",
            citations: [],
            refused: true,
            rephrase_suggestion: rephraseSuggestion,
            skills_applied: [],
          });
          setPendingQuestion(null);
          setStreamingTurn(null);
          void queryClient.invalidateQueries({
            queryKey: ["messages", conversationId],
          });
        },
        onError: (detail) => {
          setPendingQuestion(null);
          setStreamingTurn(null);
          setSendError(detail);
        },
      }, controller.signal);
    } catch (error) {
      // streamChat itself doesn't throw — the callback funnel is the only path.
      // Catch only aborts and unexpected fetch errors.
      if (controller.signal.aborted) return;
      setPendingQuestion(null);
      setStreamingTurn(null);
      setSendError(error instanceof Error ? error.message : "Lỗi không xác định.");
    } finally {
      sendingRef.current = false;
      if (abortRef.current === controller) abortRef.current = null;
    }
  }

  function resize(event: React.ChangeEvent<HTMLTextAreaElement>): void {
    setInput(event.target.value);
    event.target.style.height = "auto";
    event.target.style.height = `${Math.min(event.target.scrollHeight, 160)}px`;
  }

  const empty = !isLoading && turns.length === 0 && !hasLive && streamingTurn === null;

  return (
    <div className="mx-auto flex h-full w-full max-w-[720px] flex-col px-4">
      <div className="min-h-0 flex-1 space-y-6 overflow-y-auto overscroll-contain py-6">
        {empty && (
          <div className="flex h-full flex-col items-center justify-center gap-6 text-center">
            <h1 className="text-2xl font-semibold tracking-tight">CourseMate</h1>
            <div className="grid w-full max-w-md gap-2">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  onClick={() => void submit(suggestion)}
                  className="rounded-xl border bg-white px-4 py-3 text-left text-sm shadow-sm transition hover:border-indigo-300 dark:bg-zinc-900"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((turn, index) => (
          <ChatMessage key={index} turn={turn} />
        ))}

        {hasLive && (
          <div className="flex justify-end">
            <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl bg-indigo-100 px-4 py-2.5 text-sm dark:bg-indigo-900/60">
              {pendingQuestion}
            </div>
          </div>
        )}

        {streamingTurn !== null && (
          <ChatMessage
            turn={{
              ...streamingTurn,
              text: streamingTurn.text + (streamingTurn.text.length === 0 ? "" : ""),
            }}
          />
        )}

        {streamingTurn === null && hasLive && <ThinkingIndicator />}

        {sendError !== null && (
          <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950">
            Không gửi được câu trả lời: {sendError}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="bg-gradient-to-t from-white via-white pb-4 pt-2 dark:from-zinc-950 dark:via-zinc-950">
        <div className="relative">
          {features?.has_web_search === true && (
            <Button
              variant="ghost"
              size="icon"
              type="button"
              onClick={() => setUseWeb(!useWeb)}
              aria-label={useWeb ? "Tắt tìm kiếm web" : "Bật tìm kiếm web"}
              aria-pressed={useWeb}
              title={useWeb ? "Tìm kiếm web: BẬT" : "Tìm kiếm web: tắt"}
              className={`absolute bottom-2.5 left-2.5 h-8 w-8 rounded-full transition ${
                useWeb
                  ? "bg-indigo-100 text-indigo-700 hover:bg-indigo-200 dark:bg-indigo-900/60 dark:text-indigo-300 dark:hover:bg-indigo-900"
                  : "text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-800"
              }`}
            >
              <Globe size={16} />
            </Button>
          )}
          <textarea
            ref={textareaRef}
            value={input}
            onChange={resize}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void submit(input);
              }
            }}
            rows={1}
            placeholder="Hỏi về môn học hoặc tài liệu…"
            className={`w-full resize-none rounded-2xl border py-3 text-sm shadow-sm outline-none focus:ring-2 focus:ring-indigo-500/40 dark:bg-zinc-900 ${features?.has_web_search === true ? "pl-12" : "pl-4"} pr-12`}
          />
          <Button
            size="icon"
            disabled={input.trim().length === 0 || sendingRef.current}
            onClick={() => void submit(input)}
            className="absolute bottom-2.5 right-2.5 h-8 w-8 rounded-full"
            aria-label="Gửi"
          >
            <ArrowUp size={16} />
          </Button>
        </div>
        <p className="mt-1.5 text-center text-[11px] text-zinc-400">
          Enter để gửi · Shift+Enter để xuống dòng
        </p>
      </div>
    </div>
  );
}