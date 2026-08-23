import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUp } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import ChatMessage from "@/components/ChatMessage";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useWebToggle } from "@/lib/web-toggle-context";

const SUGGESTIONS = [
  "Giải thích bảng băm là gì?",
  "Nên học môn nào trước khi học Trí tuệ nhân tạo?",
  "So sánh danh sách liên kết và mảng.",
];

function TypingIndicator() {
  return (
    <div className="flex gap-1 py-2">
      {[0, 150, 300].map((delay) => (
        <span
          key={delay}
          className="h-2 w-2 animate-bounce rounded-full bg-zinc-400"
          style={{ animationDelay: `${delay}ms` }}
        />
      ))}
    </div>
  );
}

export default function ChatPage() {
  const { conversationId } = useParams();
  const queryClient = useQueryClient();
  const { useWeb } = useWebToggle();
  const [input, setInput] = useState("");
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [sendError, setSendError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const { data: turns = [], isLoading } = useQuery({
    queryKey: ["messages", conversationId],
    queryFn: () => api.messages(conversationId!),
    enabled: conversationId !== undefined,
  });

  const sendMutation = useMutation({
    mutationFn: (message: string) => api.chat(conversationId!, message, useWeb),
    onMutate: (message) => {
      setPendingQuestion(message);
      setSendError(null);
    },
    onSuccess: () => {
      setPendingQuestion(null);
      setInput("");
      if (textareaRef.current !== null) textareaRef.current.style.height = "auto";
      void queryClient.invalidateQueries({ queryKey: ["messages", conversationId] });
      void queryClient.invalidateQueries({ queryKey: ["conversations"] });
    },
    onError: (error) => {
      setPendingQuestion(null);
      setSendError(error instanceof Error ? error.message : "Lỗi không xác định.");
    },
  });

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns.length, pendingQuestion]);

  function submit(text: string): void {
    const trimmed = text.trim();
    if (trimmed.length === 0 || sendMutation.isPending) return;
    sendMutation.mutate(trimmed);
  }

  function resize(event: React.ChangeEvent<HTMLTextAreaElement>): void {
    setInput(event.target.value);
    event.target.style.height = "auto";
    event.target.style.height = `${Math.min(event.target.scrollHeight, 160)}px`;
  }

  const empty = !isLoading && turns.length === 0 && pendingQuestion === null;

  return (
    <div className="mx-auto flex h-full w-full max-w-[720px] flex-col px-4">
      <div className="min-h-0 flex-1 space-y-6 py-6">
        {empty && (
          <div className="flex h-full flex-col items-center justify-center gap-6 text-center">
            <h1 className="text-2xl font-semibold tracking-tight">CourseMate</h1>
            <div className="grid w-full max-w-md gap-2">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  onClick={() => submit(suggestion)}
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

        {pendingQuestion !== null && (
          <>
            <div className="flex justify-end">
              <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl bg-indigo-100 px-4 py-2.5 text-sm dark:bg-indigo-900/60">
                {pendingQuestion}
              </div>
            </div>
            <TypingIndicator />
          </>
        )}

        {sendError !== null && (
          <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950">
            Không gửi được câu trả lời: {sendError}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="sticky bottom-0 bg-gradient-to-t from-white via-white pb-4 pt-2 dark:from-zinc-950 dark:via-zinc-950">
        <div className="relative">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={resize}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit(input);
              }
            }}
            rows={1}
            placeholder="Hỏi về môn học hoặc tài liệu…"
            className="w-full resize-none rounded-2xl border py-3 pl-4 pr-12 text-sm shadow-sm outline-none focus:ring-2 focus:ring-indigo-500/40 dark:bg-zinc-900"
          />
          <Button
            size="icon"
            disabled={input.trim().length === 0 || sendMutation.isPending}
            onClick={() => submit(input)}
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
