import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Paperclip, Trash2, Upload } from "lucide-react";
import { useRef } from "react";
import { useParams } from "react-router-dom";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useAttachmentViewer } from "@/lib/attachment-viewer-context";

const KIND_ICONS: Record<string, string> = { pdf: "📄", md: "📝", txt: "🗒️" };
const MAX_FILES = 3; // MAX_FILES_PER_CONVERSATION in rag_core.attachments

export default function AttachmentsSection() {
  const { conversationId } = useParams();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { openAttachment } = useAttachmentViewer();

  const { data: attachments = [] } = useQuery({
    queryKey: ["attachments", conversationId],
    queryFn: () => api.attachments(conversationId!),
    enabled: conversationId !== undefined,
  });

  const invalidate = () =>
    queryClient.invalidateQueries({ queryKey: ["attachments", conversationId] });

  const uploadMutation = useMutation({
    mutationFn: (file: File) => api.uploadAttachment(conversationId!, file),
    onSuccess: (meta) => {
      toast.success(`Đã xử lý ${meta.filename}: ${meta.chunk_count} đoạn.`);
      void invalidate();
    },
    onError: (error) => toast.error(error instanceof Error ? error.message : String(error)),
  });

  const deleteMutation = useMutation({
    mutationFn: api.deleteAttachment,
    onSuccess: invalidate,
    onError: (error) => toast.error(String(error)),
  });

  const atLimit = attachments.length >= MAX_FILES;

  return (
    <section className="border-t pt-3">
      <p className="flex items-center gap-1.5 px-2 pb-1 text-xs font-medium uppercase tracking-wide text-zinc-400">
        <Paperclip size={12} /> Tài liệu đính kèm ({attachments.length}/{MAX_FILES})
      </p>
      <ul className="space-y-1 px-2">
        {attachments.map((attachment) => (
          <li
            key={attachment.id}
            className="group flex items-center gap-1.5 rounded-lg px-1 py-1 hover:bg-zinc-50 dark:hover:bg-zinc-900"
          >
            <span className="text-sm">{KIND_ICONS[attachment.file_kind] ?? "📄"}</span>
            <button
              type="button"
              onClick={() => openAttachment(attachment.id)}
              className="min-w-0 flex-1 text-left"
              aria-label={`Xem ${attachment.filename}`}
            >
              <span className="block truncate text-xs">{attachment.filename}</span>
              <span className="block text-[10px] text-zinc-400">
                {attachment.chunk_count} đoạn ·{" "}
                {Math.max(1, Math.floor(attachment.size_bytes / 1024))} KB
              </span>
            </button>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 opacity-0 transition group-hover:opacity-100"
              aria-label="Xóa tài liệu"
              onClick={() => deleteMutation.mutate(attachment.id)}
            >
              <Trash2 size={12} />
            </Button>
          </li>
        ))}
      </ul>
      {atLimit ? (
        <p className="px-2 pt-1 text-[11px] text-zinc-400">
          Đã đạt giới hạn {MAX_FILES} tài liệu cho chat này.
        </p>
      ) : (
        <>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.md"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file !== undefined) uploadMutation.mutate(file);
              event.target.value = "";
            }}
          />
          <Button
            variant="outline"
            size="sm"
            className="mx-2 mt-1 w-[calc(100%-1rem)] justify-start gap-2 rounded-xl"
            disabled={uploadMutation.isPending}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload size={13} />{" "}
            {uploadMutation.isPending ? "Đang xử lý…" : "Thêm tài liệu"}
          </Button>
        </>
      )}
    </section>
  );
}
