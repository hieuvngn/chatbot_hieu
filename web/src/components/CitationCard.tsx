import type { ReactElement } from "react";
import { useQuery } from "@tanstack/react-query";
import { useParams } from "react-router-dom";
import { ExternalLink, GraduationCap, Calendar, Building2, BookOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { api } from "@/lib/api";
import { useAttachmentViewer } from "@/lib/attachment-viewer-context";
import type { Citation } from "@/lib/types";

const KIND_LABEL: Record<string, string> = {
  slides: "slides",
  textbook: "giáo trình",
  upload: "đính kèm",
  web: "web",
  course: "môn học",
};

const ENTITY_LABEL: Record<string, string> = {
  instructor: "Giảng viên",
  program: "Chương trình",
  term: "Học kỳ",
  department: "Khoa",
};

const ENTITY_ICON: Record<string, ReactElement> = {
  instructor: <GraduationCap size={12} />,
  program: <BookOpen size={12} />,
  term: <Calendar size={12} />,
  department: <Building2 size={12} />,
};

const UPLOAD_PREFIX = "upload_";

export default function CitationCard({ citation }: { citation: Citation }) {
  const { source } = citation;
  const isWeb = source.kind === "web";
  const isEntity = !!source.entity_type;
  const isUpload = !isEntity && !isWeb && source.kind === "upload";
  const entityLabel = source.entity_type ? ENTITY_LABEL[source.entity_type] ?? source.entity_type : null;
  const { conversationId } = useParams();
  const { openAttachment } = useAttachmentViewer();
  const { data: attachments = [] } = useQuery({
    queryKey: ["attachments", conversationId],
    queryFn: () => api.attachments(conversationId!),
    enabled: isUpload && conversationId !== undefined,
  });
  // Upload document_ids carry the attachment id prefix ("upload_" + hex);
  // resolve the live attachment so the viewer can show the cited chapter.
  const target = isUpload
    ? attachments.find((a) => a.id.startsWith(source.document_id.slice(UPLOAD_PREFIX.length)))
    : undefined;

  const body = (
    <>
      <span className="mr-1 font-semibold text-indigo-600 dark:text-indigo-400">{citation.marker}</span>
      <span className="line-clamp-2">{source.document_title}</span>
      <span className="mt-0.5 flex items-center gap-1 text-zinc-400">
        <span className="flex items-center gap-1 rounded bg-zinc-100 px-1 dark:bg-zinc-800">
          {isEntity && ENTITY_ICON[source.entity_type ?? ""]}
          {entityLabel ?? KIND_LABEL[source.kind] ?? source.kind}
        </span>
        {!isEntity && <span className="truncate">{isWeb ? "liên kết" : source.chapter}</span>}
      </span>
    </>
  );
  const triggerClass =
    "w-full rounded-xl border bg-white px-3 py-2 text-left text-xs shadow-sm transition hover:border-indigo-300 dark:bg-zinc-900";

  if (target !== undefined) {
    return (
      <button
        type="button"
        onClick={() => openAttachment(target.id, source.chapter)}
        aria-label={`Xem trích dẫn từ ${source.document_title}`}
        className={triggerClass}
      >
        {body}
      </button>
    );
  }

  return (
    <Dialog>
      <DialogTrigger render={<button type="button" className={triggerClass} />}>
        {body}
      </DialogTrigger>
      <DialogContent className="max-w-md rounded-2xl">
        <DialogHeader>
          <DialogTitle>
            {citation.marker} {source.document_title}
          </DialogTitle>
          <DialogDescription>{entityLabel ?? source.chapter}</DialogDescription>
        </DialogHeader>
        <dl className="space-y-1.5 text-sm">
          {isEntity ? (
            <>
              <Row label="Mã" value={source.entity_id ?? source.document_id} />
              <Row label="Loại" value={entityLabel ?? source.kind} />
            </>
          ) : (
            <>
              <Row label="Mã tài liệu" value={source.document_id} />
              {!isWeb && <Row label="Khóa học" value={source.course_code} />}
              <Row label="Loại" value={KIND_LABEL[source.kind] ?? source.kind} />
              <Row label="Ngôn ngữ" value={source.language} />
            </>
          )}
        </dl>
        {isWeb && (
          <Button
            variant="outline"
            className="gap-2 rounded-xl"
            render={<a href={source.chapter} target="_blank" rel="noreferrer" />}
          >
            <ExternalLink size={14} /> Mở nguồn
          </Button>
        )}
      </DialogContent>
    </Dialog>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  if (!value) return null;
  return (
    <div className="flex gap-2">
      <dt className="w-24 shrink-0 text-zinc-400">{label}</dt>
      <dd className="min-w-0 break-words">{value}</dd>
    </div>
  );
}