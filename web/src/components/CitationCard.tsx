import { ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import type { Citation } from "@/lib/types";

const KIND_LABEL: Record<string, string> = {
  slides: "slides",
  textbook: "giáo trình",
  upload: "đính kèm",
  web: "web",
};

export default function CitationCard({ citation }: { citation: Citation }) {
  const { source } = citation;
  const isWeb = source.kind === "web";
  return (
    <Dialog>
      {/* Adapted for Base UI: brief had `<DialogTrigger asChild><button ...>...</button></DialogTrigger>`;
          @base-ui/react Trigger has no asChild — uses render prop instead (pattern per Sidebar.tsx/Header.tsx). */}
      <DialogTrigger
        render={
          <button
            type="button"
            className="w-full rounded-xl border bg-white px-3 py-2 text-left text-xs shadow-sm transition hover:border-indigo-300 dark:bg-zinc-900"
          />
        }
      >
        <span className="mr-1 font-semibold text-indigo-600 dark:text-indigo-400">{citation.marker}</span>
        <span className="line-clamp-2">{source.document_title}</span>
        <span className="mt-0.5 flex items-center gap-1 text-zinc-400">
          <span className="rounded bg-zinc-100 px-1 dark:bg-zinc-800">{KIND_LABEL[source.kind] ?? source.kind}</span>
          <span className="truncate">{isWeb ? "liên kết" : source.chapter}</span>
        </span>
      </DialogTrigger>
      <DialogContent className="max-w-md rounded-2xl">
        <DialogHeader>
          <DialogTitle>
            {citation.marker} {source.document_title}
          </DialogTitle>
          <DialogDescription>{source.chapter}</DialogDescription>
        </DialogHeader>
        <dl className="space-y-1.5 text-sm">
          <Row label="Mã tài liệu" value={source.document_id} />
          {!isWeb && <Row label="Khóa học" value={source.course_code} />}
          <Row label="Loại" value={KIND_LABEL[source.kind] ?? source.kind} />
          <Row label="Ngôn ngữ" value={source.language} />
        </dl>
        {/* Adapted for Base UI: brief had `<Button asChild ...><a ...>...</a></Button>`;
            @base-ui/react Button has no asChild — uses render prop instead (pattern per dialog.tsx close button). */}
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
