import { Outlet, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import Header from "./Header";
import Sidebar from "./Sidebar";
import AttachmentViewer from "./AttachmentViewer";
import { useSidebar } from "@/lib/sidebar-context";
import {
  AttachmentViewerProvider,
  type AttachmentViewerValue,
} from "@/lib/attachment-viewer-context";
import { api } from "@/lib/api";

interface ViewerTarget {
  id: string;
  chapter: string | null;
}

export default function AppLayout() {
  const { collapsed } = useSidebar();
  const { conversationId } = useParams();
  const [target, setTarget] = useState<ViewerTarget | null>(null);
  useEffect(() => {
    setTarget(null);
  }, [conversationId]);

  const { data: attachments = [] } = useQuery({
    queryKey: ["attachments", conversationId],
    queryFn: () => api.attachments(conversationId!),
    enabled: conversationId !== undefined && target !== null,
  });
  const selected =
    target === null ? null : attachments.find((a) => a.id === target.id) ?? null;

  const viewerValue: AttachmentViewerValue = {
    openAttachment: (id, chapter) => setTarget({ id, chapter: chapter ?? null }),
  };

  return (
    <AttachmentViewerProvider value={viewerValue}>
      <div className="flex h-screen overflow-hidden">
        <div
          className={`hidden shrink-0 border-r bg-white transition-[width] duration-200 md:block dark:bg-zinc-900 ${
            collapsed ? "w-14" : "w-72"
          }`}
        >
          <Sidebar />
        </div>
        <div className="flex min-w-0 flex-1 flex-col">
          <Header />
          <main className="flex min-h-0 flex-1 overflow-hidden">
            <div className="min-w-0 flex-1 overflow-y-auto">
              <Outlet />
            </div>
            {selected !== null && target !== null && (
              <div className="hidden w-[40%] max-w-[560px] shrink-0 md:block">
                <AttachmentViewer
                  attachment={selected}
                  focusChapter={target.chapter}
                  onClose={() => setTarget(null)}
                />
              </div>
            )}
          </main>
        </div>
      </div>
    </AttachmentViewerProvider>
  );
}
