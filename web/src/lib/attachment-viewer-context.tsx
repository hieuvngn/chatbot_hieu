import { createContext, useContext } from "react";
import type { ReactNode } from "react";

export interface AttachmentViewerValue {
  /** Open the side viewer on an attachment, optionally focusing a chapter. */
  openAttachment: (id: string, chapter?: string) => void;
}

const AttachmentViewerContext = createContext<AttachmentViewerValue>({
  openAttachment: () => undefined,
});

/** The layout owns the viewer state and injects it here. */
export function AttachmentViewerProvider({
  value,
  children,
}: {
  value: AttachmentViewerValue;
  children: ReactNode;
}) {
  return (
    <AttachmentViewerContext.Provider value={value}>
      {children}
    </AttachmentViewerContext.Provider>
  );
}

export function useAttachmentViewer(): AttachmentViewerValue {
  return useContext(AttachmentViewerContext);
}
