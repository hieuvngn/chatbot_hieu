import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { RenderResult } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { AttachmentViewerProvider } from "@/lib/attachment-viewer-context";
import type { Citation, Source, Turn } from "@/lib/types";
import { WebToggleProvider } from "@/lib/web-toggle-context";
interface RenderOptions {
  /** Initial history entry, so useParams() resolves inside the component. */
  route?: string;
  /** Route pattern the component is mounted at. */
  path?: string;
  /** Stands in for the layout's viewer state, so clicks are observable. */
  openAttachment?: (id: string, chapter?: string) => void;
}

/** Mount a component inside the providers App.tsx supplies. */
export function renderWithProviders(
  ui: ReactElement,
  options: RenderOptions = {},
): RenderResult {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  const openAttachment = options.openAttachment ?? (() => undefined);
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[options.route ?? "/c/conv-1"]}>
        <WebToggleProvider>
          <AttachmentViewerProvider value={{ openAttachment }}>
            <Routes>
              <Route path={options.path ?? "/c/:conversationId"} element={ui} />
            </Routes>
          </AttachmentViewerProvider>
        </WebToggleProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

export function makeSource(overrides: Partial<Source> = {}): Source {
  return {
    document_id: "DOC-001",
    document_title: "Slides Cấu trúc dữ liệu",
    chapter: "Chương 3: Bảng băm",
    course_code: "CS101",
    kind: "slides",
    language: "vi",
    ...overrides,
  };
}

export function makeCitation(overrides: Partial<Source> = {}): Citation {
  return { marker: "[1]", source: makeSource(overrides) };
}

export function makeTurn(overrides: Partial<Turn> = {}): Turn {
  return {
    role: "assistant",
    text: "",
    citations: [],
    refused: false,
    rephrase_suggestion: "",
    skills_applied: [],
    ...overrides,
  };
}
