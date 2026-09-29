import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ChatMessage from "@/components/ChatMessage";
import type { Attachment } from "@/lib/types";
import { makeCitation, makeTurn, renderWithProviders } from "@/test/harness";

const apiMock = vi.hoisted(() => ({
  attachments: vi.fn(async (): Promise<Attachment[]> => []),
  attachmentContent: vi.fn(async () => ({
    sections: [] as { chapter: string; text: string }[],
  })),
}));

vi.mock("@/lib/api", () => ({
  api: apiMock,
  getToken: () => null,
  setToken: () => undefined,
  streamChat: () => undefined,
}));

/**
 * ChatMessage is the join point of the answer UI: it takes a Turn from
 * ChatPage and dispatches to Markdown, WebSearchResults and CitationCard.
 * The branch it picks decides whether citations survive, so each branch is
 * pinned here. Rendering goes through renderWithProviders because
 * CitationCard calls useQuery even when the query is disabled.
 */

beforeEach(() => {
  apiMock.attachments.mockResolvedValue([]);
});

describe("ChatMessage", () => {
  it("renders a user turn as a plain bubble with no citation section", () => {
    renderWithProviders(<ChatMessage turn={makeTurn({ role: "user", text: "bảng băm là gì?" })} />);

    expect(screen.getByText("bảng băm là gì?")).toBeDefined();
    expect(screen.queryByText(/Tài liệu tham khảo/)).toBeNull();
  });

  it("renders a refusal with its rephrase suggestion and no citations", () => {
    renderWithProviders(
      <ChatMessage
        turn={makeTurn({
          refused: true,
          rephrase_suggestion: "Hãy hỏi lại với từ khóa khác.",
        })}
      />,
    );

    expect(screen.getByText(/Không tìm đủ tài liệu/)).toBeDefined();
    expect(screen.getByText(/Hãy hỏi lại với từ khóa khác\./)).toBeDefined();
    expect(screen.queryByText(/Tài liệu tham khảo/)).toBeNull();
  });

  it("omits the suggestion line when a refusal carries none", () => {
    renderWithProviders(<ChatMessage turn={makeTurn({ refused: true, rephrase_suggestion: "" })} />);

    expect(screen.getByText(/Không tìm đủ tài liệu/)).toBeDefined();
    expect(screen.queryByText(/Gợi ý viết lại/)).toBeNull();
  });

  it("counts only non-web citations under Tài liệu tham khảo", () => {
    renderWithProviders(
      <ChatMessage
        turn={makeTurn({
          text: "Bảng băm tra cứu khóa–giá trị [1][2].",
          citations: [
            makeCitation(),
            makeCitation({
              document_id: "https://example.com/btree",
              document_title: "B-tree",
              chapter: "https://example.com/btree",
              course_code: "",
              kind: "web",
            }),
          ],
        })}
      />,
    );

    expect(screen.getByText("Tài liệu tham khảo (1)")).toBeDefined();
    expect(screen.getByText("Web search results (1)")).toBeDefined();
    expect(screen.getByRole("link", { name: /B-tree/ }).getAttribute("href")).toBe(
      "https://example.com/btree",
    );
  });

  it("flags an uncited answer that opens with the fallback disclaimer", () => {
    renderWithProviders(
      <ChatMessage
        turn={makeTurn({
          text: "Lưu ý: Không tìm thấy tài liệu phù hợp.\n\nBảng băm dùng để tra cứu.",
          citations: [],
        })}
      />,
    );

    expect(screen.getByText(/trả lời theo hiểu biết chung/)).toBeDefined();
    expect(screen.queryByText(/Tài liệu tham khảo/)).toBeNull();
  });

  it("does not flag a cited answer that happens to open with Lưu ý:", () => {
    renderWithProviders(
      <ChatMessage
        turn={makeTurn({ text: "Lưu ý: đây là tóm tắt [1].", citations: [makeCitation()] })}
      />,
    );

    expect(screen.queryByText(/trả lời theo hiểu biết chung/)).toBeNull();
    expect(screen.getByText("Tài liệu tham khảo (1)")).toBeDefined();
  });

  it("hides the skill row when the turn reports no skills", () => {
    renderWithProviders(<ChatMessage turn={makeTurn({ text: "x", skills_applied: [] })} />);
    expect(screen.queryByText("eli5")).toBeNull();
  });

  it("joins applied skills into one row", () => {
    renderWithProviders(
      <ChatMessage turn={makeTurn({ text: "x", skills_applied: ["eli5", "exam-prep"] })} />,
    );
    expect(screen.getByText("eli5, exam-prep")).toBeDefined();
  });

  it("opens a citation card with the source details the backend sent", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ChatMessage
        turn={makeTurn({ text: "Có [1]", citations: [makeCitation({ kind: "textbook" })] })}
      />,
    );

    await user.click(screen.getByRole("button", { name: /\[1\]/ }));

    expect(await screen.findByText("DOC-001")).toBeDefined();
    expect(screen.getByText("CS101")).toBeDefined();
    // Once per card badge and once in the dialog's detail table.
    expect(screen.getAllByText("giáo trình")).toHaveLength(2);
  });

  it("resolves an upload citation to the live attachment and opens the viewer", async () => {
    const user = userEvent.setup();
    const opened: [string, string | undefined][] = [];
    apiMock.attachments.mockResolvedValue([
      {
        id: "abc123",
        conversation_id: "conv-1",
        filename: "notes.md",
        file_kind: "md",
        size_bytes: 2048,
        chunk_count: 2,
        created_at: "2026-01-01T00:00:00",
      },
    ]);

    renderWithProviders(
      <ChatMessage
        turn={makeTurn({
          text: "Trích từ tài liệu [1]",
          citations: [
            makeCitation({
              document_id: "upload_abc123",
              document_title: "notes.md",
              kind: "upload",
              chapter: "Chương 1",
            }),
          ],
        })}
      />,
      { openAttachment: (id, chapter) => opened.push([id, chapter]) },
    );

    // The card only becomes a viewer trigger once the attachments list resolves,
    // so wait for that state rather than the generic dialog trigger.
    const trigger = await screen.findByRole("button", { name: /Xem trích dẫn từ notes\.md/ });
    await user.click(trigger);

    await waitFor(() => expect(opened).toEqual([["abc123", "Chương 1"]]));
    expect(screen.queryByText("DOC-001")).toBeNull();
  });
});
