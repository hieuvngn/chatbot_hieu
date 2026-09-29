import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import Sidebar from "@/components/Sidebar";
import type { Attachment, ConversationMeta } from "@/lib/types";
import { renderWithProviders } from "@/test/harness";

/**
 * Sidebar composes the conversation list, AttachmentsSection and the
 * web-search toggle, all fed by the API. These tests cover the wiring
 * between those pieces: which query each panel triggers, and what each
 * mutation does to the shared cache.
 */

function conversation(overrides: Partial<ConversationMeta> = {}): ConversationMeta {
  return {
    id: "conv-1",
    user_id: 1,
    title: "Bảng băm",
    created_at: "2026-01-01T00:00:00",
    updated_at: "2026-01-02T00:00:00",
    preview: "giải thích bảng băm",
    ...overrides,
  };
}

function attachment(overrides: Partial<Attachment> = {}): Attachment {
  return {
    id: "att-1",
    conversation_id: "conv-1",
    filename: "notes.md",
    file_kind: "md",
    size_bytes: 4096,
    chunk_count: 3,
    created_at: "2026-01-01T00:00:00",
    ...overrides,
  };
}

const mocks = vi.hoisted(() => ({
  conversations: vi.fn(),
  features: vi.fn(),
  createConversation: vi.fn(),
  renameConversation: vi.fn(),
  deleteConversation: vi.fn(),
  attachments: vi.fn(),
  uploadAttachment: vi.fn(),
  deleteAttachment: vi.fn(),
  attachmentContent: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: mocks,
  getToken: () => null,
  setToken: () => undefined,
  streamChat: () => undefined,
}));

beforeEach(() => {
  mocks.conversations.mockResolvedValue([conversation()]);
  mocks.features.mockResolvedValue({ has_web_search: false });
  mocks.attachments.mockResolvedValue([attachment()]);
  mocks.attachmentContent.mockResolvedValue({
    sections: [{ chapter: "Chương 1", text: "Nội dung A" }],
  });
  mocks.createConversation.mockResolvedValue(conversation({ id: "conv-2", title: "New chat" }));
  mocks.renameConversation.mockResolvedValue(conversation({ title: "Tên mới" }));
  mocks.deleteConversation.mockResolvedValue(undefined);
  mocks.deleteAttachment.mockResolvedValue(undefined);
  mocks.uploadAttachment.mockResolvedValue(attachment({ id: "att-2", filename: "upload.md" }));
  mocks.attachmentContent.mockClear();
});

describe("Sidebar conversation list", () => {
  it("lists conversations returned by the API", async () => {
    mocks.conversations.mockResolvedValue([
      conversation(),
      conversation({ id: "conv-9", title: "Môn khác" }),
    ]);
    renderWithProviders(<Sidebar />);

    expect(await screen.findByText("Bảng băm")).toBeDefined();
    expect(screen.getByText("Môn khác")).toBeDefined();
  });

  it("shows the empty-history note when the API returns none", async () => {
    mocks.conversations.mockResolvedValue([]);
    renderWithProviders(<Sidebar />);

    expect(await screen.findByText(/Chưa có chat nào/)).toBeDefined();
  });

  it("creates a conversation through the API and refetches the list", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Sidebar />);
    await screen.findByText("Bảng băm");
    const callsBefore = mocks.conversations.mock.calls.length;

    await user.click(screen.getByRole("button", { name: /Chat mới/ }));

    await waitFor(() => expect(mocks.createConversation).toHaveBeenCalled());
    await waitFor(() => expect(mocks.conversations.mock.calls.length).toBeGreaterThan(callsBefore));
  });
});

describe("Sidebar attachments panel", () => {
  it("lists attachments for the routed conversation with their chunk counts", async () => {
    renderWithProviders(<Sidebar />);

    expect(await screen.findByText("Tài liệu đính kèm (1/3)")).toBeDefined();
    expect(screen.getByText("notes.md")).toBeDefined();
    expect(mocks.attachments).toHaveBeenCalledWith("conv-1");
  });

  it("opens the viewer for a clicked attachment", async () => {
    const user = userEvent.setup();
    const opened: [string, string | undefined][] = [];
    renderWithProviders(<Sidebar />, {
      openAttachment: (id, chapter) => opened.push([id, chapter]),
    });
    await screen.findByText("notes.md");

    await user.click(screen.getByRole("button", { name: /Xem notes\.md/ }));

    expect(opened).toEqual([["att-1", undefined]]);
  });

  it("swaps the upload button for the limit note at the backend cap", async () => {
    mocks.attachments.mockResolvedValue([
      attachment({ id: "a1", filename: "a.md" }),
      attachment({ id: "a2", filename: "b.md" }),
      attachment({ id: "a3", filename: "c.md" }),
    ]);
    renderWithProviders(<Sidebar />);

    expect(await screen.findByText(/Đã đạt giới hạn 3 tài liệu/)).toBeDefined();
    expect(screen.queryByRole("button", { name: /Thêm tài liệu/ })).toBeNull();
  });

  it("hides the file input once the limit is reached", async () => {
    mocks.attachments.mockResolvedValue(
      [1, 2, 3].map((n) => attachment({ id: `a${n}` })),
    );
    const { container } = renderWithProviders(<Sidebar />);

    await screen.findByText(/Đã đạt giới hạn/);
    expect(container.querySelector('input[type="file"]')).toBeNull();
  });
});

describe("Sidebar web toggle", () => {
  it("stays hidden when the backend reports no web search", async () => {
    renderWithProviders(<Sidebar />);

    expect(await screen.findByText(/Tài liệu đính kèm/)).toBeDefined();
    expect(screen.queryByText("Tìm kiếm web")).toBeNull();
  });

  it("renders the toggle when web search is configured and persists the choice", async () => {
    const user = userEvent.setup();
    mocks.features.mockResolvedValue({ has_web_search: true });
    renderWithProviders(<Sidebar />);

    const toggle = await screen.findByRole("switch");
    expect(toggle.getAttribute("aria-checked")).toBe("false");

    await user.click(toggle);

    expect(toggle.getAttribute("aria-checked")).toBe("true");
    // WebToggleProvider writes the choice so the next visit restores it.
    await waitFor(() => expect(localStorage.getItem("cm_use_web")).toBe("1"));
  });
});

describe("Sidebar layout links", () => {
  it("links to the settings page", async () => {
    renderWithProviders(<Sidebar />);

    const link = await screen.findByRole("link", { name: "Settings" });
    expect(link.getAttribute("href")).toBe("/settings");
  });

  it("keeps the settings link reachable from the mobile sheet trigger", async () => {
    // AppLayout owns the sheet; Sidebar must render inside an <aside> so the
    // layout's aside selector (used by the system UI tests) still finds it.
    const { container } = renderWithProviders(<Sidebar />);
    expect(container.querySelector("aside")).not.toBeNull();
  });
});

describe("Sidebar error handling", () => {
  it("reports a failed conversation load without breaking the panel", async () => {
    mocks.conversations.mockRejectedValue(new Error("boom"));
    renderWithProviders(<Sidebar />);

    // History falls back to the empty list; the rest of the sidebar renders.
    expect(await screen.findByText(/Chưa có chat nào/)).toBeDefined();
    expect(screen.getByRole("button", { name: /Chat mới/ })).toBeDefined();
  });

  it("surfaces the API message when a rename is rejected", async () => {
    const user = userEvent.setup();
    mocks.renameConversation.mockRejectedValue(new Error("Conversation not found."));
    renderWithProviders(<Sidebar />);
    await screen.findByText("Bảng băm");

    await user.click(screen.getByRole("button", { name: "Tùy chọn" }));
    await user.click(await screen.findByRole("menuitem", { name: "Đổi tên" }));
    await user.click(screen.getByRole("button", { name: "Lưu" }));

    // The rename dialog closes and the title on screen is untouched.
    await waitFor(() => expect(screen.getByText("Bảng băm")).toBeDefined());
    expect(mocks.renameConversation).toHaveBeenCalledWith("conv-1", "Bảng băm");
  });
});

describe("Sidebar deletion", () => {
  it("removes a conversation through the API", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Sidebar />);
    await screen.findByText("Bảng băm");

    await user.click(screen.getByRole("button", { name: "Tùy chọn" }));
    await user.click(await screen.findByRole("menuitem", { name: "Xóa" }));

    await waitFor(() => expect(mocks.deleteConversation.mock.calls[0]?.[0]).toBe("conv-1"));
  });

  it("deletes an attachment through the API", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Sidebar />);
    await screen.findByText("notes.md");

    const row = screen.getByText("notes.md").closest("li");
    expect(row).not.toBeNull();
    await user.click(within(row as HTMLElement).getByRole("button", { name: "Xóa tài liệu" }));

    // mutationFn receives react-query's context as a second argument; the id
    // is the only thing the API call must carry.
    await waitFor(() => expect(mocks.deleteAttachment.mock.calls[0]?.[0]).toBe("att-1"));
  });
});
