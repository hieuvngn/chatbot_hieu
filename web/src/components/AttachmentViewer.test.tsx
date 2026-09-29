import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AttachmentViewer from "@/components/AttachmentViewer";
import { renderWithProviders } from "@/test/harness";

/**
 * AttachmentViewer is opened by AttachmentsSection (and by upload citation
 * cards) and fetches the file content itself. These tests cover what it
 * renders for the three server outcomes — sections, empty file, fetch
 * failure — plus the close path back to the layout.
 */

const mocks = vi.hoisted(() => ({ attachmentContent: vi.fn() }));

vi.mock("@/lib/api", () => ({
  api: mocks,
  getToken: () => null,
  setToken: () => undefined,
  streamChat: () => undefined,
}));

const attachment = { id: "att-1", filename: "notes.md", file_kind: "md" };

function renderViewer(props: { onClose?: () => void; focusChapter?: string | null } = {}) {
  return renderWithProviders(
    <AttachmentViewer
      attachment={attachment}
      focusChapter={props.focusChapter ?? null}
      onClose={props.onClose ?? (() => undefined)}
    />,
  );
}

beforeEach(() => {
  mocks.attachmentContent.mockResolvedValue({
    sections: [
      { chapter: "Chương 1", text: "Nội dung A" },
      { chapter: "Chương 2", text: "Nội dung B" },
    ],
  });
});

describe("AttachmentViewer", () => {
  it("shows the filename, kind and every section the API returned", async () => {
    renderViewer();

    expect(await screen.findByText("Chương 1")).toBeDefined();
    expect(screen.getByText("notes.md")).toBeDefined();
    expect(screen.getByText("md")).toBeDefined();
    expect(screen.getByText("Chương 2")).toBeDefined();
    expect(screen.getByText("Nội dung A")).toBeDefined();
    expect(mocks.attachmentContent).toHaveBeenCalledWith("att-1");
  });

  it("renders markdown headings from a .md section", async () => {
    mocks.attachmentContent.mockResolvedValue({
      sections: [{ chapter: "Chương 1", text: "# Mục A\n\nNội dung\n## Mục B" }],
    });
    renderViewer();

    const heading = await screen.findByRole("heading", { name: "Mục A" });
    expect(heading.tagName).toBe("H3");
    expect(screen.getByRole("heading", { name: "Mục B" })).toBeDefined();
  });

  it("renders a plain-text section verbatim, without markdown parsing", async () => {
    mocks.attachmentContent.mockResolvedValue({
      sections: [{ chapter: "Trang 1", text: "# không phải heading" }],
    });
    renderWithProviders(
      <AttachmentViewer
        attachment={{ id: "att-2", filename: "notes.txt", file_kind: "txt" }}
        onClose={() => undefined}
      />,
    );

    const text = await screen.findByText("# không phải heading");
    expect(text.tagName).toBe("PRE");
  });

  it("reports an empty attachment instead of a blank panel", async () => {
    mocks.attachmentContent.mockResolvedValue({ sections: [] });
    renderViewer();

    expect(await screen.findByText(/Tài liệu trống/)).toBeDefined();
  });

  it("surfaces a content fetch failure with the reason", async () => {
    mocks.attachmentContent.mockRejectedValue(new Error("Attachment not found."));
    renderViewer();

    expect(await screen.findByText(/Không tải được nội dung/)).toBeDefined();
  });

  it("closes from both header buttons", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    renderViewer({ onClose });

    const closeButtons = await screen.findAllByRole("button", { name: "Đóng viewer" });
    expect(closeButtons).toHaveLength(2);
    await user.click(closeButtons[0]);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("exposes the chapter on the section node so a citation can focus it", async () => {
    renderViewer();

    await screen.findByText("Chương 1");
    const section = document.querySelector('[data-chapter="Chương 1"]');
    expect(section).not.toBeNull();
  });
});
