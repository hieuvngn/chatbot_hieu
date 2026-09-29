import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { StreamSource } from "@/lib/api";
import type { Turn } from "@/lib/types";
import { makeTurn, renderWithProviders } from "@/test/harness";
import ChatPage from "@/pages/ChatPage";

/**
 * ChatPage owns the whole send cycle: it streams through streamChat, renders
 * the live turn through ChatMessage, then drops that turn and refetches the
 * conversation. These tests drive the cycle with a fake stream plus a
 * `persisted` list standing in for what the server stored, so the
 * component-to-component and component-to-API wiring is covered without a
 * browser.
 */

interface StreamCallbacks {
  onStart?: (payload: { skills_applied: string[]; sources: StreamSource[] }) => void;
  onDelta: (delta: string) => void;
  onDone: (result: {
    answer: string;
    citations: { marker: string; source: StreamSource }[];
    refused: boolean;
    rephrase_suggestion: string;
    skills_applied: string[];
  }) => void;
  onRefused: (rephraseSuggestion: string) => void;
  onError: (detail: string) => void;
}

const mocks = vi.hoisted(() => ({
  messages: vi.fn(),
  features: vi.fn(),
  streamChat: vi.fn(),
  conversations: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: {
    messages: mocks.messages,
    features: mocks.features,
    conversations: mocks.conversations,
  },
  getToken: () => null,
  setToken: () => undefined,
  streamChat: mocks.streamChat,
}));

const SOURCE: StreamSource = {
  document_id: "DOC-001",
  document_title: "Slides Cấu trúc dữ liệu",
  chapter: "Chương 3: Bảng băm",
  course_code: "CS101",
  kind: "slides",
  language: "vi",
};

/** What the server has stored; the refetched history renders from this. */
let persisted: Turn[] = [];

/** Drive the fake stream through the callbacks api.ts would invoke. */
function scriptStream(steps: (callbacks: StreamCallbacks) => void): void {
  mocks.streamChat.mockImplementation(
    async (_id: string, _message: string, _useWeb: boolean, callbacks: StreamCallbacks) => {
      steps(callbacks);
    },
  );
}

async function ask(user: ReturnType<typeof userEvent.setup>, question: string): Promise<void> {
  const composer = await screen.findByPlaceholderText(/Hỏi về môn học/);
  await user.type(composer, `${question}{Enter}`);
}

beforeEach(() => {
  persisted = [];
  mocks.messages.mockImplementation(async () => persisted);
  mocks.features.mockResolvedValue({ has_web_search: true });
  mocks.conversations.mockResolvedValue([]);
  mocks.streamChat.mockReset();
});

describe("ChatPage history", () => {
  it("renders stored turns through ChatMessage", async () => {
    persisted = [
      makeTurn({ role: "user", text: "câu hỏi cũ" }),
      makeTurn({ text: "câu trả lời cũ", citations: [{ marker: "[1]", source: SOURCE }] }),
    ];
    renderWithProviders(<ChatPage />);

    expect(await screen.findByText("câu hỏi cũ")).toBeDefined();
    expect(screen.getByText("câu trả lời cũ")).toBeDefined();
    expect(screen.getByText("Tài liệu tham khảo (1)")).toBeDefined();
  });

  it("shows the empty state only when the conversation has no turns", async () => {
    renderWithProviders(<ChatPage />);

    expect(await screen.findByRole("button", { name: /Giải thích bảng băm/ })).toBeDefined();
  });
});

describe("ChatPage streaming", () => {
  it("shows the question immediately, then settles on the persisted answer", async () => {
    const user = userEvent.setup();
    scriptStream((cb) => {
      cb.onStart?.({ skills_applied: ["eli5"], sources: [SOURCE] });
      cb.onDelta("Bảng băm là một cấu trúc dữ liệu.");
      persisted = [
        makeTurn({ role: "user", text: "bảng băm là gì?" }),
        makeTurn({
          text: "Bảng băm là một cấu trúc dữ liệu.",
          citations: [{ marker: "[1]", source: SOURCE }],
          skills_applied: ["eli5"],
        }),
      ];
      cb.onDone({
        answer: "Bảng băm là một cấu trúc dữ liệu.",
        citations: [{ marker: "[1]", source: SOURCE }],
        refused: false,
        rephrase_suggestion: "",
        skills_applied: ["eli5"],
      });
    });
    renderWithProviders(<ChatPage />);

    await ask(user, "bảng băm là gì?");

    expect(mocks.streamChat).toHaveBeenCalledWith(
      "conv-1",
      "bảng băm là gì?",
      false,
      expect.anything(),
      expect.anything(),
    );
    expect(await screen.findByText("Tài liệu tham khảo (1)")).toBeDefined();
    expect(screen.getByText("eli5")).toBeDefined();
    expect(screen.queryByText(/Đang suy nghĩ/)).toBeNull();
  });

  it("keeps the streamed text on screen between tokens", async () => {
    const user = userEvent.setup();
    scriptStream((cb) => {
      cb.onStart?.({ skills_applied: [], sources: [SOURCE] });
      cb.onDelta("Bảng băm là ");
    });
    renderWithProviders(<ChatPage />);

    await ask(user, "bảng băm");

    expect(await screen.findByText("Bảng băm là")).toBeDefined();
  });

  it("shows the thinking indicator until the first token arrives", async () => {
    const user = userEvent.setup();
    scriptStream(() => undefined);
    renderWithProviders(<ChatPage />);

    await ask(user, "bảng băm");

    expect(await screen.findByText(/Đang suy nghĩ/)).toBeDefined();
  });

  it("renders the persisted refusal as the amber card, not as an answer", async () => {
    const user = userEvent.setup();
    scriptStream((cb) => {
      persisted = [
        makeTurn({ role: "user", text: "câu hỏi mơ hồ" }),
        makeTurn({ refused: true, rephrase_suggestion: "Hãy hỏi lại với từ khóa khác." }),
      ];
      cb.onRefused("Hãy hỏi lại với từ khóa khác.");
    });
    renderWithProviders(<ChatPage />);

    await ask(user, "câu hỏi mơ hồ");

    expect(await screen.findByText(/Không tìm đủ tài liệu/)).toBeDefined();
    expect(screen.getByText(/Gợi ý viết lại: Hãy hỏi lại/)).toBeDefined();
    expect(screen.queryByText(/Tài liệu tham khảo/)).toBeNull();
  });

  it("shows the server error detail and clears the in-flight state", async () => {
    const user = userEvent.setup();
    scriptStream((cb) => {
      cb.onError("Internal Server Error");
    });
    renderWithProviders(<ChatPage />);

    await ask(user, "hỏi gì đó");

    expect(
      await screen.findByText(/Không gửi được câu trả lời: Internal Server Error/),
    ).toBeDefined();
    expect(screen.queryByText(/Đang suy nghĩ/)).toBeNull();
  });

  it("sends the sidebar toggle state so the backend can enable web search", async () => {
    const user = userEvent.setup();
    scriptStream((cb) => {
      cb.onDone({
        answer: "ok",
        citations: [],
        refused: false,
        rephrase_suggestion: "",
        skills_applied: [],
      });
    });
    renderWithProviders(<ChatPage />);

    await user.click(await screen.findByRole("button", { name: /Bật tìm kiếm web/ }));
    await ask(user, "tra web");

    await waitFor(() =>
      expect(mocks.streamChat).toHaveBeenCalledWith(
        "conv-1",
        "tra web",
        true,
        expect.anything(),
        expect.anything(),
      ),
    );
  });

  it("hides the web toggle when the backend reports no web search", async () => {
    mocks.features.mockResolvedValue({ has_web_search: false });
    renderWithProviders(<ChatPage />);

    expect(await screen.findByPlaceholderText(/Hỏi về môn học/)).toBeDefined();
    expect(screen.queryByRole("button", { name: /tìm kiếm web/i })).toBeNull();
  });

  it("sends a suggestion chip when the conversation is empty", async () => {
    const user = userEvent.setup();
    scriptStream((cb) => {
      cb.onDone({
        answer: "ok",
        citations: [],
        refused: false,
        rephrase_suggestion: "",
        skills_applied: [],
      });
    });
    renderWithProviders(<ChatPage />);

    await user.click(await screen.findByRole("button", { name: /Giải thích bảng băm/ }));

    await waitFor(() =>
      expect(mocks.streamChat).toHaveBeenCalledWith(
        "conv-1",
        "Giải thích bảng băm là gì?",
        false,
        expect.anything(),
        expect.anything(),
      ),
    );
  });

  it("ignores a blank submission", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ChatPage />);

    await ask(user, "   ");

    expect(mocks.streamChat).not.toHaveBeenCalled();
  });

  it("refetches the conversation after a completed stream", async () => {
    const user = userEvent.setup();
    scriptStream((cb) => {
      cb.onDone({
        answer: "ok",
        citations: [],
        refused: false,
        rephrase_suggestion: "",
        skills_applied: [],
      });
    });
    renderWithProviders(<ChatPage />);
    await screen.findByPlaceholderText(/Hỏi về môn học/);
    const before = mocks.messages.mock.calls.length;

    await ask(user, "hỏi gì");

    await waitFor(() => expect(mocks.messages.mock.calls.length).toBeGreaterThan(before));
  });
});
