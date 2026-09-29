import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { api, setToken, streamChat } from "./api";
import type { StreamDone } from "./api";

/**
 * The real client against a real server.
 *
 * globalSetup.ts starts `tests/fake_api_server.py` — the actual FastAPI app
 * (routing, Pydantic, multipart, SQLite) with only the LLM pipeline stubbed.
 * Nothing here is mocked, so these are the tests that prove the UI's fetch
 * calls still line up with the backend and that what the server persists is
 * what comes back.
 */

let conversationId = "";

beforeAll(async () => {
  const auth = await api.register("vitest-user", "pw-vitest");
  setToken(auth.token);
  const created = await api.createConversation();
  conversationId = created.id;
});

afterAll(async () => {
  await api.deleteConversation(conversationId);
  setToken(null);
});

describe("auth", () => {
  it("registers, then resolves the stored user through me()", async () => {
    const me = await api.me();
    expect(me.username).toBe("vitest-user");
    expect(me.language).toBe("vi");
  });

  it("rejects a wrong password with the server's detail", async () => {
    await expect(api.login("vitest-user", "wrong")).rejects.toThrow(
      /Invalid username or password/,
    );
  });

  it("refuses authenticated calls once the token is cleared", async () => {
    const token = localStorage.getItem("cm_token");
    setToken(null);
    await expect(api.me()).rejects.toMatchObject({ status: 401 });
    setToken(token);
  });

  it("persists a profile change and returns the updated user", async () => {
    const updated = await api.updateProfile({ display_name: "Hieu", language: "en" });
    expect(updated.display_name).toBe("Hieu");
    expect((await api.me()).language).toBe("en");
  });
});

describe("conversations", () => {
  it("creates, lists, renames and deletes through the API", async () => {
    const created = await api.createConversation();
    expect(created.id).not.toBe(conversationId);

    const listed = await api.conversations();
    expect(listed.map((c) => c.id)).toContain(created.id);

    const renamed = await api.renameConversation(created.id, "Tên mới");
    expect(renamed.title).toBe("Tên mới");
    expect((await api.conversations()).find((c) => c.id === created.id)?.title).toBe("Tên mới");

    await api.deleteConversation(created.id);
    expect((await api.conversations()).map((c) => c.id)).not.toContain(created.id);
  });

  it("rejects a rename on a conversation the user does not own", async () => {
    const ownerToken = localStorage.getItem("cm_token");
    const other = await api.register("vitest-other", "pw");
    setToken(other.token);
    const foreign = await api.createConversation();

    setToken(ownerToken);
    await expect(api.renameConversation(foreign.id, "cướp")).rejects.toMatchObject({
      status: 404,
    });
    await expect(api.attachments(foreign.id)).rejects.toMatchObject({ status: 404 });

    setToken(other.token);
    await api.deleteConversation(foreign.id);
    setToken(ownerToken);
  });
});

describe("attachments", () => {
  it("uploads a file and returns the metadata the sidebar renders", async () => {
    const file = new File(["# Bảng băm\n\nBảng băm là cấu trúc dữ liệu.\n"], "bang-ham.md", {
      type: "text/markdown",
    });
    const meta = await api.uploadAttachment(conversationId, file);

    expect(meta.filename).toBe("bang-ham.md");
    expect(meta.file_kind).toBe("md");
    expect(meta.chunk_count).toBeGreaterThan(0);

    const listed = await api.attachments(conversationId);
    expect(listed.map((a) => a.id)).toContain(meta.id);
  });

  it("returns the stored sections, keyed by heading, for the viewer", async () => {
    const file = new File(["# Chuong 1\nNoi dung A\n# Chuong 2\nNoi dung B\n"], "notes.md", {
      type: "text/markdown",
    });
    const meta = await api.uploadAttachment(conversationId, file);

    const content = await api.attachmentContent(meta.id);
    expect(content.sections.map((s) => s.chapter)).toEqual(["Chuong 1", "Chuong 2"]);
    expect(content.sections[0].text).toContain("Noi dung A");

    await api.deleteAttachment(meta.id);
    await api.deleteAttachment(meta.id).catch((error: { status: number }) => {
      expect(error.status).toBe(404);
    });
  });

  it("rejects an unsupported file type with the server's reason", async () => {
    const file = new File(["MZ"], "virus.exe", { type: "application/octet-stream" });
    await expect(api.uploadAttachment(conversationId, file)).rejects.toThrow();
  });
});

describe("features and entities", () => {
  it("reports the web-search capability the sidebar gates the toggle on", async () => {
    const features = await api.features();
    expect(typeof features.has_web_search).toBe("boolean");
  });

  it("returns an entity bundle with all four collections populated", async () => {
    const bundle = await api.entities();
    for (const key of ["departments", "instructors", "programs", "terms"] as const) {
      expect(bundle[key].length).toBeGreaterThan(0);
      expect(bundle[key][0].id).not.toBe("");
      expect(typeof bundle[key][0].detail).toBe("object");
    }
  });
});

describe("streaming chat", () => {
  it("streams start, tokens and done, then persists the exchange", async () => {
    const deltas: string[] = [];
    let started: { skills_applied: string[] } | undefined;
    let done: StreamDone | undefined;

    await streamChat(conversationId, "giải thích bảng băm là gì?", false, {
      onStart: (meta) => {
        started = meta;
      },
      onDelta: (delta) => void deltas.push(delta),
      onDone: (result) => {
        done = result;
      },
      onError: (detail) => {
        throw new Error(`unexpected stream error: ${detail}`);
      },
      onRefused: () => {
        throw new Error("a grounded question must not be refused");
      },
    });

    expect(started?.skills_applied).toEqual(["eli5"]);
    expect(deltas.join("")).toBe(done?.answer);
    expect(done?.citations.length).toBeGreaterThan(0);

    // The server appends the exchange, so a later fetch must return it.
    const history = await api.messages(conversationId);
    const last = history[history.length - 1];
    expect(last.role).toBe("assistant");
    expect(last.text).toBe(done?.answer);
    expect(last.citations.length).toBeGreaterThan(0);
  });

  it("delivers a refusal with its rephrase suggestion", async () => {
    let suggestion = "";
    await streamChat(conversationId, "câu hỏi không rõ", false, {
      onDelta: () => undefined,
      onDone: () => {
        throw new Error("a refused stream must not report done");
      },
      onRefused: (value) => {
        suggestion = value;
      },
      onError: (detail) => {
        throw new Error(`unexpected stream error: ${detail}`);
      },
    });

    expect(suggestion).toBe("Hãy hỏi lại với từ khóa khác.");
  });

  it("reports the server's status when the conversation is unknown", async () => {
    const detail = await new Promise<string>((resolve) => {
      void streamChat("no-such-conversation", "hỏi gì", false, {
        onDelta: () => undefined,
        onDone: () => resolve("done"),
        onRefused: () => resolve("refused"),
        onError: resolve,
      });
    });

    expect(detail).toMatch(/Conversation not found/);
  });
});

describe("non-streaming chat", () => {
  it("answers through POST /chat with a turn the UI can render", async () => {
    const turn = await api.chat(conversationId, "bảng băm là gì?", false);

    expect(turn.role).toBe("assistant");
    expect(turn.text).not.toBe("");
    expect(turn.refused).toBe(false);
    expect(turn.citations[0].marker).toBe("[1]");
    expect(turn.citations[0].source.document_id).toBe("DOC-001");
  });

  it("rejects an empty message", async () => {
    await expect(api.chat(conversationId, "   ", false)).rejects.toMatchObject({ status: 400 });
  });
});
