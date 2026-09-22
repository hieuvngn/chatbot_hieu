import type {
  Attachment,
  ConversationMeta,
  EntityBundle,
  Features,
  Turn,
  User,
} from "./types";

const TOKEN_KEY = "cm_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string | null): void {
  if (token === null) localStorage.removeItem(TOKEN_KEY);
  else localStorage.setItem(TOKEN_KEY, token);
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (!(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const token = getToken();
  if (token !== null) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(`/api${path}`, { ...options, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body === "object" && body !== null && "detail" in body) {
        detail = String(body.detail);
      }
    } catch {
      // Fall back to status text.
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export interface StreamSource {
  document_id: string;
  document_title: string;
  chapter: string;
  course_code: string;
  kind: string;
  language: string;
  entity_type?: string;
  entity_id?: string;
}

export interface StreamCitation {
  marker: string;
  source: StreamSource;
}

export interface StreamDone {
  answer: string;
  citations: StreamCitation[];
  sources: StreamSource[];
  refused: boolean;
  rephrase_suggestion: string;
  skills_applied: string[];
}

export async function streamChat(
  conversationId: string,
  message: string,
  useWeb: boolean,
  callbacks: {
    onStart?: (payload: { skills_applied: string[]; sources: StreamSource[] }) => void;
    onDelta: (delta: string) => void;
    onDone: (result: StreamDone) => void;
    onRefused: (rephraseSuggestion: string) => void;
    onError: (detail: string) => void;
  },
  signal?: AbortSignal,
): Promise<void> {
  const token = getToken();
  const headers = new Headers();
  headers.set("Content-Type", "application/json");
  if (token !== null) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(
    `/api/conversations/${conversationId}/chat/stream`,
    {
      method: "POST",
      headers,
      body: JSON.stringify({ message, use_web: useWeb }),
      signal,
    },
  );
  if (!res.ok) {
    const body: unknown = await res.json().catch(() => ({ detail: res.statusText }));
    const detail =
      typeof body === "object" && body !== null && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : res.statusText;
    callbacks.onError(detail);
    return;
  }
  const reader = res.body?.getReader();
  if (reader === undefined) {
    callbacks.onError("Response body is empty.");
    return;
  }
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline = buffer.indexOf("\n");
    while (newline >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line.length > 0) {
        let parsed: { event: string } & Record<string, unknown>;
        try {
          parsed = JSON.parse(line) as typeof parsed;
        } catch {
          callbacks.onError(`Malformed NDJSON line: ${line}`);
          return;
        }
        switch (parsed.event) {
          case "start":
            callbacks.onStart?.({
              skills_applied:
                (parsed.skills_applied as string[] | undefined) ?? [],
              sources: (parsed.sources as StreamSource[] | undefined) ?? [],
            });
            break;
          case "token":
            callbacks.onDelta(String(parsed.delta ?? ""));
            break;
          case "done":
            callbacks.onDone({
              answer: String(parsed.answer ?? ""),
              citations: (parsed.citations as StreamCitation[] | undefined) ?? [],
              sources: (parsed.sources as StreamSource[] | undefined) ?? [],
              refused: Boolean(parsed.refused),
              rephrase_suggestion: String(parsed.rephrase_suggestion ?? ""),
              skills_applied:
                (parsed.skills_applied as string[] | undefined) ?? [],
            });
            return;
          case "refused":
            callbacks.onRefused(String(parsed.rephrase_suggestion ?? ""));
            return;
          case "error":
            callbacks.onError(String(parsed.detail ?? "Unknown error."));
            return;
          default:
            break;
        }
      }
      newline = buffer.indexOf("\n");
    }
  }
}

export interface AuthResponse {
  token: string;
  user: User;
}

export const api = {
  register: (username: string, password: string) =>
    request<AuthResponse>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  login: (username: string, password: string) =>
    request<AuthResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  me: () => request<User>("/auth/me"),
  updateProfile: (patch: { display_name?: string; language?: string }) =>
    request<User>("/users/me", { method: "PATCH", body: JSON.stringify(patch) }),

  conversations: () => request<ConversationMeta[]>("/conversations"),
  createConversation: () =>
    request<ConversationMeta>("/conversations", {
      method: "POST",
      body: JSON.stringify({}),
    }),
  renameConversation: (id: string, title: string) =>
    request<ConversationMeta>(`/conversations/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    }),
  deleteConversation: (id: string) =>
    request<void>(`/conversations/${id}`, { method: "DELETE" }),
  clearConversations: () => request<void>("/conversations", { method: "DELETE" }),
  messages: (id: string) => request<Turn[]>(`/conversations/${id}/messages`),
  chat: (id: string, message: string, useWeb: boolean) =>
    request<Turn>(`/conversations/${id}/chat`, {
      method: "POST",
      body: JSON.stringify({ message, use_web: useWeb }),
    }),

  features: () => request<Features>("/features"),
  entities: () => request<EntityBundle>("/entities"),
  attachments: (conversationId: string) =>
    request<Attachment[]>(
      `/attachments?conversation_id=${encodeURIComponent(conversationId)}`,
    ),
  uploadAttachment: (conversationId: string, file: File) => {
    const fd = new FormData();
    fd.append("conversation_id", conversationId);
    fd.append("file", file);
    return request<Attachment>("/attachments", { method: "POST", body: fd });
  },
  attachmentContent: (id: string) =>
    request<{ sections: { chapter: string; text: string }[] }>(
      `/attachments/${id}/content`,
    ),
  deleteAttachment: (id: string) =>
    request<void>(`/attachments/${id}`, { method: "DELETE" }),
};