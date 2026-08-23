import type { Attachment, ConversationMeta, Features, Turn, User } from "./types";

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
    const body: unknown = await res.json().catch(() => ({ detail: res.statusText }));
    const detail =
      typeof body === "object" && body !== null && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : res.statusText;
    throw new ApiError(res.status, detail);
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
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
    request<ConversationMeta>("/conversations", { method: "POST", body: JSON.stringify({}) }),
  renameConversation: (id: string, title: string) =>
    request<ConversationMeta>(`/conversations/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    }),
  deleteConversation: (id: string) => request<void>(`/conversations/${id}`, { method: "DELETE" }),
  clearConversations: () => request<void>("/conversations", { method: "DELETE" }),
  messages: (id: string) => request<Turn[]>(`/conversations/${id}/messages`),
  chat: (id: string, message: string, useWeb: boolean) =>
    request<Turn>(`/conversations/${id}/chat`, {
      method: "POST",
      body: JSON.stringify({ message, use_web: useWeb }),
    }),

  features: () => request<Features>("/features"),
  attachments: (conversationId: string) =>
    request<Attachment[]>(`/attachments?conversation_id=${encodeURIComponent(conversationId)}`),
  uploadAttachment: (conversationId: string, file: File) => {
    const fd = new FormData();
    fd.append("conversation_id", conversationId);
    fd.append("file", file);
    return request<Attachment>("/attachments", { method: "POST", body: fd });
  },
  deleteAttachment: (id: string) => request<void>(`/attachments/${id}`, { method: "DELETE" }),
};
