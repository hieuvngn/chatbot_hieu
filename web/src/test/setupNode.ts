/**
 * Node-environment shims for the api.ts integration suite.
 *
 * api.ts is browser code: it reads the token from localStorage and calls
 * `fetch("/api/...")` with a relative URL. Both are provided here so the
 * real client code runs unmodified against a real server. Must match
 * globalSetup.ts and tests/fake_api_server.py.
 */

const API_BASE = "http://127.0.0.1:8766";

/** localStorage stands in for the browser store; keys are the two the app uses. */
const store: Record<string, string> = {};
Object.defineProperty(globalThis, "localStorage", {
  value: {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => {
      store[key] = value;
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      for (const key of Object.keys(store)) delete store[key];
    },
  },
  configurable: true,
});

const realFetch = globalThis.fetch;

globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
  if (typeof input === "string" && input.startsWith("/")) {
    return realFetch(`${API_BASE}${input}`, init);
  }
  return realFetch(input as RequestInfo, init);
}) as typeof fetch;
