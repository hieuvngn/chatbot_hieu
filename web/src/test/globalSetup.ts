import { spawn } from "node:child_process";
import path from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";

/** Must match API_BASE in src/test/setupNode.ts and tests/fake_api_server.py. */
const PORT = 8766;
const BASE = `http://127.0.0.1:${PORT}`;
const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");

export async function setup(): Promise<() => void> {
  const server = spawn(
    "uv",
    ["run", "python", "-m", "tests.fake_api_server", "--port", String(PORT)],
    { cwd: REPO_ROOT, stdio: "ignore" },
  );

  for (let attempt = 0; attempt < 60; attempt += 1) {
    try {
      const res = await fetch(`${BASE}/api/features`);
      if (res.ok) return () => server.kill();
    } catch {
      // Not listening yet — keep polling.
    }
    await delay(500);
  }
  server.kill();
  throw new Error(`fake API server never became ready on ${BASE}`);
}
