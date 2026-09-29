import path from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

/**
 * Two suites, because they need different environments:
 *
 * - `component` (jsdom) renders real components with `@/lib/api` mocked, so
 *   component-to-component wiring is exercised without a network.
 * - `api` (node) runs the real `api.ts` against a real uvicorn + SQLite
 *   process started by `src/test/globalSetup.ts`, covering the client-to-
 *   backend half of the contract.
 */
export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  test: {
    projects: [
      {
        plugins: [react()],
        resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
        test: {
          name: "component",
          environment: "jsdom",
          globals: false,
          include: ["src/**/*.test.tsx"],
          setupFiles: ["src/test/setupDom.ts"],
        },
      },
      {
        test: {
          name: "api",
          environment: "node",
          globals: false,
          include: ["src/**/*.node.test.ts"],
          globalSetup: ["src/test/globalSetup.ts"],
          setupFiles: ["src/test/setupNode.ts"],
          fileParallelism: false,
        },
      },
    ],
  },
});
