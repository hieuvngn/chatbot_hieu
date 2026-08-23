import { MutationCache, QueryCache, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { Toaster } from "sonner";
import App from "./App";
import { ApiError, setToken } from "./lib/api";
import "./index.css";
import "highlight.js/styles/github-dark-dimmed.css";

const on401 = (error: unknown): void => {
  if (error instanceof ApiError && error.status === 401) {
    setToken(null);
    window.location.assign("/login");
  }
};

const queryClient = new QueryClient({
  queryCache: new QueryCache({ onError: on401 }),
  mutationCache: new MutationCache({ onError: on401 }),
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
      <Toaster position="top-center" richColors />
    </QueryClientProvider>
  </StrictMode>,
);
