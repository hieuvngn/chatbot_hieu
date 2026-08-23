import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { Navigate, Route, Routes, useNavigate } from "react-router-dom";
import AppLayout from "@/components/AppLayout";
import RequireAuth from "@/components/RequireAuth";
import { api } from "@/lib/api";
import { WebToggleProvider } from "@/lib/web-toggle-context";
import ChatPage from "./pages/ChatPage";
import LoginPage from "./pages/LoginPage";
import SettingsPage from "./pages/SettingsPage";

function ChatRedirect() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { data: conversations, isLoading } = useQuery({
    queryKey: ["conversations"],
    queryFn: api.conversations,
  });

  useEffect(() => {
    if (isLoading || conversations === undefined) return;
    if (conversations.length > 0) {
      navigate(`/c/${conversations[0].id}`, { replace: true });
      return;
    }
    api.createConversation().then((created) => {
      void queryClient.invalidateQueries({ queryKey: ["conversations"] });
      navigate(`/c/${created.id}`, { replace: true });
    });
  }, [conversations, isLoading, navigate, queryClient]);

  return <div className="p-8 text-sm text-zinc-400">Đang tải…</div>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<RequireAuth />}>
        <Route
          element={
            <WebToggleProvider>
              <AppLayout />
            </WebToggleProvider>
          }
        >
          <Route index element={<ChatRedirect />} />
          <Route path="/c/:conversationId" element={<ChatPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Route>
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
