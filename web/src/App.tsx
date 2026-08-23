import { Navigate, Route, Routes } from "react-router-dom";
import RequireAuth from "@/components/RequireAuth";
import ChatPage from "./pages/ChatPage";
import LoginPage from "./pages/LoginPage";
import SettingsPage from "./pages/SettingsPage";

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<RequireAuth />}>
        <Route path="/" element={<ChatPage />} />
        <Route path="/c/:conversationId" element={<ChatPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
