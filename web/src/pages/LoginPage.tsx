import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api, setToken } from "@/lib/api";

const DEMO_CREDENTIALS = { username: "hieu", password: "demo-password" } as const;

export default function LoginPage() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  function useDemo(): void {
    setMode("login");
    setUsername(DEMO_CREDENTIALS.username);
    setPassword(DEMO_CREDENTIALS.password);
    setError(null);
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setPending(true);
    try {
      const fn = mode === "login" ? api.login : api.register;
      const { token } = await fn(username.trim(), password);
      setToken(token);
      navigate("/", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Đã có lỗi xảy ra.");
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 px-4 dark:bg-zinc-950">
      <div className="w-full max-w-sm rounded-2xl border bg-white p-8 shadow-sm dark:bg-zinc-900">
        <h1 className="text-2xl font-semibold tracking-tight">CourseMate</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Tư vấn môn học và hỏi đáp trên tài liệu học tập.
        </p>

        <div className="mt-6 grid grid-cols-2 gap-1 rounded-lg bg-zinc-100 p-1 dark:bg-zinc-800">
          {(["login", "register"] as const).map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => {
                setMode(tab);
                setError(null);
              }}
              className={`rounded-md py-1.5 text-sm font-medium transition ${
                mode === tab
                  ? "bg-white shadow-sm dark:bg-zinc-700"
                  : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
              }`}
            >
              {tab === "login" ? "Đăng nhập" : "Đăng ký"}
            </button>
          ))}
        </div>

        <div className="mt-4 rounded-lg border border-dashed border-zinc-300 bg-zinc-50 px-3 py-2 text-xs text-zinc-600 dark:border-zinc-700 dark:bg-zinc-900/50 dark:text-zinc-400">
          <p className="font-medium text-zinc-700 dark:text-zinc-300">
            Tài khoản demo:{" "}
            <button
              type="button"
              onClick={useDemo}
              className="font-mono underline-offset-2 hover:underline"
              title="Điền vào form đăng nhập"
            >
              hieu / demo-password
            </button>
          </p>
          <p className="mt-0.5 text-[11px] leading-snug text-zinc-500 dark:text-zinc-500">
            Chạy <code className="font-mono">uv run python -m demo_memory</code>
            {" "}để tạo user này trên <code className="font-mono">data/demo.db</code>,
            {" "}hoặc bấm <strong>Đăng ký</strong> để tạo tài khoản mới.
          </p>
        </div>

        <form onSubmit={submit} className="mt-6 space-y-4">
          <Input placeholder="Tên đăng nhập" value={username} onChange={(e) => setUsername(e.target.value)} autoFocus />
          <Input type="password" placeholder="Mật khẩu" value={password} onChange={(e) => setPassword(e.target.value)} />
          {error !== null && <p className="text-sm text-red-600">{error}</p>}
          <Button type="submit" disabled={pending} className="w-full rounded-xl">
            {pending ? "Đang xử lý…" : mode === "login" ? "Đăng nhập" : "Đăng ký"}
          </Button>
        </form>
      </div>
    </div>
  );
}