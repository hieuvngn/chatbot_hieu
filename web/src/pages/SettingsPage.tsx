import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api } from "@/lib/api";

export default function SettingsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { data: me } = useQuery({ queryKey: ["me"], queryFn: api.me });

  const [displayName, setDisplayName] = useState("");
  const [language, setLanguage] = useState("vi");
  const [confirmText, setConfirmText] = useState("");

  useEffect(() => {
    if (me !== undefined) {
      setDisplayName(me.display_name);
      setLanguage(me.language);
    }
  }, [me]);

  const saveMutation = useMutation({
    mutationFn: () => api.updateProfile({ display_name: displayName, language }),
    onSuccess: (updated) => {
      queryClient.setQueryData(["me"], updated);
      toast.success("Đã lưu thay đổi.");
    },
    onError: (error) => toast.error(error instanceof Error ? error.message : String(error)),
  });

  const clearMutation = useMutation({
    mutationFn: api.clearConversations,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["conversations"] });
      navigate("/");
    },
    onError: (error) => toast.error(String(error)),
  });

  return (
    <div className="mx-auto w-full max-w-xl space-y-8 p-6">
      <div className="flex items-center gap-3">
        {/* Adapted for Base UI: brief had `<Button asChild ...><Link to="/">...</Link></Button>`;
            @base-ui/react Button has no asChild — uses render prop instead
            (pattern per Header.tsx / CitationCard.tsx). */}
        <Button render={<Link to="/" />} variant="ghost" size="icon" aria-label="Quay lại">
          <ArrowLeft size={18} />
        </Button>
        <h1 className="text-xl font-semibold tracking-tight">Cài đặt</h1>
      </div>

      <section className="space-y-4 rounded-2xl border bg-white p-6 shadow-sm dark:bg-zinc-900">
        <h2 className="font-medium">Hồ sơ</h2>
        <label className="block space-y-1.5">
          <span className="text-sm text-zinc-500">Tên hiển thị</span>
          <Input value={displayName} maxLength={50} onChange={(event) => setDisplayName(event.target.value)} />
        </label>
        <label className="block space-y-1.5">
          <span className="text-sm text-zinc-500">Ngôn ngữ trả lời</span>
          <select
            value={language}
            onChange={(event) => setLanguage(event.target.value)}
            className="w-full rounded-xl border bg-transparent px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-indigo-500/40 dark:bg-zinc-900"
          >
            <option value="vi">Tiếng Việt</option>
            <option value="en">English</option>
          </select>
        </label>
        <Button
          className="rounded-xl"
          disabled={saveMutation.isPending || displayName.trim().length === 0}
          onClick={() => saveMutation.mutate()}
        >
          Lưu thay đổi
        </Button>
        <p className="text-xs text-zinc-400">Tên đăng nhập: {me?.username}</p>
      </section>

      <section className="space-y-3 rounded-2xl border border-red-200 bg-red-50 p-6 dark:border-red-900 dark:bg-red-950">
        <h2 className="font-medium text-red-800 dark:text-red-200">Vùng nguy hiểm</h2>
        <p className="text-sm text-red-700 dark:text-red-300">
          Xóa tất cả hội thoại và tin nhắn của bạn. Không thể khôi phục.
        </p>
        <Input
          placeholder="Gõ DELETE để xác nhận"
          value={confirmText}
          onChange={(event) => setConfirmText(event.target.value)}
          className="max-w-xs border-red-300 dark:border-red-800"
        />
        <Button
          variant="destructive"
          className="rounded-xl"
          disabled={confirmText !== "DELETE" || clearMutation.isPending}
          onClick={() => clearMutation.mutate()}
        >
          Xóa tất cả lịch sử
        </Button>
      </section>
    </div>
  );
}
