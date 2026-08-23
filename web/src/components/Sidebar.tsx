import { Globe, MoreHorizontal, Plus } from "lucide-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { api } from "@/lib/api";
import AttachmentsSection from "@/components/AttachmentsSection";
import { useWebToggle } from "@/lib/web-toggle-context";

export default function Sidebar({ onNavigated }: { onNavigated?: () => void }) {
  const navigate = useNavigate();
  const { conversationId } = useParams();
  const queryClient = useQueryClient();
  const { useWeb, setUseWeb } = useWebToggle();
  const { data: conversations = [] } = useQuery({
    queryKey: ["conversations"],
    queryFn: api.conversations,
  });
  const { data: features } = useQuery({ queryKey: ["features"], queryFn: api.features });

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ["conversations"] });

  const createMutation = useMutation({
    mutationFn: api.createConversation,
    onSuccess: (created) => {
      void invalidate();
      onNavigated?.();
      navigate(`/c/${created.id}`);
    },
    onError: (error) => toast.error(String(error)),
  });
  const renameMutation = useMutation({
    mutationFn: ({ id, title }: { id: string; title: string }) => api.renameConversation(id, title),
    onSuccess: invalidate,
    onError: (error) => toast.error(String(error)),
  });
  const deleteMutation = useMutation({
    mutationFn: api.deleteConversation,
    onSuccess: (_data, deletedId) => {
      void invalidate();
      if (deletedId === conversationId) navigate("/");
    },
    onError: (error) => toast.error(String(error)),
  });

  return (
    <aside className="flex h-full w-full flex-col gap-2 p-3">
      <Button className="justify-start gap-2 rounded-xl" onClick={() => createMutation.mutate()} disabled={createMutation.isPending}>
        <Plus size={16} /> Chat mới
      </Button>

      <nav className="min-h-0 flex-1 space-y-0.5 overflow-y-auto pt-2">
        <p className="px-2 pb-1 text-xs font-medium uppercase tracking-wide text-zinc-400">Lịch sử chat</p>
        {conversations.map((conversation) => (
          <ConversationRow
            key={conversation.id}
            id={conversation.id}
            title={conversation.title}
            active={conversation.id === conversationId}
            onOpen={() => {
              onNavigated?.();
              navigate(`/c/${conversation.id}`);
            }}
            onRename={(title) => renameMutation.mutate({ id: conversation.id, title })}
            onDelete={() => deleteMutation.mutate(conversation.id)}
          />
        ))}
        {conversations.length === 0 && <p className="px-2 text-sm text-zinc-400">Chưa có chat nào.</p>}
      </nav>

      <AttachmentsSection />

      {features?.has_web_search && (
        <label className="flex cursor-pointer items-center justify-between rounded-xl px-2 py-1.5 text-sm hover:bg-zinc-50 dark:hover:bg-zinc-900">
          <span className="flex items-center gap-2"><Globe size={14} /> Tìm kiếm web</span>
          <Switch checked={useWeb} onCheckedChange={setUseWeb} />
        </label>
      )}

      <p className="px-2 pb-1 text-[11px] leading-snug text-zinc-400">
        <Link to="/settings" className="underline-offset-2 hover:underline" onClick={onNavigated}>
          Settings
        </Link>
      </p>
    </aside>
  );
}

function ConversationRow(props: {
  id: string;
  title: string;
  active: boolean;
  onOpen: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
}) {
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState(props.title);
  return (
    <div className={`group flex items-center gap-1 rounded-lg px-2 ${props.active ? "bg-zinc-100 dark:bg-zinc-800" : "hover:bg-zinc-50 dark:hover:bg-zinc-900"}`}>
      <button type="button" onClick={props.onOpen} className="min-w-0 flex-1 truncate py-1.5 text-left text-sm">
        {props.title}
      </button>
      <DropdownMenu>
        <DropdownMenuTrigger render={<Button variant="ghost" size="icon" className="h-7 w-7 opacity-0 transition group-hover:opacity-100" aria-label="Tùy chọn" />}>
          <MoreHorizontal size={14} />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          <DropdownMenuItem onClick={() => { setDraft(props.title); setRenaming(true); }}>Đổi tên</DropdownMenuItem>
          <DropdownMenuItem className="text-red-600" onClick={props.onDelete}>Xóa</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog open={renaming} onOpenChange={setRenaming}>
        <DialogContent className="max-w-sm rounded-2xl">
          <DialogHeader><DialogTitle>Đổi tên chat</DialogTitle></DialogHeader>
          <Input value={draft} onChange={(e) => setDraft(e.target.value)} maxLength={50} />
          <DialogFooter>
            <Button
              onClick={() => {
                const trimmed = draft.trim();
                if (trimmed.length > 0) props.onRename(trimmed);
                setRenaming(false);
              }}
            >
              Lưu
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
