import { LogOut, Menu, Settings } from "lucide-react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Sheet, SheetContent, SheetTitle, SheetTrigger,
} from "@/components/ui/sheet";
import { api, setToken } from "@/lib/api";
import Sidebar from "./Sidebar";
import ThemeToggle from "./ThemeToggle";

export default function Header() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const { data: conversations } = useQuery({ queryKey: ["conversations"], queryFn: api.conversations });
  const { data: me } = useQuery({ queryKey: ["me"], queryFn: api.me });
  const active = conversations?.find((c) => c.id === conversationId);

  function logout(): void {
    setToken(null);
    queryClient.clear();
    navigate("/login");
  }

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b px-4">
      <div className="flex items-center gap-2">
        <Sheet open={open} onOpenChange={setOpen}>
          <SheetTrigger render={<Button variant="ghost" size="icon" className="md:hidden" aria-label="Menu" />}>
            <Menu size={18} />
          </SheetTrigger>
          <SheetContent side="left" className="w-72 p-0">
            <SheetTitle className="sr-only">Menu</SheetTitle>
            <Sidebar onNavigated={() => setOpen(false)} />
          </SheetContent>
        </Sheet>
        <span className="hidden font-semibold tracking-tight md:inline">CourseMate</span>
        <span className="mx-2 hidden truncate text-sm text-zinc-500 md:inline">{active?.title ?? ""}</span>
      </div>
      <div className="flex items-center gap-1">
        <ThemeToggle />
        <DropdownMenu>
          <DropdownMenuTrigger render={<Button variant="ghost" size="icon" aria-label="Tài khoản" />}>
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-indigo-600 text-xs font-semibold text-white">
              {(me?.display_name ?? "?").charAt(0).toUpperCase()}
            </span>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem render={<Link to="/settings" />}>
              <Settings size={14} className="mr-2" />Settings
            </DropdownMenuItem>
            <DropdownMenuItem onClick={logout}><LogOut size={14} className="mr-2" />Log out</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
}