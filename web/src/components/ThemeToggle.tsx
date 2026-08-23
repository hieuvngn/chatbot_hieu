import { Monitor, Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useTheme } from "@/lib/theme";

export default function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger render={<Button variant="ghost" size="icon" aria-label="Đổi giao diện" />}>
        {theme === "dark" ? <Moon size={18} /> : theme === "light" ? <Sun size={18} /> : <Monitor size={18} />}
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => setTheme("light")}><Sun size={14} className="mr-2" />Sáng</DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("dark")}><Moon size={14} className="mr-2" />Tối</DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("system")}><Monitor size={14} className="mr-2" />Theo hệ thống</DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
