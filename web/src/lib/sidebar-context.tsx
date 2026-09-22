import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

const KEY = "cm_sidebar_collapsed";

interface SidebarContextValue {
  collapsed: boolean;
  toggle: () => void;
  setCollapsed: (value: boolean) => void;
}

const SidebarContext = createContext<SidebarContextValue>({
  collapsed: false,
  toggle: () => undefined,
  setCollapsed: () => undefined,
});

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState<boolean>(
    () => localStorage.getItem(KEY) === "1",
  );
  useEffect(() => {
    localStorage.setItem(KEY, collapsed ? "1" : "0");
  }, [collapsed]);
  const toggle = useCallback(() => {
    setCollapsed((prev) => !prev);
  }, []);
  return (
    <SidebarContext.Provider value={{ collapsed, toggle, setCollapsed }}>
      {children}
    </SidebarContext.Provider>
  );
}

export function useSidebar(): SidebarContextValue {
  return useContext(SidebarContext);
}