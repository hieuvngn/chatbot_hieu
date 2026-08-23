import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

const KEY = "cm_use_web";

interface WebToggleValue {
  useWeb: boolean;
  setUseWeb: (value: boolean) => void;
}

const WebToggleContext = createContext<WebToggleValue>({
  useWeb: false,
  setUseWeb: () => undefined,
});

export function WebToggleProvider({ children }: { children: ReactNode }) {
  const [useWeb, setUseWeb] = useState<boolean>(() => localStorage.getItem(KEY) === "1");
  useEffect(() => {
    localStorage.setItem(KEY, useWeb ? "1" : "0");
  }, [useWeb]);
  return (
    <WebToggleContext.Provider value={{ useWeb, setUseWeb }}>
      {children}
    </WebToggleContext.Provider>
  );
}

export function useWebToggle(): WebToggleValue {
  return useContext(WebToggleContext);
}
