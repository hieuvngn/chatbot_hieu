import { useCallback, useEffect, useState } from "react";

export type Theme = "light" | "dark" | "system";
const KEY = "cm_theme";

function apply(theme: Theme): void {
  const dark =
    theme === "dark" ||
    (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", dark);
}

export function useTheme(): { theme: Theme; setTheme: (t: Theme) => void } {
  const [theme, setThemeState] = useState<Theme>(
    () => (localStorage.getItem(KEY) as Theme | null) ?? "system",
  );

  useEffect(() => {
    apply(theme);
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const listener = () => {
      if ((localStorage.getItem(KEY) as Theme | null) === "system") apply("system");
    };
    media.addEventListener("change", listener);
    return () => media.removeEventListener("change", listener);
  }, [theme]);

  const setTheme = useCallback((next: Theme) => {
    localStorage.setItem(KEY, next);
    setThemeState(next);
  }, []);

  return { theme, setTheme };
}
