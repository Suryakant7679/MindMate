import { useEffect, useState } from "react";
export function useTheme() {
  const [dark, setDark] = useState(() => {
    try {
      return localStorage.getItem("mindmate-theme") === "dark";
    } catch {
      return false;
    }
  });
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    try {
      localStorage.setItem("mindmate-theme", dark ? "dark" : "light");
    } catch {
      /* Theme still works when storage is blocked. */
    }
  }, [dark]);
  return { dark, toggle: () => setDark((value) => !value) };
}
