import { Sun, Moon } from "lucide-react";
export function ThemeToggle({
  dark,
  toggle,
}: {
  dark: boolean;
  toggle: () => void;
}) {
  const label = `Switch to ${dark ? "light" : "dark"} mode`;
  return (
    <button
      className="theme-button"
      onClick={toggle}
      aria-label={label}
      title={label}
    >
      {dark ? <Moon size={19} /> : <Sun size={19} />}
    </button>
  );
}
