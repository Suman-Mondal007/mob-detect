import {
  Activity,
  Bell,
  Camera,
  ClipboardList,
  Cctv,
  Info,
  Menu,
  Moon,
  ShieldAlert,
  Settings,
  Sun,
  Upload,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { useTheme } from "./ThemeProvider";

const navigation = [
  { to: "/monitor", label: "Monitoring", icon: Cctv },
  { to: "/live", label: "Live", icon: Activity },
  { to: "/upload", label: "Upload", icon: Upload },
  { to: "/alerts", label: "Alerts", icon: Bell },
{ to: "/cameras", label: "Cameras", icon: Camera },
{ to: "/incidents", label: "Incidents", icon: ClipboardList },
  { to: "/about", label: "About", icon: Info },
  { to: "/settings", label: "Settings", icon: Settings },
];

export function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const { theme, toggleTheme } = useTheme();
  const menuToggleRef = useRef<HTMLButtonElement>(null);
  const location = useLocation();

  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    if (!menuOpen) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMenuOpen(false);
        menuToggleRef.current?.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [menuOpen]);

  return (
    <header className="app-header">
      <div className="header-inner">
        <NavLink
          to="/live"
          className="brand-link"
          aria-label="Mob-Alert home"
          onClick={() => setMenuOpen(false)}
        >
          <span className="brand-mark">
            <ShieldAlert size={20} strokeWidth={2.2} aria-hidden="true" />
          </span>
          <span className="brand-name">Mob<span>-</span>Alert</span>
          <span className="brand-version">PROTOTYPE</span>
        </NavLink>

        <nav className="desktop-nav" aria-label="Primary navigation">
          {navigation.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `nav-link ${isActive ? "nav-link-active" : ""}`
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="header-actions">
          <button
            type="button"
            className="theme-toggle"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
            title={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
          >
            {theme === "light" ? (
              <Moon size={18} aria-hidden="true" />
            ) : (
              <Sun size={18} aria-hidden="true" />
            )}
          </button>
          <button
            ref={menuToggleRef}
            type="button"
            className="mobile-menu-toggle"
            onClick={() => setMenuOpen((open) => !open)}
            aria-expanded={menuOpen}
            aria-controls="mobile-navigation"
            aria-label={menuOpen ? "Close navigation menu" : "Open navigation menu"}
          >
            {menuOpen ? (
              <X size={20} aria-hidden="true" />
            ) : (
              <Menu size={20} aria-hidden="true" />
            )}
          </button>
        </div>
      </div>

      {menuOpen && (
        <nav
          id="mobile-navigation"
          className="mobile-nav"
          aria-label="Mobile navigation"
        >
          {navigation.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              onClick={() => setMenuOpen(false)}
              className={({ isActive }) =>
                `mobile-nav-link ${isActive ? "mobile-nav-link-active" : ""}`
              }
            >
              <Icon size={18} aria-hidden="true" />
              {label}
            </NavLink>
          ))}
        </nav>
      )}
    </header>
  );
}
