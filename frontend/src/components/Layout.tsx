import { Outlet } from "react-router-dom";
import { Navbar } from "./Navbar";

export function Layout() {
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <Navbar />
      <main id="main-content" className="page-main" tabIndex={-1}>
        <Outlet />
      </main>
      <footer className="app-footer">
        <div className="footer-inner">
          <span className="footer-brand">Mob-Alert <span>·</span> Prototype</span>
          <p>Prototype screening tool, not a validated safety system.</p>
        </div>
      </footer>
    </div>
  );
}
