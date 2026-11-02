import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { ThemeProvider } from "./components/ThemeProvider";
import { DetectionPreferencesProvider } from "./components/DetectionPreferences";
import { ErrorBoundary } from "./components/ErrorBoundary";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <ThemeProvider>
        <DetectionPreferencesProvider>
          <BrowserRouter>
            <App />
          </BrowserRouter>
        </DetectionPreferencesProvider>
      </ThemeProvider>
    </ErrorBoundary>
  </React.StrictMode>,
);
