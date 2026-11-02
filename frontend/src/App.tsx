import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { AboutPage } from "./pages/AboutPage";
import { AlertsPage } from "./pages/AlertsPage";
import { CamerasPage } from "./pages/CamerasPage";
import { IncidentsPage } from "./pages/IncidentsPage";
import { LivePage } from "./pages/LivePage";
import { LoginPage } from "./pages/LoginPage";
import { SystemHealthPage } from "./pages/SystemHealthPage";
import { MonitoringPage } from "./pages/MonitoringPage";
import { SettingsPage } from "./pages/SettingsPage";
import { UploadPage } from "./pages/UploadPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Navigate to="/live" replace />} />
        <Route path="monitor" element={<MonitoringPage />} />
        <Route path="live" element={<LivePage />} />
        <Route path="upload" element={<UploadPage />} />
        <Route path="alerts" element={<AlertsPage />} />
<Route path="cameras" element={<CamerasPage />} />
<Route path="incidents" element={<IncidentsPage />} />
        <Route path="about" element={<AboutPage />} />
        <Route path="settings" element={<SettingsPage />} />
<Route path="login" element={<LoginPage />} />
<Route path="health" element={<SystemHealthPage />} />
        <Route path="*" element={<Navigate to="/live" replace />} />
      </Route>
    </Routes>
  );
}
