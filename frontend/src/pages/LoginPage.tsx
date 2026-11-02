import { useState, type FormEvent } from "react";
import { Navigate } from "react-router-dom";
import { login, setAuthToken, getAuthToken } from "../api/client";
import { Button } from "../components/Button";
import { Card } from "../components/Card";
import { PageIntro } from "../components/PageIntro";

export function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  if (getAuthToken()) {
    return <Navigate to="/live" replace />;
  }

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const result = await login(username.trim(), password);
      setAuthToken(result.access_token);
      window.location.href = "/live";
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Login failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container login-page">
      <PageIntro eyebrow="Security" title="Sign in." description="Authenticate to access Mob-Alert." />
      <Card className="login-card">
        <form className="login-form" onSubmit={handleSubmit}>
          <label>
            Username
            <input value={username} onChange={(event) => setUsername(event.target.value)} required />
          </label>
          <label>
            Password
            <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} required />
          </label>
          {error && <p className="camera-test-error">{error}</p>}
          <Button type="submit" disabled={loading}>
            {loading ? "Signing in..." : "Sign in"}
          </Button>
        </form>
        <p className="login-hint">Default users: admin/admin, operator/operator, viewer/viewer.</p>
      </Card>
    </div>
  );
}
