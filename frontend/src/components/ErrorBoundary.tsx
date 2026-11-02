import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RotateCcw } from "lucide-react";
import { Button } from "./Button";
import { Card } from "./Card";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Mob-Alert encountered an unexpected rendering error.", {
      error,
      componentStack: errorInfo.componentStack,
    });
  }

  render() {
    if (this.state.error) {
      return (
        <main className="error-boundary-page" aria-labelledby="app-error-title">
          <Card className="error-boundary-card">
            <span className="error-boundary-icon">
              <AlertTriangle size={22} aria-hidden="true" />
            </span>
            <h1 id="app-error-title">The app hit an unexpected error.</h1>
            <p>
              Reload the page to try again. If the problem persists, check the
              browser console for details.
            </p>
            <Button
              variant="primary"
              onClick={() => window.location.reload()}
              icon={<RotateCcw size={16} aria-hidden="true" />}
            >
              Reload app
            </Button>
          </Card>
        </main>
      );
    }
    return this.props.children;
  }
}
