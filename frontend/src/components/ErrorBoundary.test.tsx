import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ErrorBoundary } from "./ErrorBoundary";

function BrokenView(): never {
  throw new Error("Render failed");
}

describe("ErrorBoundary", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders a reload action when a child throws", () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    vi.spyOn(window.console, "error").mockImplementation(() => undefined);
    const preventDefaultErrorLog = (event: ErrorEvent) => event.preventDefault();
    window.addEventListener("error", preventDefaultErrorLog);

    render(
      <ErrorBoundary>
        <BrokenView />
      </ErrorBoundary>,
    );
    window.removeEventListener("error", preventDefaultErrorLog);

    expect(
      screen.getByRole("heading", {
        name: "The app hit an unexpected error.",
      }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Reload app" })).toBeDefined();
  });
});
