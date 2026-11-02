import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AboutPage } from "./AboutPage";

describe("AboutPage", () => {
  it("explains the workflow, limitations, and video retention", () => {
    const { container } = render(<AboutPage />);

    expect(screen.getByRole("heading", { name: "How it works" })).toBeDefined();
    expect(
      screen.getByRole("heading", { name: "Limitations to keep in mind" }),
    ).toBeDefined();
    expect(
      screen.getByRole("heading", { name: "Privacy and video handling" }),
    ).toBeDefined();
    expect(container.textContent).toContain(
      "Uploaded video is processed from a temporary file and deleted when analysis finishes.",
    );
    expect(container.textContent).toContain(
      "a single JPEG snapshot are retained",
    );
    expect(container.textContent).toContain(
      "downscaled frames to the configured backend",
    );
  });
});
