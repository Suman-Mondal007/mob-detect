import {
  act,
  fireEvent,
  render,
  renderHook,
  screen,
  waitFor,
} from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SettingsPage } from "../pages/SettingsPage";
import {
  DetectionPreferencesProvider,
  useDetectionPreferences,
} from "./DetectionPreferences";

function wrapper({ children }: { children: ReactNode }) {
  return (
    <DetectionPreferencesProvider>
      {children}
    </DetectionPreferencesProvider>
  );
}

describe("useDetectionPreferences", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("uses safe defaults and persists preference updates", async () => {
    const { result } = renderHook(() => useDetectionPreferences(), { wrapper });

    expect(result.current.preferences).toEqual({
      alertThreshold: 0.8,
      sensitivity: "balanced",
      soundEnabled: false,
    });

    act(() => {
      result.current.updatePreferences({
        alertThreshold: 0.65,
        sensitivity: "high",
      });
    });

    await waitFor(() => {
      expect(
        JSON.parse(
          window.localStorage.getItem("mob-alert-detection-preferences") ?? "null",
        ),
      ).toEqual({
        alertThreshold: 0.65,
        sensitivity: "high",
        soundEnabled: false,
      });
    });
  });

  it("rejects out-of-range stored thresholds", () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    window.localStorage.setItem(
      "mob-alert-detection-preferences",
      JSON.stringify({
        alertThreshold: 2,
        sensitivity: "high",
        soundEnabled: true,
      }),
    );

    const { result } = renderHook(() => useDetectionPreferences(), { wrapper });

    expect(result.current.preferences.alertThreshold).toBe(0.8);
    expect(result.current.preferences.soundEnabled).toBe(false);
  });

  it("exposes labelled settings controls and updates the threshold", () => {
    render(
      <DetectionPreferencesProvider>
        <SettingsPage />
      </DetectionPreferencesProvider>,
    );
    const threshold = screen.getByRole("slider", { name: "Alert threshold" });

    fireEvent.change(threshold, { target: { value: "0.65" } });

    expect((threshold as HTMLInputElement).value).toBe("0.65");
    expect(
      screen.getByRole("combobox", { name: "Sensitivity" }),
    ).toBeDefined();
    expect(screen.getByRole("switch", { name: "Sound on alert" })).toBeDefined();
  });
});
