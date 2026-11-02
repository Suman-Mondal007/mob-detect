import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  DetectionPreferencesProvider,
  useDetectionPreferences,
} from "../components/DetectionPreferences";
import { useLiveDetection } from "./useLiveDetection";

vi.mock("../api/client", () => ({
  getAlerts: vi.fn().mockResolvedValue({ alerts: [] }),
  getLiveWebSocketUrl: vi.fn().mockReturnValue("ws://localhost/ws/live"),
}));

class MockWebSocket {
  static OPEN = 1;
  static CLOSING = 2;
  static instances: MockWebSocket[] = [];

  binaryType = "blob";
  bufferedAmount = 0;
  readyState = 0;
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  sent: (string | Blob)[] = [];

  constructor(readonly url: string) {
    MockWebSocket.instances.push(this);
  }

  send(data: string | Blob) {
    this.sent.push(data);
  }

  close() {
    this.readyState = 3;
    this.onclose?.();
  }

  open() {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.();
  }

  receive(data: object) {
    this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent<string>);
  }
}

function wrapper({ children }: { children: ReactNode }) {
  return (
    <DetectionPreferencesProvider>
      {children}
    </DetectionPreferencesProvider>
  );
}

describe("useLiveDetection", () => {
  beforeEach(() => {
    window.localStorage.clear();
    MockWebSocket.instances = [];
    vi.stubGlobal("WebSocket", MockWebSocket);
    const track = {
      stop: vi.fn(),
      addEventListener: vi.fn(),
    };
    const mediaDevices = {
      enumerateDevices: vi.fn().mockResolvedValue([
        {
          kind: "videoinput",
          deviceId: "camera-1",
          label: "Test camera",
        },
      ]),
      getUserMedia: vi.fn().mockResolvedValue({
        getTracks: () => [track],
        getVideoTracks: () => [track],
      }),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    };
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: mediaDevices,
    });
  });

  it("sends saved alert settings and updates from live predictions", async () => {
    const { result } = renderHook(
      () => ({
        live: useLiveDetection(),
        preferences: useDetectionPreferences(),
      }),
      { wrapper },
    );

    act(() => {
      result.current.preferences.updatePreferences({
        alertThreshold: 0.7,
        sensitivity: "high",
      });
    });
    await act(async () => {
      await result.current.live.start();
    });

    const socket = MockWebSocket.instances[0];
    expect(socket).toBeDefined();
    act(() => socket.open());
    expect(socket.sent).toContain(
      JSON.stringify({ threshold: 0.7, sensitivity: "high" }),
    );

    act(() => {
      socket.receive({
        label: "Violence",
        confidence: 0.91,
        smoothed_confidence: 0.86,
        alert: true,
        inference_ms: 37,
        timestamp: "2026-10-29T04:00:00+00:00",
      });
    });

    await waitFor(() => {
      expect(result.current.live.prediction?.label).toBe("Violence");
      expect(result.current.live.alerts).toHaveLength(1);
    });
    expect(result.current.live.prediction?.inference_ms).toBe(37);
  });
});
