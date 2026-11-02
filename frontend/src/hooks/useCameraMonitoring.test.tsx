import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCameraMonitoring } from "./useCameraMonitoring";

vi.mock("../api/client", () => ({
  getCameras: vi.fn(),
  getCameraEventsWebSocketUrl: vi.fn(() => "ws://localhost/ws/events"),
  getHealth: vi.fn(),
  getIncidents: vi.fn(),
}));

import {
  getCameras,
  getHealth,
  getIncidents,
} from "../api/client";

class MockWebSocket {
  static instances: MockWebSocket[] = [];

  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  closed = false;

  constructor(readonly url: string) {
    MockWebSocket.instances.push(this);
  }

  close() {
    this.closed = true;
    this.onclose?.();
  }

  open() {
    this.onopen?.();
  }

  receive(value: object) {
    this.onmessage?.({ data: JSON.stringify(value) } as MessageEvent<string>);
  }
}

const camera = {
  id: "lobby",
  camera_id: "lobby",
  name: "Lobby",
  location: "Main entrance",
  source: "webcam:0",
  source_type: "webcam" as const,
  source_uri: "0",
  enabled: true,
  created_at: "2026-10-29T00:00:00Z",
  online: true,
  last_frame_at: null,
  last_inference_at: null,
  inference_ms: null,
  predicted_class: null,
  confidence: null,
  smoothed_confidence: null,
  alert: false,
  state: "NORMAL" as const,
  input_fps: 0,
  inference_fps: 0,
  error: null,
};

describe("useCameraMonitoring", () => {
  afterEach(cleanup);

  beforeEach(() => {
    MockWebSocket.instances = [];
    vi.useRealTimers();
    vi.stubGlobal("WebSocket", MockWebSocket);
    vi.mocked(getCameras).mockResolvedValue({ cameras: [camera] });
    vi.mocked(getHealth).mockResolvedValue({
      model_loaded: true,
      device: "cuda:0",
      version: "0.1.0",
    });
    vi.mocked(getIncidents).mockResolvedValue({
      incidents: [],
      total: 0,
      page: 1,
      page_size: 100,
      total_pages: 0,
    });
  });

  it("loads enabled cameras and applies event updates", async () => {
    const { result } = renderHook(() => useCameraMonitoring());
    await waitFor(() => expect(result.current.cameras).toHaveLength(1));
    expect(result.current.health?.device).toBe("cuda:0");
    expect(result.current.incidents).toEqual([]);

    const socket = MockWebSocket.instances[0];
    act(() => socket.open());
    expect(result.current.connectionState).toBe("connected");
    act(() =>
      socket.receive({
        type: "camera_status",
        timestamp: "2026-10-29T11:00:00Z",
        cameras: [
          {
            camera_id: "lobby",
            online: true,
            state: "ALERT",
            confidence: 0.95,
            smoothed_confidence: 0.9,
            inference_ms: 50,
          },
        ],
      }),
    );
    expect(result.current.cameras[0].state).toBe("ALERT");
    expect(result.current.cameras[0].name).toBe("Lobby");

    act(() =>
      socket.receive({
        type: "alert",
        timestamp: "2026-10-29T11:00:01Z",
        camera_id: "lobby",
        confidence: 0.95,
        smoothed_confidence: 0.9,
        state: "ALERT",
      }),
    );
    expect(result.current.lastAlert?.type).toBe("alert");
    expect(getIncidents).toHaveBeenCalledTimes(2);
  });

  it("reconnects after the socket closes and cleans up on unmount", async () => {
    vi.useFakeTimers();
    const { result, unmount } = renderHook(() => useCameraMonitoring());
    await act(async () => {
      await Promise.resolve();
    });
    const first = MockWebSocket.instances[0];
    act(() => first.onclose?.());
    expect(result.current.connectionState).toBe("reconnecting");

    act(() => {
      vi.advanceTimersByTime(500);
    });
    expect(MockWebSocket.instances).toHaveLength(2);
    unmount();
    expect(MockWebSocket.instances[1].closed).toBe(true);
    vi.useRealTimers();
  });
});
