import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ManagedCamera } from "../api/client";
import { MonitoringPage } from "./MonitoringPage";
import { useCameraMonitoring } from "../hooks/useCameraMonitoring";

vi.mock("../hooks/useCameraMonitoring", () => ({
  useCameraMonitoring: vi.fn(),
}));

function makeCamera(id: string, overrides: Partial<ManagedCamera> = {}): ManagedCamera {
  return {
    id,
    camera_id: id,
    name: `Camera ${id}`,
    location: `Zone ${id}`,
    source: "webcam:0",
    source_type: "webcam",
    source_uri: "0",
    enabled: true,
    created_at: "2026-10-29T00:00:00Z",
    online: true,
    last_frame_at: "2026-10-29T11:00:00Z",
    last_inference_at: "2026-10-29T11:00:00Z",
    inference_ms: 42,
    predicted_class: "NonViolence",
    confidence: 0.9,
    smoothed_confidence: 0.15,
    alert: false,
    state: "NORMAL",
    input_fps: 15,
    inference_fps: 3,
    error: null,
    ...overrides,
  };
}

const cameraFixtures = [
  makeCamera("1"),
  makeCamera("2"),
  makeCamera("3"),
  makeCamera("4"),
  makeCamera("5", {
    state: "ALERT",
    alert: true,
    confidence: 0.96,
    smoothed_confidence: 0.91,
  }),
  makeCamera("6", { online: false, state: "NORMAL" }),
];

const monitoringHookResult = {
  cameras: cameraFixtures,
  connectionState: "connected" as const,
  health: { model_loaded: true, device: "cuda:0", version: "0.1.0" },
  incidents: [],
  error: null,
  incidentError: null,
  lastAlert: null,
  refreshIncidents: vi.fn(),
  retry: vi.fn(),
};

describe("MonitoringPage", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    vi.mocked(useCameraMonitoring).mockReturnValue(monitoringHookResult);
  });

  it("shows operational status and moves alert cameras to the first tile", async () => {
    render(<MonitoringPage />);

    expect(screen.getByRole("heading", { name: "Monitoring" })).toBeTruthy();
    expect(screen.getByText("1", { selector: "strong" })).toBeTruthy();
    expect(screen.getByText("GPU · cuda:0")).toBeTruthy();
    expect(screen.getByText("Latency healthy")).toBeTruthy();
    expect(screen.getByText("Prototype screening tool, not a validated safety system.")).toBeTruthy();

    const tiles = document.querySelectorAll(".monitor-camera-tile");
    expect(tiles).toHaveLength(4);
    expect(tiles[0].getAttribute("data-camera-id")).toBe("5");
    expect(tiles[0].classList.contains("is-alert")).toBe(true);
    await waitFor(() => {
      expect(screen.getByRole("img", { name: "Live feed from Camera 5" })).toBeTruthy();
    });
  });

  it("supports single-camera paging and opens the focus details", async () => {
    render(<MonitoringPage />);
    fireEvent.click(screen.getByTitle("Single camera"));

    let tiles = document.querySelectorAll(".monitor-camera-tile");
    expect(tiles).toHaveLength(1);
    expect(tiles[0].getAttribute("data-camera-id")).toBe("5");

    fireEvent.click(screen.getByRole("button", { name: "Next camera page" }));
    tiles = document.querySelectorAll(".monitor-camera-tile");
    expect(tiles[0].getAttribute("data-camera-id")).toBe("1");
    fireEvent.click(tiles[0]);

    expect(await screen.findByRole("dialog", { name: "Camera 1" })).toBeTruthy();
    expect(screen.getByRole("img", { name: "Full camera feed from Camera 1" })).toBeTruthy();
    expect(screen.getByText("Grid paused while focus view is open")).toBeTruthy();
    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("supports fullscreen with the full-quality camera stream", async () => {
    const requestFullscreen = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(HTMLElement.prototype, "requestFullscreen", {
      configurable: true,
      value: requestFullscreen,
    });
    render(<MonitoringPage />);

    const tile = document.querySelector(".monitor-camera-tile");
    expect(tile).not.toBeNull();
    fireEvent.click(tile!);
    fireEvent.click(tile!);
    fireEvent.doubleClick(tile!);

    expect(requestFullscreen).toHaveBeenCalledOnce();
    expect(screen.queryByRole("dialog")).toBeNull();
    await waitFor(() => {
      const image = document.querySelector(".monitor-stream-image");
      expect(image).not.toBeNull();
      expect(image?.getAttribute("src")).toContain("quality=full");
    });
  });

  it("only attaches streams to tiles observed in or near the viewport", async () => {
    class MockIntersectionObserver {
      static instances: MockIntersectionObserver[] = [];
      readonly targets: Element[] = [];

      constructor(private readonly callback: IntersectionObserverCallback) {
        MockIntersectionObserver.instances.push(this);
      }

      observe(target: Element) {
        this.targets.push(target);
      }

      disconnect() {}

      trigger(target: Element, isIntersecting: boolean) {
        this.callback(
          [{ target, isIntersecting } as IntersectionObserverEntry],
          this as unknown as IntersectionObserver,
        );
      }
    }

    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
    render(<MonitoringPage />);
    await waitFor(() => {
      expect(MockIntersectionObserver.instances).toHaveLength(4);
    });

    expect(document.querySelectorAll(".monitor-stream-image")).toHaveLength(0);
    const alertTile = document.querySelector('[data-camera-id="5"]');
    expect(alertTile).not.toBeNull();
    MockIntersectionObserver.instances[0].trigger(alertTile!, true);
    await waitFor(() => {
      expect(document.querySelectorAll(".monitor-stream-image")).toHaveLength(1);
    });
  });

  it("offers alarm and notification controls and supports keyboard layout shortcuts", () => {
    render(<MonitoringPage />);

    expect(screen.getByRole("button", { name: "Enable alarm sound" })).toBeTruthy();
    expect(document.querySelector(".monitoring-alarm-actions")?.textContent).toMatch(
      /notifications/i,
    );
    expect(screen.getByRole("complementary", { name: "Active incidents" })).toBeTruthy();

    fireEvent.keyDown(window, { key: "3" });
    expect(screen.getByTitle("3 by 3 cameras").getAttribute("aria-pressed")).toBe("true");

    fireEvent.keyDown(window, { key: "m" });
    expect(screen.getByText("Alarm is muted.")).toBeTruthy();
  });

  it("lists active incidents and opens the camera focus view", () => {
    vi.mocked(useCameraMonitoring).mockReturnValue({
      ...monitoringHookResult,
      incidents: [
        {
          id: 8,
          camera_id: "5",
          start_time: "2026-10-29T11:00:00Z",
          end_time: null,
          peak_confidence: 0.91,
          snapshot_url: "/incidents/8/snapshot",
          clip_url: null,
          status: "new",
        },
      ],
    });
    render(<MonitoringPage />);

    expect(screen.getAllByText("Camera 5").length).toBeGreaterThan(0);
    expect(screen.getByRole("img", { name: "Incident snapshot from Camera 5" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "View Camera 5" }));
    expect(screen.getByRole("dialog", { name: "Camera 5" })).toBeTruthy();
  });
});
