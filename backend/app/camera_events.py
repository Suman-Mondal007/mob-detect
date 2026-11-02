import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from backend.app.camera_manager import CameraManager
from backend.app.schemas import CameraStatusResponse


LOGGER = logging.getLogger("mob_alert.backend")
CameraEvent = dict[str, Any]


class CameraEventHub:
    def __init__(
        self,
        camera_manager: CameraManager,
        *,
        update_interval_seconds: float = 0.5,
    ) -> None:
        self._camera_manager = camera_manager
        self._update_interval_seconds = update_interval_seconds
        self._subscribers: set[asyncio.Queue[CameraEvent]] = set()
        self._running = True
        self._last_states: dict[str, str] = {}

    def subscribe(self) -> asyncio.Queue[CameraEvent]:
        queue: asyncio.Queue[CameraEvent] = asyncio.Queue(maxsize=32)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[CameraEvent]) -> None:
        self._subscribers.discard(queue)

    async def run(self) -> None:
        while self._running:
            statuses = self._camera_manager.list_cameras()
            timestamp = datetime.now(timezone.utc).isoformat()
            status_event: CameraEvent = {
                "type": "camera_status",
                "timestamp": timestamp,
                "cameras": [status.model_dump() for status in statuses],
            }
            events = [status_event]
            for status in statuses:
                previous_state = self._last_states.get(status.camera_id)
                self._last_states[status.camera_id] = status.state
                if status.state == "ALERT" and previous_state != "ALERT":
                    events.append(
                        {
                            "type": "alert",
                            "timestamp": timestamp,
                            "camera_id": status.camera_id,
                            "confidence": status.confidence,
                            "smoothed_confidence": status.smoothed_confidence,
                            "state": status.state,
                        }
                    )

            for queue in tuple(self._subscribers):
                for event in events:
                    self._offer_latest(queue, event)
            await asyncio.sleep(self._update_interval_seconds)

    def close(self) -> None:
        self._running = False
        self._subscribers.clear()

    @staticmethod
    def _offer_latest(
        queue: asyncio.Queue[CameraEvent],
        event: CameraEvent,
    ) -> None:
        if queue.full() and event["type"] == "camera_status":
            return
        if queue.full():
            buffered_events: list[CameraEvent] = []
            try:
                while True:
                    buffered_events.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                pass
            status_index = next(
                (
                    index
                    for index, buffered_event in enumerate(buffered_events)
                    if buffered_event["type"] == "camera_status"
                ),
                None,
            )
            if status_index is None:
                LOGGER.warning("Dropping an old camera alert for a slow WebSocket client.")
                buffered_events.pop(0)
            else:
                buffered_events.pop(status_index)
            for buffered_event in buffered_events:
                queue.put_nowait(buffered_event)
        queue.put_nowait(event)
