# Mob-Alert

Mob-Alert is a web app for screening live camera and uploaded video with the
existing Ultralytics YOLO checkpoint. Its FastAPI backend reuses the inference
and temporal alert logic, while the React frontend provides live screening,
video analysis, and persisted alert history. Model preparation, training, and
standalone inference utilities remain available as command-line tools.

## Requirements

- Windows
- Python 3.10 or newer
- Node.js 20.19+ or 22.12+

## Frontend (Phase 8)

From the project root in PowerShell, install dependencies and create a local
environment file:

```powershell
Set-Location frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open the Vite URL printed in the terminal (by default
`http://localhost:5173`). Set `VITE_API_URL` in `frontend\.env` to the FastAPI
base URL when it differs from `http://127.0.0.1:8000`. The Live page supports
webcam selection, a local video preview, live analysis, and recent alerts.
The Upload page supports video submission, job progress, timeline review, key
frames, and JSON/CSV reports. Alerts and About pages are also available. Build
for production with `npm run build`.

The **Monitoring** page (`/monitor`) is the multi-camera control-room view. It
uses `/ws/events` for camera state updates, automatically reconnects the event
socket, and only requests grid MJPEG while a camera tile is in or near the
viewport. Choose a 1, 2×2, 3×3, 4×4, or auto-fit layout; select a tile to open a
full-quality focus view or double-click it for fullscreen.

The Monitoring page includes a browser-generated beep-beep alarm. Select
**Enable alarm sound** after opening the page to satisfy browser autoplay
restrictions; this is required again after a reload. Use the mute button and
volume slider at any time. The alarm repeats while at least one camera remains
in an unacknowledged `ALERT` state and stops when the alert is acknowledged.
Incident actions acknowledge both the incident and its camera, or mark the
incident as a false positive; failed API actions are surfaced in a toast.
Incidents awaiting review are refreshed periodically and can be opened in the
camera focus view.

Choose **Enable notifications** to grant browser notification permission.
Notifications are emitted for new alert events; when an alert is active and the
tab is in the background, the browser tab title flashes until it becomes
visible. Monitoring keyboard shortcuts are **A** (acknowledge the focused
alert, or the first active alert), **M** (mute/unmute), **F** (fullscreen the
focused or first alert camera), and **1–4** (single, 2×2, 3×3, and 4×4 layouts).
The global mute state and volume are saved in browser storage; sound enablement
is intentionally per page load because browsers require a fresh user gesture.

The browser accepts MP4, AVI, or MOV videos up to 500 MB by default. Change
`VITE_MAX_VIDEO_UPLOAD_SIZE_MB` in `frontend\.env` if the backend's
`MAX_VIDEO_UPLOAD_SIZE_MB` setting differs, then restart Vite for the change to
take effect.

### Live camera screening

Start the backend in a separate PowerShell window using the FastAPI instructions
below, then start the frontend as above. Open the Vite URL, select a camera, and
choose **Start live screening**. Grant camera permission when your browser asks.
The browser keeps the preview local and sends downscaled JPEG frames (up to 320
pixels wide) about every 250 ms over `/ws/live`. Use **Stop screening** to close
the WebSocket and release the camera. Camera access requires HTTPS or
`localhost`; remote deployments must serve the frontend over HTTPS.

### Multi-camera ingestion (Phase 1)

The CCTV module adds persistent camera management alongside the existing live
webcam, upload-analysis, and alerts features. Cameras are stored in the additive
`camera` SQLite table. CRUD uses `POST /cameras`, `GET /cameras`,
`PUT /cameras/{id}`, and `DELETE /cameras/{id}`. Each record has an ID, name,
location, source type (`rtsp`, `file`, or `webcam`), source URI, enabled flag,
and creation time. Use `POST /cameras/{id}/test` to open a source and receive a
one-frame JPEG preview encoded as base64; `GET /cameras/{id}/snapshot` returns
the latest JPEG from an enabled camera worker.

The backend accepts RTSP/HTTP stream URLs, local video files (looped), and
webcam indexes. Source credentials remain server-side and are masked in API
responses. For example, from PowerShell while the API is running:

```powershell
$camera = @{
    id = "lobby"
    name = "Lobby"
    location = "Main building"
    source_type = "webcam"
    source_uri = "0"
    enabled = $true
} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/cameras -Method Post `
    -ContentType "application/json" -Body $camera
Invoke-RestMethod -Uri http://127.0.0.1:8000/cameras
```

For a laptop-only simulation, put at least one MP4, AVI, MOV, MKV, or M4V file
in the project-root `demo_videos\` directory, then run this once before starting
the backend. Available files are reused if fewer than six are present:

```powershell
.\.venv\Scripts\python.exe backend\seed_demo_cameras.py
```

The seed script creates or refreshes six enabled file-backed cameras
(`demo-01` through `demo-06`); each video source loops. Camera records and
enabled state persist across backend restarts.

`CAMERA_INFERENCE_FPS` controls the default per-camera rate, while
`CAMERA_TOTAL_INFERENCE_FPS` caps aggregate scheduled inference across cameras.
`MAX_CAMERAS` and `CAMERA_FRAME_WIDTH` / `CAMERA_FRAME_HEIGHT` configure the
camera limit and inference frame size. `CAMERA_DISPLAY_FRAME_WIDTH` and
`CAMERA_DISPLAY_FRAME_HEIGHT` bound the separately retained display frame;
inference continues to use the smaller inference frame. File sources restart at
end-of-file; failed camera connections reconnect with exponential backoff and
bounded read timeouts. `CAMERA_FILE_FPS` caps file playback rate.

### Camera scheduler status and benchmark (Phase 2)

`GET /cameras/status` reports each enabled worker's online state, input FPS,
measured inference FPS, latency, smoothed confidence, and alert state. When a
camera is in `ALERT`, acknowledge it with `POST
/cameras/{id}/acknowledge`. Once the smoothed signal clears, the camera enters
`COOLDOWN` for `CAMERA_ALERT_COOLDOWN_SECONDS` before returning to `NORMAL`.
The shared scheduler batches up to eight latest frames per model call and limits
the aggregate inference rate with `CAMERA_TOTAL_INFERENCE_FPS`.

### Multi-camera streaming (Phase 3)

`GET /stream/{camera_id}?quality=grid|full` streams MJPEG from the same
camera-worker frame used by the scheduler; additional viewers do not capture
frames or run inference. Grid output is capped at 480 pixels wide and 15 FPS;
full output is capped at 960 pixels wide and 15 FPS. The worker keeps a bounded
display frame separately from the smaller inference frame.

Connect one WebSocket to `/ws/events` for `camera_status`, `alert`, and periodic
`heartbeat` events. Status messages contain each camera's online state,
confidence, smoothed confidence, alert state, and inference latency. Slow
clients may miss intermediate status snapshots; alert events are prioritized
over status snapshots in the bounded client queue.

Seed six demo cameras, start the backend, then open
`http://127.0.0.1:8000/stream-test` to view up to six streams and the event
connection from the same origin. The test page requires enabled cameras with
valid sources. Stream bandwidth, JPEG encoding, and CPU/GPU use depend on the
source resolution, viewer count, and hardware; measure on the deployment
machine before sizing it for production.

To measure six enabled cameras from the project root, first seed the cameras as
described above, then run:

```powershell
.\.venv\Scripts\python.exe backend\camera_benchmark.py --duration 30
```

The console reports per-camera input FPS, measured inference FPS, batch latency,
state, and allocated/reserved/peak GPU memory against device total. Results are
runtime measurements and depend on video decode, model, CUDA/PyTorch build, and
other GPU workloads.

### Camera incidents (Phase 4)

Each camera transition into `ALERT` creates an incident in the additive
`incident` SQLite table. The record includes the camera ID, UTC start time, peak
violence confidence, a JPEG snapshot, and status (`new`, `acknowledged`, or
`false-positive`). A downscaled per-camera ring buffer retains about ten
seconds of evidence frames; when the alert ends, the buffered pre-alert frames
and available alert-period frames are written to an MP4 clip asynchronously.
The incident record is available while its clip is being written; `clip_url`
appears once the clip is ready.

Use `GET /incidents?page=1&page_size=20` to list incidents. Filter with
`camera_id`, `status`, `start_time`, `end_time`, or `search`; use
`GET /incidents/{id}` for details. Operators can use
`POST /incidents/{id}/acknowledge` or
`POST /incidents/{id}/false-positive`. `GET /incidents/export.csv` accepts the
same filters. Snapshot and clip links are served at
`/incidents/{id}/snapshot` and `/incidents/{id}/clip`.

Set `INCIDENT_EVIDENCE_DIR`, `INCIDENT_RETENTION_DAYS`,
`INCIDENT_RING_BUFFER_SECONDS`, `INCIDENT_EVIDENCE_FPS`,
`INCIDENT_FRAME_WIDTH`, and `INCIDENT_FRAME_HEIGHT` in `backend\.env`.
`INCIDENT_CLEANUP_INTERVAL_SECONDS` controls how often a background cleanup
checks retention. Expired evidence files are deleted and their links cleared;
incident metadata is retained. The default evidence location is
`outputs\incidents`.

## FastAPI backend (Phase 8)

Create a backend environment, install its dependencies, and copy the example
environment file:

```powershell
py -3.10 -m venv backend\.venv
.\backend\.venv\Scripts\python.exe -m pip install --upgrade pip
.\backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env
```

By default, the backend loads `models/best.pt` relative to the project root.
Edit `backend\.env` to change the model path, allowed CORS origins, prediction
settings, upload limits, or port. `CORS_ORIGINS` must be a JSON string array.
For GPU inference, install a PyTorch build compatible with the deployment
machine's CUDA drivers; otherwise the backend selects CPU.

Start the API from the project root in PowerShell:

```powershell
$env:PORT = ((Get-Content backend\.env | Where-Object { $_ -match '^PORT=' } | Select-Object -First 1) -replace '^PORT=', '')
.\backend\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port $env:PORT
```

Open `http://127.0.0.1:8000/docs`, expand **POST /predict-image**, select
**Try it out**, choose a JPEG, PNG, or WebP image, and execute. **GET /health**
reports whether the model loaded, its selected device, and the API version.
The first request may take longer while the model initializes its inference
backend.

With the API running, use the included smoke test:

```powershell
.\backend\.venv\Scripts\python.exe backend\test_api.py "path\to\image.jpg"
```

The test checks the health response and submits the image for inference. Its
HTTP timeout defaults to 90 seconds and can be changed with `--timeout`.

### Video jobs (Phase 2)

The backend accepts MP4, AVI, or MOV uploads at `POST /predict-video` and
returns a job ID with HTTP 202. Poll `GET /jobs/{job_id}` for its status and
progress; completed jobs include the overall alert verdict, a smoothed
probability timeline, alert time ranges, and up to five JPEG key frames encoded
as base64. Temporary video uploads are removed after processing. Job state and
results are held in memory and are lost when the backend restarts.

In `/docs`, upload a video with **POST /predict-video**, copy the returned
`job_id`, then enter it in **GET /jobs/{job_id}** to check progress and retrieve
the results. The same flow is covered by the backend test suite below.

Video limits and sampling/alert settings are configurable in `backend\.env`:
`MAX_VIDEO_UPLOAD_SIZE_MB`, `VIDEO_SAMPLING_FPS`,
`VIDEO_WINDOW_SIZE`, `VIDEO_ALERT_DURATION_SECONDS`,
`MAX_VIDEO_DURATION_SECONDS`, `MAX_VIDEO_FRAME_PIXELS`,
`MAX_ACTIVE_VIDEO_JOBS`, and `MAX_STORED_VIDEO_JOBS`.
The default video upload limit is 500 MB; the default sampling rate is 1 frame
per second. The alert window, probability threshold, and alert duration use the
same temporal smoothing implementation as the existing inference code.

On the frontend **Upload** page, choose or drop a video, review its local
preview, and select **Analyze video**. The page polls the job endpoint for
processing progress, then displays the verdict, probability timeline, alert
time ranges, and key frames. Select a chart point or key frame to seek the
preview. Download the result as JSON or CSV from the report buttons.

### Live webcam streaming (Phase 3)

Connect to `ws://127.0.0.1:8000/ws/live` and send JPEG frames as binary
WebSocket messages. The server keeps only the newest queued frame per client,
processes it on a worker thread, and responds with the predicted label,
confidence, smoothed Violence confidence, alert status, inference time in
milliseconds, and a UTC timestamp. Configure `MAX_LIVE_FRAME_SIZE_BYTES`,
`MAX_LIVE_FRAME_PIXELS`, `MAX_LIVE_CLIENTS`,
`ALERTS_DATABASE_PATH`, and `ALERT_SNAPSHOTS_DIR` in `backend\.env`. Alert
events are stored in SQLite and snapshot thumbnails are saved on disk. The
`GET /alerts` endpoint supports source, search, and pagination filters and
returns summary stats; `GET /alerts.csv` exports matching records. Live and
uploaded-video alert transitions are both persisted.

### Alerts history (Phase 7)

Open **Alerts** in the frontend to review persisted live and upload alerts.
Filter by source, search labels or source filenames, move through pages, and
export the current filtered results as CSV. The page displays total alerts,
the last-24-hour count, average confidence, and available snapshot thumbnails.
SQLite data and thumbnails are written to `outputs\alerts.db` and
`outputs\alert_snapshots` by default, and are retained across backend restarts.

Run the webcam test client from the project root while the API is running:

```powershell
.\backend\.venv\Scripts\python.exe backend\live_test_client.py --camera 0
```

Grant webcam access to the process. The client sends up to 320-pixel-wide
frames about every 250 ms and reports sent/result FPS and average inference
latency. Change the camera with `--camera`, the stream URL with `--url`, or
the send interval with `--interval`. Add `--duration 20` to run a bounded
20-second measurement and print overall result FPS and average inference
latency.

Run the backend tests with pytest from the project root:

```powershell
.\backend\.venv\Scripts\python.exe -m pytest backend -v
```

Run the frontend hook tests and production build:

```powershell
Set-Location frontend
npm test
npm run build
```

### About and settings (Phase 8)

The **About** page explains the frame analysis and temporal alert flow,
limitations, and data handling. Uploaded video is processed from a temporary
file and deleted when the job finishes; full live video is not archived, but
downscaled frames are transmitted to the backend. Alert metadata and one JPEG
snapshot per alert are retained by the backend for history.

The **Settings** page saves the live alert threshold, sensitivity, and sound
preference in browser local storage. These settings are sent to the backend for
live sessions only. High sensitivity confirms after about 0.5 seconds, low
sensitivity waits about 4 seconds, and balanced uses the backend's configured
confirmation period. Uploaded-video analysis uses server environment settings.
Use `--duration` with the webcam client to measure result FPS and inference
latency on the local machine; numbers depend on the camera, CPU/GPU, and model
runtime.
```

## Setup

Run these commands from the project root in PowerShell. Python 3.10 is used
below; if you have another supported Python version installed, change `-3.10`
to match it.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.check_env
```

If PowerShell blocks activation, use the venv interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.check_env
```

The environment check prints the Python, PyTorch, and Ultralytics versions and
whether CUDA/GPU is available. If no GPU is available, training can be run on
Google Colab or Kaggle. CUDA availability depends on the installed PyTorch build
and compatible NVIDIA drivers.

The project folders are `src/`, `data/`, `runs/`, and `models/`. The supplied
dataset currently remains in `Real Life Violence Dataset/`; paths and initial
video/model settings are in `config.yaml`, with relative paths resolved from
that file.

## Dataset note

The current dataset has `NonViolence/` and `Violence/` video folders with
video-level labels. Those labels do not provide bounding boxes, so they cannot
directly supervise YOLO object-detection training. The appropriate modeling
approach will be addressed in a later phase.

## Dataset inspection

Inspect both class folders and write per-video metadata and readability results
to `data/dataset_report.csv`:

```powershell
.\.venv\Scripts\python.exe -m src.inspect_dataset
```

The terminal summary reports class counts and balance, formats, unreadable
videos, resolution, FPS, and duration statistics.

## Prepare classification frames

Split videos at the video level into stratified train/validation/test sets, then
extract resized frames into the Ultralytics classification directory layout:

```powershell
.\.venv\Scripts\python.exe -m src.prepare_data
```

Frame interval, per-video frame cap, output size, random seed, and split ratios
are configured in `config.yaml`. Each video is assigned to one split before
frame extraction. Re-running replaces existing `.jpg` files in the managed
`data/yolo_cls/<split>/<class>/` output folders.

## Train a YOLO11 classification model

Start fine-tuning the pretrained nano classifier with settings from
`config.yaml`:

```powershell
.\.venv\Scripts\python.exe -m src.train
```

Select small or medium instead of nano with `--model-size s` or
`--model-size m`. To resume an interrupted run for the selected size, add
`--resume`, for example `.\.venv\Scripts\python.exe -m src.train --model-size n --resume`.
Training runs are saved under `runs/classify/yolo11{size}_cls/`; the best
checkpoint is copied to `models/best.pt`. Existing run folders are not
overwritten: use `--resume` to continue a run.

Epochs, image size, batch size, early-stopping patience, seed, workers, and
classification augmentation settings are configured in the `training` section
of `config.yaml`. When enabled, Ultralytics classification training uses
RandAugment and random erasing for training images.

## Evaluate on the test split

After training creates `models/best.pt`, evaluate on test images only:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate
```

The script reports accuracy, per-class precision/recall/F1, saves the confusion
matrix to `runs/errors/confusion_matrix.png`, and saves up to 20 annotated
misclassified test images under `runs/errors/`. Violence recall measures what
fraction of actual Violence examples the model catches; missed violence can
create a safety risk, so prioritize this metric over accuracy alone. Accuracy is
the overall fraction of correct predictions; it can hide poor Violence detection.
Violence precision is the fraction of Violence alerts that are correct, while
Violence F1 balances precision and recall. Low Violence recall means more missed
violence; low precision means more false alarms. If Violence recall is weak,
review false negatives, add diverse labeled videos, address class imbalance,
check split and label quality, and tune training. Since violence is temporal, also
consider a video action-recognition model rather than relying only on individual
frames. A higher recall may increase false alarms, so monitor precision as well.
Evaluation batch size, example limit, and the Violence-recall warning threshold
are set in the `evaluation` section of `config.yaml`.

## Run video or webcam inference

After training creates `models/best.pt`, run inference on a video:

```powershell
.\.venv\Scripts\python.exe -m src.infer_video --source "path\to\video.mp4"
```

Or use webcam index `0`:

```powershell
.\.venv\Scripts\python.exe -m src.infer_video --source 0
```

By default, the annotated video and append-only `alerts.csv` are saved under
`runs/inference/`. Use `--output`, `--alerts-csv`, `--threshold`,
`--alert-seconds`, and `--window-size` to override those settings at runtime;
`--no-display` disables the preview window. The inference section of
`config.yaml` sets the default 16-prediction smoothing window, Violence
probability threshold, required alert duration, inference image size, and
fallback FPS. The alert banner is latched until the smoothed probability falls
below its threshold; a new alert event is logged if it crosses again afterward.

## Person detection plus violence classification

The Option A prototype detects COCO people with pretrained `yolo11n.pt`,
classifies expanded square regions around each detected person with
`models/best.pt`, and temporally smooths the highest per-person Violence
probability. Ultralytics downloads `yolo11n.pt` on first use if it is not
available locally. Run it on a video or webcam:

```powershell
.\.venv\Scripts\python.exe -m src.infer_people --source "Real Life Violence Dataset\Violence\V_1.mp4"
.\.venv\Scripts\python.exe -m src.infer_people --source 0
```

Annotated videos and alert timestamps are saved under
`runs/inference/people/`. Use `--no-display` for headless inference; thresholds,
ROI context, and person detector settings are in `config.yaml`.

This is a prototype, not a validated safety detector: the current Violence
classifier was trained on full frames, not person crops, so ROI classification
has a domain shift and may reduce accuracy. People detection only localizes
people; it does not detect violence. Evaluate it on representative, video-level
held-out data and label crop/context examples before operational use.




cd C:\Users\sm713\OneDrive\Desktop\Mob-Alert
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload