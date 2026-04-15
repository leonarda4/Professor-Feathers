from __future__ import annotations

import json
import queue
import sys
import threading
import time
import traceback
import webbrowser
from collections import deque
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import live_main
from audio import MicrophoneStream, describe_default_input
from collect import build_endpointer, save_batch_utterance, start_recording_batch
from config import load_app_config
from parrot_feedback import AudioFeedbackPlayer
from storage import ensure_storage


UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Professor Feathers</title>
  <style>
    :root {
      --bg: #f7f3ea;
      --card: #fffaf2;
      --border: #dbcdb8;
      --text: #231b14;
      --muted: #6d5b4d;
      --accent: #9c5f2d;
      --sing: #2d7a78;
      --dance: #c96b2c;
      --start: #3d7f3b;
      --stop: #8c3d2f;
      --shadow: 0 20px 45px rgba(92, 61, 25, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(201, 107, 44, 0.10), transparent 32%),
        radial-gradient(circle at top right, rgba(45, 122, 120, 0.10), transparent 28%),
        linear-gradient(180deg, #fbf8f2 0%, var(--bg) 100%);
    }
    .shell {
      width: min(1120px, calc(100% - 32px));
      margin: 24px auto 40px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: var(--shadow);
      padding: 22px;
    }
    .hero h1 {
      margin: 0 0 8px;
      font-size: clamp(2rem, 5vw, 3.2rem);
      line-height: 1.05;
      letter-spacing: -0.03em;
    }
    .hero .sub {
      margin: 0;
      color: var(--accent);
      font-size: 1.1rem;
      font-weight: 700;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 18px;
    }
    .stack {
      display: grid;
      gap: 18px;
      margin-top: 18px;
    }
    h2 {
      margin: 0 0 8px;
      font-size: 1.35rem;
      letter-spacing: -0.02em;
    }
    p.help {
      margin: 0 0 14px;
      color: var(--muted);
      line-height: 1.5;
    }
    button {
      border: 0;
      border-radius: 14px;
      padding: 13px 18px;
      font: inherit;
      font-weight: 700;
      color: white;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }
    button:hover:enabled { transform: translateY(-1px); }
    button:disabled {
      opacity: 0.45;
      cursor: default;
      transform: none;
    }
    .btn-sing { background: var(--sing); width: 100%; }
    .btn-dance { background: var(--dance); width: 100%; }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .list, .log {
      border: 1px solid var(--border);
      border-radius: 16px;
      background: #fffdf9;
      overflow: hidden;
    }
    .list {
      min-height: 300px;
      max-height: 420px;
      overflow-y: auto;
    }
    .list-item {
      padding: 14px 16px;
      border-top: 1px solid rgba(219, 205, 184, 0.55);
    }
    .list-item:first-child { border-top: 0; }
    .list-item strong {
      display: block;
      margin-bottom: 4px;
      font-size: 1rem;
    }
    .list-item span {
      color: var(--muted);
      font-size: 0.95rem;
    }
    .log {
      padding: 0;
      min-height: 260px;
      max-height: 360px;
      overflow-y: auto;
    }
    .log-line {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.93rem;
      white-space: pre-wrap;
      padding: 10px 14px;
      border-top: 1px solid rgba(219, 205, 184, 0.45);
    }
    .log-line:first-child { border-top: 0; }
    @media (max-width: 860px) {
      .grid { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="card hero">
      <h1>Professor Feathers</h1>
      <p class="sub">Secret keywords are named automatically.</p>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Recorded Secret Keywords</h2>
        <p class="help" id="keywordCount">Loading...</p>
        <div class="list" id="keywordList"></div>
      </div>

      <div class="card">
        <h2>Record A New Secret Keyword</h2>
        <p class="help" id="recordingHelp">Choose whether the new keyword should trigger sing or dance. The name is assigned automatically.</p>
        <div class="row" style="margin-top: 14px;">
          <button class="btn-sing" id="recordSingBtn">Record for Sing</button>
          <button class="btn-dance" id="recordDanceBtn">Record for Dance</button>
        </div>
      </div>
    </section>

    <section class="stack">
      <div class="card">
        <h2>Activity</h2>
        <div class="log" id="log"></div>
      </div>
    </section>
  </div>

  <script>
    const els = {
      keywordCount: document.getElementById("keywordCount"),
      keywordList: document.getElementById("keywordList"),
      log: document.getElementById("log"),
      recordSingBtn: document.getElementById("recordSingBtn"),
      recordDanceBtn: document.getElementById("recordDanceBtn"),
      recordingHelp: document.getElementById("recordingHelp"),
    };

    let lastLogSignature = "";

    async function api(path, method = "GET", body = null) {
      const options = { method, headers: {} };
      if (body !== null) {
        options.headers["Content-Type"] = "application/json";
        options.body = JSON.stringify(body);
      }
      const response = await fetch(path, options);
      const payload = await response.json();
      if (!response.ok || payload.ok === false) {
        throw new Error(payload.error || `Request failed: ${response.status}`);
      }
      return payload;
    }

    function renderKeywords(keywords) {
      if (!keywords.length) {
        els.keywordList.innerHTML = '<div class="list-item"><strong>No secret keywords recorded yet.</strong><span>Use the recording buttons to create one.</span></div>';
        return;
      }
      els.keywordList.innerHTML = keywords.map(item => `
        <div class="list-item">
          <strong>${escapeHtml(item.stored_label)} -> ${escapeHtml(item.action_label)}</strong>
          <span>${item.sample_count} recordings</span>
        </div>
      `).join("");
    }

    function renderLogs(logs) {
      const signature = logs.join("\\n");
      if (signature === lastLogSignature) return;
      lastLogSignature = signature;
      els.log.innerHTML = logs.length
        ? logs.map(line => `<div class="log-line">${escapeHtml(line)}</div>`).join("")
        : '<div class="log-line">No activity yet.</div>';
      els.log.scrollTop = els.log.scrollHeight;
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function renderState(state) {
      els.keywordCount.textContent = `Custom keywords: ${state.keywords.length} recorded`;
      els.recordingHelp.textContent = `Each button records ${state.batch_size} examples automatically and assigns the keyword name for you.`;
      els.recordSingBtn.disabled = !state.can_record_keyword;
      els.recordDanceBtn.disabled = !state.can_record_keyword;
      renderKeywords(state.keywords);
      renderLogs(state.logs);
    }

    async function refreshState() {
      try {
        const payload = await api("/api/state");
        renderState(payload.state);
      } catch (error) {
        console.error(error);
      }
    }

    async function postAction(path, body = null) {
      try {
        const payload = await api(path, "POST", body);
        renderState(payload.state);
      } catch (error) {
        console.error(error);
      }
    }

    els.recordSingBtn.addEventListener("click", () => postAction("/api/record", {
      action: "sing",
    }));
    els.recordDanceBtn.addEventListener("click", () => postAction("/api/record", {
      action: "dance",
    }));
    els.recordDanceBtn.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        postAction("/api/record", {
          action: "dance",
        });
      }
    });
    els.recordSingBtn.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        postAction("/api/record", {
          action: "sing",
        });
      }
    });

    refreshState();
    setInterval(refreshState, 900);
  </script>
</body>
</html>
"""


@dataclass(frozen=True)
class DynamicKeywordSummary:
    stored_label: str
    action_label: str
    sample_count: int


def infer_action_label(stored_label: str) -> str:
    cleaned = stored_label.strip().lower()
    if cleaned.startswith(f"{live_main.SING_PREFIX}_"):
        return live_main.SING_ACTION_LABEL
    if cleaned.startswith(f"{live_main.DANCE_PREFIX}_"):
        return live_main.DANCE_ACTION_LABEL
    return cleaned


def list_dynamic_keywords(source_root: Path) -> list[DynamicKeywordSummary]:
    root = Path(source_root).resolve()
    if not root.exists():
        return []

    summaries: list[DynamicKeywordSummary] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir():
            continue
        sample_count = sum(1 for path in child.rglob("*.wav") if path.is_file())
        summaries.append(
            DynamicKeywordSummary(
                stored_label=child.name,
                action_label=infer_action_label(child.name),
                sample_count=sample_count,
            )
        )
    return summaries


def build_automatic_keyword_label(source_root: Path, action_label: str) -> str:
    cleaned = str(action_label).strip().lower()
    if cleaned == live_main.SING_ACTION_LABEL:
        return live_main._next_anonymous_keyword(source_root, live_main.SING_PREFIX)
    if cleaned == live_main.DANCE_ACTION_LABEL:
        return live_main._next_anonymous_keyword(source_root, live_main.DANCE_PREFIX)
    raise ValueError("Recording action must be either sing or dance.")


def build_keyword_payload(summary: DynamicKeywordSummary) -> dict[str, Any]:
    return {
        "stored_label": summary.stored_label,
        "action_label": summary.action_label,
        "sample_count": summary.sample_count,
    }


class LiveListenerWorker(threading.Thread):
    def __init__(self, *, model, config, ui_queue: "queue.Queue[tuple[str, object]]") -> None:
        super().__init__(daemon=True)
        self.model = model
        self.config = config
        self.ui_queue = ui_queue
        self.stop_event = threading.Event()
        self._microphone: MicrophoneStream | None = None

    def stop(self) -> None:
        self.stop_event.set()
        if self._microphone is not None:
            self._microphone.stop()

    def _post(self, event_type: str, *payload: object) -> None:
        self.ui_queue.put((event_type, *payload))

    def _pause_after_feedback(self, endpointer, *, playback_started: bool, pause_seconds: float) -> None:
        if not playback_started:
            endpointer.arm()
            return

        deadline = time.monotonic() + max(0.0, float(pause_seconds))
        while time.monotonic() < deadline and not self.stop_event.is_set():
            if self._microphone is None:
                break
            _ = self._microphone.read_block(timeout=0.05)
        endpointer.arm()

    def run(self) -> None:
        try:
            feedback = AudioFeedbackPlayer(
                live_main.PARROT_FEEDBACK_ROOT,
                rng_seed=42,
                dance_trigger_probability=live_main.DANCE_FEEDBACK_PROBABILITY,
            )
            sample_rate = int(self.config.audio.sample_rate)
            endpointer = build_endpointer(self.config)
            default_device = describe_default_input()

            if default_device:
                self._post(
                    "log",
                    (
                        f"Using microphone {default_device.index}: {default_device.name} "
                        f"({default_device.default_samplerate:.0f} Hz default)"
                    ),
                )

            self._post(
                "log",
                "Listening for sing, dance, or one of the recorded secret keywords.",
            )
            self._post("status", "Listening")

            with MicrophoneStream(
                sample_rate=sample_rate,
                channels=self.config.audio.channels,
                block_ms=self.config.audio.block_ms,
                device=self.config.audio.device,
            ) as microphone:
                self._microphone = microphone
                endpointer.arm()

                while not self.stop_event.is_set():
                    for status_message in microphone.pop_status_messages():
                        self._post("log", f"audio-status: {status_message}")

                    chunk = microphone.read_block(timeout=0.1)
                    if chunk is None or self.stop_event.is_set():
                        continue

                    for event in endpointer.process_chunk(chunk):
                        if self.stop_event.is_set():
                            break
                        if event.kind != "utterance" or event.audio is None or event.audio.size == 0:
                            continue

                        base_prediction = live_main.predict_base_command(
                            self.model,
                            event.audio,
                            sample_rate,
                        )
                        if base_prediction.is_known and base_prediction.predicted_label is not None:
                            action_label = base_prediction.predicted_label
                            self._post("recognized", f"Heard {action_label}")
                            self._post(
                                "log",
                                (
                                    f"Base command recognized: {action_label} "
                                    f"(distance {base_prediction.mean_neighbor_distance:.3f})"
                                ),
                            )
                            live_main._perform_action(action_label)
                            playback_started = feedback.maybe_play_recognized(action_label)
                            self._pause_after_feedback(
                                endpointer,
                                playback_started=playback_started,
                                pause_seconds=live_main.FEEDBACK_RECOGNIZED_PAUSE_SECONDS,
                            )
                            continue

                        dynamic_prediction = live_main.predict_dynamic_command(
                            self.model,
                            event.audio,
                            sample_rate,
                        )
                        if dynamic_prediction.is_known and dynamic_prediction.action_label is not None:
                            action_label = dynamic_prediction.action_label
                            self._post("recognized", f"Heard custom keyword -> {action_label}")
                            self._post(
                                "log",
                                (
                                    f"Secret keyword recognized -> {action_label} "
                                    f"(stored as {dynamic_prediction.closest_label})"
                                ),
                            )
                            live_main._perform_action(action_label)
                            playback_started = feedback.maybe_play_recognized(action_label)
                            self._pause_after_feedback(
                                endpointer,
                                playback_started=playback_started,
                                pause_seconds=live_main.FEEDBACK_RECOGNIZED_PAUSE_SECONDS,
                            )
                            continue

                        self._post("recognized", "Not recognized")
                        self._post(
                            "log",
                            "Heard something, but it did not match sing, dance, or a secret keyword.",
                        )
                        playback_started = feedback.play_not_recognized()
                        self._pause_after_feedback(
                            endpointer,
                            playback_started=playback_started,
                            pause_seconds=live_main.FEEDBACK_NOT_RECOGNIZED_PAUSE_SECONDS,
                        )
        except Exception as exc:
            self._post("worker_error", f"Live listening failed: {exc}\n{traceback.format_exc()}")
        finally:
            self._microphone = None
            self._post("live_stopped")


class KeywordRecorderWorker(threading.Thread):
    def __init__(
        self,
        *,
        model,
        config,
        project_root: Path,
        dynamic_root: Path,
        keyword_label: str,
        action_label: str,
        ui_queue: "queue.Queue[tuple[str, object]]",
    ) -> None:
        super().__init__(daemon=True)
        self.model = model
        self.config = config
        self.project_root = Path(project_root).resolve()
        self.dynamic_root = Path(dynamic_root).resolve()
        self.keyword_label = keyword_label
        self.action_label = action_label
        self.ui_queue = ui_queue
        self.stop_event = threading.Event()
        self._microphone: MicrophoneStream | None = None

    def stop(self) -> None:
        self.stop_event.set()
        if self._microphone is not None:
            self._microphone.stop()

    def _post(self, event_type: str, *payload: object) -> None:
        self.ui_queue.put((event_type, *payload))

    def _pause_after_feedback(self, endpointer, *, playback_started: bool, pause_seconds: float) -> None:
        if not playback_started:
            endpointer.arm()
            return

        deadline = time.monotonic() + max(0.0, float(pause_seconds))
        while time.monotonic() < deadline and not self.stop_event.is_set():
            if self._microphone is None:
                break
            _ = self._microphone.read_block(timeout=0.05)
        endpointer.arm()

    def run(self) -> None:
        try:
            feedback = AudioFeedbackPlayer(
                live_main.PARROT_FEEDBACK_ROOT,
                rng_seed=42,
                dance_trigger_probability=live_main.DANCE_FEEDBACK_PROBABILITY,
            )
            sample_rate = int(self.config.audio.sample_rate)
            batch_size = max(1, int(self.config.collection.batch_size))
            endpointer = build_endpointer(self.config)
            _, manifest_path = ensure_storage(
                project_root=self.project_root,
                data_dir=self.config.storage.data_dir,
                manifest_path=self.config.storage.manifest_path,
            )
            batch = start_recording_batch(self.keyword_label, batch_size=batch_size)

            self._post(
                "log",
                (
                    f"Recording a new secret keyword for {self.action_label}. "
                    f"Say the same word clearly {batch.total} times."
                ),
            )
            self._post("status", f"Recording for {self.action_label} ({batch.total} examples)")

            with MicrophoneStream(
                sample_rate=sample_rate,
                channels=self.config.audio.channels,
                block_ms=self.config.audio.block_ms,
                device=self.config.audio.device,
            ) as microphone:
                self._microphone = microphone
                endpointer.arm()

                while batch.remaining > 0 and not self.stop_event.is_set():
                    for status_message in microphone.pop_status_messages():
                        self._post("log", f"audio-status: {status_message}")

                    chunk = microphone.read_block(timeout=0.1)
                    if chunk is None or self.stop_event.is_set():
                        continue

                    for event in endpointer.process_chunk(chunk):
                        if self.stop_event.is_set():
                            break
                        if event.kind == "discarded":
                            self._post(
                                "log",
                                "That clip was too short. Please say the keyword again.",
                            )
                            continue
                        if event.kind != "utterance" or event.audio is None or event.audio.size == 0:
                            continue

                        record, completed, is_complete = save_batch_utterance(
                            event.audio,
                            project_root=self.project_root,
                            raw_dir=self.dynamic_root,
                            manifest_path=manifest_path,
                            batch=batch,
                            sample_rate=sample_rate,
                        )
                        self._post(
                            "log",
                            f"Saved {completed}/{batch.total} for {self.action_label} at {record.path}",
                        )
                        self._post("status", f"Recording for {self.action_label} ({completed}/{batch.total})")

                        if not is_complete:
                            playback_started = feedback.maybe_play_training()
                            self._pause_after_feedback(
                                endpointer,
                                playback_started=playback_started,
                                pause_seconds=live_main.FEEDBACK_TRAINING_PAUSE_SECONDS,
                            )
                            continue

                        self._post("log", "Rebuilding the secret keyword model...")
                        dynamic_classifier, dynamic_threshold = live_main.prepare_dynamic_classifier(
                            self.project_root,
                            self.dynamic_root,
                        )
                        self.model.dynamic_classifier = dynamic_classifier
                        self.model.dynamic_unknown_distance_threshold = dynamic_threshold
                        self._post("keywords_changed")
                        self._post("recording_finished", self.keyword_label, self.action_label)
                        return
        except Exception as exc:
            self._post("worker_error", f"Recording failed: {exc}\n{traceback.format_exc()}")
        finally:
            self._microphone = None
            self._post("record_worker_stopped")


class AppController:
    def __init__(self) -> None:
        self.project_root = PROJECT_ROOT
        self.dynamic_root = Path(live_main.DYNAMIC_KEYWORD_SOURCE_ROOT).resolve()
        self.ui_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.lock = threading.RLock()
        self.shutdown_event = threading.Event()

        self.config = load_app_config(config_path=live_main.CONFIG_PATH)
        if live_main.QUIT_KEY:
            self.config.collection.quit_key = live_main.QUIT_KEY

        self.model = None
        self.live_worker: LiveListenerWorker | None = None
        self.record_worker: KeywordRecorderWorker | None = None
        self.pending_recording: tuple[str, str] | None = None
        self.logs: deque[str] = deque(maxlen=220)
        self.status = "Loading models..."
        self.heard = "Ready soon"
        self.error: str | None = None
        self.keywords: list[DynamicKeywordSummary] = list_dynamic_keywords(self.dynamic_root)
        self.device = self._build_device_text()

        self.event_thread = threading.Thread(target=self._event_loop, daemon=True)
        self.event_thread.start()
        threading.Thread(target=self._load_model_worker, daemon=True).start()

    def _build_device_text(self) -> str:
        try:
            default_device = describe_default_input()
        except Exception as exc:
            return f"Microphone unavailable: {exc}"
        if default_device is None:
            return "Microphone: no default input device found"
        return (
            f"Microphone: {default_device.name} "
            f"({default_device.default_samplerate:.0f} Hz default)"
        )

    def _append_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")

    def _set_error(self, message: str | None) -> None:
        self.error = message

    def _refresh_keywords_locked(self) -> None:
        self.keywords = list_dynamic_keywords(self.dynamic_root)

    def _load_model_worker(self) -> None:
        try:
            model = live_main.prepare_dual_live_model(
                project_root=self.project_root,
                base_source_root=Path(live_main.BASE_KEYWORD_SOURCE_ROOT).resolve(),
                dynamic_source_root=self.dynamic_root,
                base_model_type=live_main.BASE_MODEL_TYPE,
            )
        except Exception as exc:
            self.ui_queue.put(("model_error", f"Could not load the live models: {exc}\n{traceback.format_exc()}"))
            return
        self.ui_queue.put(("model_loaded", model))

    def _event_loop(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                event = self.ui_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._handle_event(event)

    def _handle_event(self, event: tuple[str, object]) -> None:
        event_type = event[0]
        payload = event[1:]

        with self.lock:
            if event_type == "log":
                self._append_log(str(payload[0]))
            elif event_type == "status":
                self.status = str(payload[0])
            elif event_type == "recognized":
                self.heard = str(payload[0])
            elif event_type == "model_loaded":
                self.model = payload[0]
                self.status = "Starting live listening..."
                self.heard = "Waiting for a word..."
                self._set_error(None)
                self._append_log("Live models loaded. Starting live listening automatically.")
                self.start_listening()
            elif event_type == "model_error":
                details = str(payload[0])
                self.status = "Model load failed"
                self._set_error(details.splitlines()[0])
                self._append_log(details)
            elif event_type == "worker_error":
                details = str(payload[0])
                self.status = "Something went wrong"
                self._set_error(details.splitlines()[0])
                self._append_log(details)
            elif event_type == "live_stopped":
                self.live_worker = None
                self._append_log("Live listening stopped.")
                if self.pending_recording is not None:
                    keyword_label, action_label = self.pending_recording
                    self.pending_recording = None
                    self._begin_recording_locked(keyword_label, action_label)
                elif self.record_worker is None or not self.record_worker.is_alive():
                    self.status = "Ready"
            elif event_type == "recording_finished":
                stored_label, action_label = payload
                self.heard = f"Saved {stored_label}"
                self.status = "Ready"
                self._append_log(
                    f"Finished recording {stored_label}. It now triggers {action_label}."
                )
                self._refresh_keywords_locked()
                self._set_error(None)
            elif event_type == "record_worker_stopped":
                self.record_worker = None
                if (
                    not self.shutdown_event.is_set()
                    and self.error is None
                    and (self.live_worker is None or not self.live_worker.is_alive())
                ):
                    self.start_listening()
                elif self.live_worker is None or not self.live_worker.is_alive():
                    self.status = "Ready"
            elif event_type == "keywords_changed":
                self._refresh_keywords_locked()

    def _begin_recording_locked(self, keyword_label: str, action_label: str) -> None:
        if self.model is None:
            raise ValueError("Models are still loading.")
        if self.record_worker is not None and self.record_worker.is_alive():
            raise ValueError("A recording batch is already in progress.")

        self.record_worker = KeywordRecorderWorker(
            model=self.model,
            config=self.config,
            project_root=self.project_root,
            dynamic_root=self.dynamic_root,
            keyword_label=keyword_label,
            action_label=action_label,
            ui_queue=self.ui_queue,
        )
        self.heard = f"Recording for {action_label}"
        self.status = f"Preparing to record a new keyword for {action_label}..."
        self._append_log(f"Preparing to record a new keyword for {action_label}.")
        self.record_worker.start()

    def start_listening(self) -> None:
        with self.lock:
            if self.model is None:
                raise ValueError("Models are still loading.")
            if self.record_worker is not None and self.record_worker.is_alive():
                raise ValueError("Finish the current recording batch first.")
            if self.live_worker is not None and self.live_worker.is_alive():
                return

            self._set_error(None)
            self.live_worker = LiveListenerWorker(
                model=self.model,
                config=self.config,
                ui_queue=self.ui_queue,
            )
            self.heard = "Waiting for a word..."
            self.status = "Starting live listening..."
            self._append_log("Starting live listening.")
            self.live_worker.start()

    def stop_listening(self) -> None:
        with self.lock:
            if self.live_worker is None:
                return
            self.status = "Stopping live listening..."
            self._append_log("Stopping live listening.")
            self.live_worker.stop()

    def refresh_keywords(self) -> None:
        with self.lock:
            self._refresh_keywords_locked()
            self._set_error(None)

    def request_recording(self, action_label: str) -> None:
        with self.lock:
            if self.model is None:
                raise ValueError("Models are still loading.")

            keyword_label = build_automatic_keyword_label(self.dynamic_root, action_label)

            self._set_error(None)

            if self.live_worker is not None and self.live_worker.is_alive():
                self.pending_recording = (keyword_label, action_label)
                self.status = "Stopping listening before recording..."
                self._append_log(
                    f"Stopping live listening before recording a new keyword for {action_label}."
                )
                self.live_worker.stop()
                return

            self._begin_recording_locked(keyword_label, action_label)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            listening = self.live_worker is not None and self.live_worker.is_alive()
            recording = self.record_worker is not None and self.record_worker.is_alive()
            return {
                "model_ready": self.model is not None,
                "listening": listening,
                "recording": recording,
                "status": self.status,
                "heard": self.heard,
                "device": self.device,
                "error": self.error,
                "batch_size": max(1, int(self.config.collection.batch_size)),
                "keywords": [build_keyword_payload(item) for item in self.keywords],
                "logs": list(self.logs),
                "can_start_listening": self.model is not None and not listening and not recording,
                "can_stop_listening": listening,
                "can_record_keyword": self.model is not None and not recording,
            }

    def close(self) -> None:
        self.shutdown_event.set()
        with self.lock:
            if self.live_worker is not None:
                self.live_worker.stop()
            if self.record_worker is not None:
                self.record_worker.stop()


def build_handler(controller: AppController):
    class LiveUIHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, Any], *, status: int = HTTPStatus.OK) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_html(self, html: str) -> None:
            raw = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            body = self.rfile.read(length)
            if not body:
                return {}
            return json.loads(body.decode("utf-8"))

        def _handle_action(self, action) -> None:
            try:
                action()
                self._send_json({"ok": True, "state": controller.snapshot()})
            except ValueError as exc:
                self._send_json(
                    {"ok": False, "error": str(exc), "state": controller.snapshot()},
                    status=HTTPStatus.BAD_REQUEST,
                )
            except Exception as exc:
                controller._append_log(f"Unexpected server error: {exc}\n{traceback.format_exc()}")
                self._send_json(
                    {"ok": False, "error": str(exc), "state": controller.snapshot()},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._send_html(UI_HTML)
                return
            if path == "/api/state":
                self._send_json({"ok": True, "state": controller.snapshot()})
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/start-listening":
                self._handle_action(controller.start_listening)
                return
            if path == "/api/stop-listening":
                self._handle_action(controller.stop_listening)
                return
            if path == "/api/refresh-keywords":
                self._handle_action(controller.refresh_keywords)
                return
            if path == "/api/record":
                payload = self._read_json()
                action_label = str(payload.get("action", "")).strip().lower()
                self._handle_action(lambda: controller.request_recording(action_label))
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return LiveUIHandler


def run_server(*, controller: AppController, host: str = "127.0.0.1", preferred_port: int = 8765) -> int:
    handler = build_handler(controller)
    try:
        server = ThreadingHTTPServer((host, preferred_port), handler)
    except OSError:
        server = ThreadingHTTPServer((host, 0), handler)

    actual_host, actual_port = server.server_address[:2]
    url = f"http://{actual_host}:{actual_port}/"

    print(f"Professor Feathers UI is running at {url}")
    print("Open that URL in your browser if it does not open automatically.")
    print("Press Ctrl+C in this terminal to stop the UI.")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping Professor Feathers UI.")
    finally:
        controller.close()
        server.shutdown()
        server.server_close()
    return 0


def main() -> int:
    controller = AppController()
    return run_server(controller=controller)


if __name__ == "__main__":
    raise SystemExit(main())
