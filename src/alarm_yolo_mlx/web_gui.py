from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from .config import PoseConfig
from .schedule import next_alarm_at, wait_until_alarm


class AlarmWebServer(ThreadingHTTPServer):
    def __init__(self, addr: tuple[str, int], handler) -> None:
        super().__init__(addr, handler)
        self.target: datetime | None = None
        self.status = "setup"
        self.detail = "Set an alarm time."
        self.ready = threading.Event()


class AlarmWebHandler(BaseHTTPRequestHandler):
    server: AlarmWebServer

    def do_GET(self) -> None:
        if self.path == "/status":
            self._send_json(
                {
                    "status": self.server.status,
                    "detail": self.server.detail,
                    "target": self.server.target.isoformat() if self.server.target else None,
                }
            )
            return
        if self.path not in {"/", "/index.html"}:
            self.send_error(404)
            return
        self._send_html(_index_page(datetime.now() + timedelta(minutes=1)))

    def do_POST(self) -> None:
        if self.path != "/start":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        form = parse_qs(body)

        try:
            self.server.target = _target_from_form(form)
        except argparse.ArgumentTypeError as exc:
            self._send_html(_index_page(datetime.now() + timedelta(minutes=1), str(exc)), status=400)
            return

        self.server.status = "armed"
        self.server.detail = f"Ringing at {self.server.target:%Y-%m-%d %H:%M:%S}."
        self.server.ready.set()
        self._send_html(_armed_page(self.server.target))

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send_html(self, html: str, status: int = 200) -> None:
        data = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _target_from_form(form: dict[str, list[str]]) -> datetime:
    delay = _first(form, "delay")
    if delay:
        return datetime.now() + timedelta(seconds=int(delay))

    alarm_time = _first(form, "alarm_time")
    if not alarm_time:
        raise argparse.ArgumentTypeError("Pick an alarm time or demo delay.")
    return next_alarm_at(alarm_time)


def _first(form: dict[str, list[str]], key: str) -> str:
    values = form.get(key, [])
    return values[0] if values else ""


def _index_page(default_time: datetime, error: str = "") -> str:
    error_html = f'<div class="error">{error}</div>' if error else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>6_7 Alarm</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #111315;
      color: #f5f1e8;
    }}
    body {{
      min-height: 100vh;
      margin: 0;
      display: grid;
      place-items: center;
      background:
        radial-gradient(circle at 18% 20%, rgba(255, 214, 102, 0.16), transparent 26%),
        linear-gradient(135deg, #111315 0%, #212427 100%);
    }}
    main {{
      width: min(92vw, 480px);
      padding: 32px;
      border: 1px solid #3a3f44;
      border-radius: 8px;
      background: #181b1f;
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.34);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 40px;
      letter-spacing: 0;
    }}
    p {{
      margin: 0 0 24px;
      color: #c6c0b5;
      line-height: 1.4;
    }}
    label {{
      display: block;
      margin-bottom: 8px;
      color: #f0c56b;
      font-weight: 700;
    }}
    input {{
      width: 100%;
      box-sizing: border-box;
      font: inherit;
      font-size: 42px;
      padding: 14px 16px;
      color: #f5f1e8;
      background: #0f1113;
      border: 1px solid #4b5158;
      border-radius: 8px;
    }}
    .actions {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 18px;
    }}
    button {{
      min-height: 48px;
      border: 0;
      border-radius: 8px;
      color: #141414;
      background: #f0c56b;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }}
    button.secondary {{
      color: #f5f1e8;
      background: #343a40;
    }}
    .error {{
      margin-bottom: 16px;
      padding: 12px;
      border-radius: 8px;
      background: #632626;
      color: #fff0f0;
    }}
  </style>
</head>
<body>
  <main>
    <h1>6_7 Alarm</h1>
    <p>Set the alarm time. When it rings, do the 6_7 motion to shut it up.</p>
    {error_html}
    <form method="post" action="/start">
      <label for="alarm_time">Alarm time</label>
      <input id="alarm_time" name="alarm_time" type="time" value="{default_time:%H:%M}" required>
      <div class="actions">
        <button type="submit">Start Alarm</button>
        <button class="secondary" type="submit" name="delay" value="10">10 sec demo</button>
        <button class="secondary" type="submit" name="delay" value="60">1 min demo</button>
        <button class="secondary" type="submit" name="delay" value="300">5 min demo</button>
      </div>
    </form>
  </main>
</body>
</html>"""


def _armed_page(target: datetime) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Alarm Armed</title>
  <style>
    body {{
      min-height: 100vh;
      margin: 0;
      display: grid;
      place-items: center;
      background: #111315;
      color: #f5f1e8;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(92vw, 520px);
      text-align: center;
    }}
    h1 {{
      margin: 0 0 14px;
      font-size: 42px;
      letter-spacing: 0;
    }}
    #countdown {{
      font-size: 56px;
      font-weight: 900;
      color: #f0c56b;
    }}
    .stopped #countdown {{
      color: #70d38b;
    }}
    .error #countdown {{
      color: #ff7b7b;
    }}
    p {{
      color: #c6c0b5;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Alarm Armed</h1>
    <div id="countdown">--</div>
    <p id="detail">Ringing at {target:%Y-%m-%d %H:%M:%S}. Keep this terminal running.</p>
  </main>
  <script>
    const target = new Date("{target.isoformat()}").getTime();
    const el = document.getElementById("countdown");
    const heading = document.querySelector("h1");
    const detail = document.getElementById("detail");
    let backendStatus = "armed";

    async function pollStatus() {{
      try {{
        const res = await fetch("/status", {{ cache: "no-store" }});
        const data = await res.json();
        backendStatus = data.status;
        detail.textContent = data.detail || detail.textContent;
        document.body.classList.toggle("stopped", backendStatus === "stopped");
        document.body.classList.toggle("error", backendStatus === "error");
        if (backendStatus === "preparing") {{
          heading.textContent = "Alarm Armed";
          el.textContent = "Preparing model";
        }} else if (backendStatus === "ringing") {{
          heading.textContent = "Alarm Ringing";
          el.textContent = "Do 6_7";
        }} else if (backendStatus === "stopped" || backendStatus === "exited") {{
          heading.textContent = "Alarm Stopped";
          el.textContent = "Stopped";
        }} else if (backendStatus === "error") {{
          heading.textContent = "Alarm Error";
          el.textContent = "Check terminal";
        }}
      }} catch (err) {{
        detail.textContent = "Waiting for terminal status...";
      }}
    }}

    function tick() {{
      if (["preparing", "ringing", "stopped", "exited", "error"].includes(backendStatus)) return;
      const remaining = Math.max(0, Math.floor((target - Date.now()) / 1000));
      const hours = Math.floor(remaining / 3600);
      const mins = Math.floor((remaining % 3600) / 60);
      const secs = remaining % 60;
      el.textContent = hours > 0
        ? `${{hours}}h ${{mins}}m ${{secs}}s`
        : mins > 0
          ? `${{mins}}m ${{secs}}s`
          : `${{secs}}s`;
      if (remaining === 0) el.textContent = "Ringing now";
    }}
    tick();
    pollStatus();
    setInterval(tick, 250);
    setInterval(pollStatus, 500);
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pose_alarm.yaml")
    parser.add_argument("--weights")
    parser.add_argument("--backend", choices=["mlx", "ultralytics"])
    parser.add_argument("--source")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()

    cfg = PoseConfig.from_yaml(args.config)
    if args.weights:
        cfg.weights = args.weights
    if args.backend:
        cfg.backend = args.backend
    if args.source:
        cfg.source = args.source
    if args.camera_index is not None:
        cfg.camera_index = args.camera_index
    if args.conf is not None:
        cfg.conf = args.conf
    if args.no_show:
        cfg.show = False

    server = AlarmWebServer((args.host, args.port), AlarmWebHandler)
    url = f"http://{args.host}:{server.server_port}"
    print(f"alarm setup ui: {url}")
    webbrowser.open(url)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        server.ready.wait()
    except KeyboardInterrupt:
        server.shutdown()
        print("alarm exited")
        return

    target = server.target
    detector = None
    try:
        server.status = "preparing"
        server.detail = "Preparing pose model before the alarm rings."
        from .pose import make_pose_detector

        detector = make_pose_detector(cfg.backend, cfg.weights, cfg.conf, cfg.imgsz)
        server.status = "armed"
        server.detail = f"Ready. Ringing at {target:%Y-%m-%d %H:%M:%S}."
    except Exception as exc:
        server.status = "error"
        server.detail = f"Pose model failed before alarm time: {exc}"
        print(server.detail)
        time.sleep(5)
        server.shutdown()
        thread.join()
        return

    wait_until_alarm(target)
    from .alarm import Alarm
    from .pose_app import PoseAlarmApp

    alarm = Alarm(cfg.alarm_sound)
    alarm.start()
    server.status = "ringing"
    server.detail = "Alarm is ringing. Do the 6_7 gesture to stop it."
    try:
        stopped = PoseAlarmApp(cfg, alarm=alarm, detector=detector).run()
    except Exception as exc:
        alarm.stop()
        server.status = "error"
        server.detail = f"Alarm failed: {exc}"
        if "video source" in str(exc):
            print(
                "camera failed to open. On macOS, allow camera access for your "
                "terminal/Python app in System Settings > Privacy & Security > Camera."
            )
        elif "load_npz" in str(exc) or "zip file" in str(exc):
            print(
                f"pose weights failed to load: {cfg.weights}. The MLX backend expects a valid "
                ".npz converted pose model. For the demo, run with --backend ultralytics "
                "--weights yolo26n-pose.pt, or regenerate the .npz file."
            )
        raise
    server.status = "stopped" if stopped else "exited"
    server.detail = "Alarm stopped by 6_7 gesture." if stopped else "Alarm exited."
    print("alarm stopped" if stopped else "alarm exited")
    time.sleep(3)
    server.shutdown()
    thread.join()


if __name__ == "__main__":
    main()
