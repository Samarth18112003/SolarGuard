from __future__ import annotations

from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import threading
import time

try:
    import serial  # pyserial
except Exception:
    serial = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY", "change-this-secret-key")

APP_DIR = Path(__file__).parent
USERS_FILE = APP_DIR / "users.json"
DATA_DIR = APP_DIR / "data"
HISTORY_FILE = DATA_DIR / "esp32_history.jsonl"

SERIAL_PORT = os.getenv("ESP32_SERIAL_PORT", "COM3")
BAUD_RATE = int(os.getenv("ESP32_BAUD_RATE", "115200"))
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", str(APP_DIR / "best.pt"))
DETECTION_COOLDOWN = float(os.getenv("DETECTION_COOLDOWN_SEC", "2.0"))
DETECTION_FRAME_SKIP = int(os.getenv("DETECTION_FRAME_SKIP", "2"))
DETECTION_IMGSZ = int(os.getenv("DETECTION_IMGSZ", "480"))
DETECTION_IOU = float(os.getenv("DETECTION_IOU", "0.6"))

METRIC_KEYS = [
    "Temp",
    "Volt",
    "Current",
    "Dust",
    "Airquality",
    "Airqualitystatus",
    "blowercount",
    "liquidcount",
    "smscount",
]

ALIASES = {
    "airqualitystatus": "Airqualitystatus",
    "blowercunt": "blowercount",
}

state_lock = threading.Lock()
serial_lock = threading.Lock()
serial_thread_started = False
serial_conn = None
latest_metrics: dict[str, object] = {
    "timestamp": "",
    **{k: 0 for k in METRIC_KEYS},
}

detection_lock = threading.Lock()
detection_model = None
detection_state: dict[str, object] = {
    "running": False,
    "cracked": 0,
    "soiled": 0,
    "fps": 0,
    "last_command": "",
    "message": "idle",
    "last_ts": "",
}
last_frame_jpeg: bytes | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.touch()


def seed_initial_history_if_empty() -> None:
    ensure_storage()
    with HISTORY_FILE.open("r", encoding="utf-8") as f:
        has_line = any(line.strip() for line in f)
    if has_line:
        return

    record = {"timestamp": now_iso(), **{k: 0 for k in METRIC_KEYS}}
    with state_lock:
        latest_metrics.update(record)
    append_history(record)


def load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    try:
        with USERS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users: dict) -> None:
    with USERS_FILE.open("w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated_function


def coerce_value(value: object) -> object:
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if text == "":
        return 0
    low = text.lower()
    if low in {"on", "true", "ok", "good"}:
        return 1
    if low in {"off", "false", "bad"}:
        return 0
    try:
        if "." in text:
            return float(text)
        return int(text)
    except Exception:
        return text


def normalize_key(key: str) -> str:
    clean = key.strip()
    alias_hit = ALIASES.get(clean.lower())
    if alias_hit:
        return alias_hit

    for allowed in METRIC_KEYS:
        if clean.lower() == allowed.lower():
            return allowed
    return clean


def parse_serial_payload(raw: str) -> dict[str, object]:
    line = raw.strip()
    if not line:
        return {}

    # Accept framed payloads like:
    # <Temp=30,Volt=12.4,...> or <30,12.4,1.1,20,50,A,1,0,0>
    if "<" in line and ">" in line:
        start = line.find("<")
        end = line.rfind(">")
        if start != -1 and end != -1 and end > start:
            line = line[start + 1:end].strip()
            if not line:
                return {}

    try:
        loaded = json.loads(line)
        if isinstance(loaded, dict):
            return {normalize_key(k): coerce_value(v) for k, v in loaded.items()}
    except Exception:
        pass

    payload: dict[str, object] = {}
    parts = [p.strip() for p in line.split(",") if p.strip()]
    unnamed_values: list[object] = []

    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            payload[normalize_key(key)] = coerce_value(value)
        elif ":" in part:
            key, value = part.split(":", 1)
            payload[normalize_key(key)] = coerce_value(value)
        else:
            unnamed_values.append(coerce_value(part))

    if unnamed_values:
        ordered = METRIC_KEYS[: len(unnamed_values)]
        for idx, key in enumerate(ordered):
            payload[key] = unnamed_values[idx]

    return payload


def append_history(record: dict[str, object]) -> None:
    ensure_storage()
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def normalize_history_record(record: dict[str, object]) -> dict[str, object]:
    out = dict(record)
    if "Airqualitystatus" not in out:
        if "Airqualitystatus_A" in out:
            out["Airqualitystatus"] = out.get("Airqualitystatus_A")
        elif "Airqualitystatus_B" in out:
            out["Airqualitystatus"] = out.get("Airqualitystatus_B")
    out.pop("Airqualitystatus_A", None)
    out.pop("Airqualitystatus_B", None)
    return out


def push_metrics(payload: dict[str, object]) -> dict[str, object]:
    record = {"timestamp": now_iso(), **{k: 0 for k in METRIC_KEYS}}
    with state_lock:
        for key in METRIC_KEYS:
            record[key] = latest_metrics.get(key, 0)
        for key, value in payload.items():
            normalized = normalize_key(key)
            if normalized in METRIC_KEYS:
                record[normalized] = coerce_value(value)
        latest_metrics.update(record)
    append_history(record)
    return record


def read_history(limit: int = 200) -> list[dict[str, object]]:
    ensure_storage()
    if limit < 1:
        return []

    lines: list[str] = []
    with HISTORY_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line)
    sliced = lines[-limit:]

    out: list[dict[str, object]] = []
    for line in sliced:
        try:
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                out.append(normalize_history_record(loaded))
        except Exception:
            continue
    return out


def serial_reader_loop() -> None:
    global serial_conn
    while True:
        if serial is None:
            time.sleep(2)
            continue

        try:
            with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as esp32:
                time.sleep(1.5)
                with serial_lock:
                    serial_conn = esp32
                while True:
                    raw = esp32.readline().decode("utf-8", errors="ignore").strip()
                    if not raw:
                        time.sleep(0.02)
                        continue
                    payload = parse_serial_payload(raw)
                    if payload:
                        push_metrics(payload)
        except Exception:
            with serial_lock:
                serial_conn = None
            time.sleep(2)


def start_serial_reader_once() -> None:
    global serial_thread_started
    if serial_thread_started:
        return
    serial_thread_started = True
    t = threading.Thread(target=serial_reader_loop, daemon=True)
    t.start()


def send_serial_command(command: str) -> bool:
    if not command:
        return False
    if serial is None:
        return False
    cmd_bytes = command.encode("ascii", errors="ignore")
    if not cmd_bytes:
        return False

    with serial_lock:
        if serial_conn is not None and getattr(serial_conn, "is_open", False):
            try:
                serial_conn.write(cmd_bytes)
                serial_conn.flush()
                return True
            except Exception:
                return False

    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as port:
            port.write(cmd_bytes)
            port.flush()
        return True
    except Exception:
        return False


def update_detection_state(**kwargs: object) -> None:
    with detection_lock:
        detection_state.update(kwargs)


def get_detection_state() -> dict[str, object]:
    with detection_lock:
        return dict(detection_state)


def get_or_load_detection_model():
    global detection_model
    if detection_model is not None:
        return detection_model
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed")
    detection_model = YOLO(MODEL_PATH)
    return detection_model


def detection_loop() -> None:
    global last_frame_jpeg
    if cv2 is None:
        update_detection_state(running=False, message="opencv-python not installed")
        return

    try:
        model = get_or_load_detection_model()
    except Exception as exc:
        update_detection_state(running=False, message=f"model load failed: {exc}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        update_detection_state(running=False, message="camera open failed")
        return
    # Reduce capture lag and keep UI stream smoother.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_index = 0
    prev_time = 0.0
    last_sent = {"C": 0.0, "S": 0.0}
    crack_labels = {"cracked", "crack"}
    soiled_labels = {"soiled", "soil"}
    cached_boxes: list[tuple[int, int, int, int, tuple[int, int, int], str]] = []
    cracked_count = 0
    soiled_count = 0

    try:
        while True:
            if not get_detection_state().get("running"):
                break

            ok, frame = cap.read()
            if not ok:
                update_detection_state(message="camera read failed")
                break

            frame_index += 1
            should_infer = frame_index % max(1, DETECTION_FRAME_SKIP) == 0
            if should_infer:
                results = model(
                    frame,
                    imgsz=DETECTION_IMGSZ,
                    conf=0.35,
                    iou=DETECTION_IOU,
                    device="cpu",
                    verbose=False,
                )
                cracked_count = 0
                soiled_count = 0
                cached_boxes = []
                result = results[0]

                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = str(model.names[class_id]).lower()
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if class_name in crack_labels:
                        cracked_count += 1
                        color = (0, 0, 255)
                        tag = "CRACKED"
                    elif class_name in soiled_labels:
                        soiled_count += 1
                        color = (0, 255, 255)
                        tag = "SOILED"
                    else:
                        continue

                    cached_boxes.append((x1, y1, x2, y2, color, f"{tag} {conf:.2f}"))

            for x1, y1, x2, y2, color, label in cached_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            now = time.time()
            last_command = ""

            if should_infer:
                if cracked_count > 0 and now - last_sent["C"] >= DETECTION_COOLDOWN:
                    if send_serial_command("C"):
                        last_sent["C"] = now
                        last_command = "C"

                if soiled_count > 0 and now - last_sent["S"] >= DETECTION_COOLDOWN:
                    if send_serial_command("S"):
                        last_sent["S"] = now
                        last_command = "S" if not last_command else "C+S"

            fps = int(1 / (now - prev_time)) if prev_time else 0
            prev_time = now

            ok_jpg, jpg = cv2.imencode(".jpg", frame)
            if ok_jpg:
                last_frame_jpeg = jpg.tobytes()

            update_detection_state(
                cracked=cracked_count,
                soiled=soiled_count,
                fps=fps,
                last_command=last_command,
                message="running",
                last_ts=now_iso(),
            )
    finally:
        cap.release()
        update_detection_state(running=False)


def video_stream_generator():
    while True:
        if not get_detection_state().get("running"):
            break
        frame = last_frame_jpeg
        if frame is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.route("/")
def index():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email or not password:
            return render_template("login.html", error="Email and password are required")

        users = load_users()
        if email not in users:
            return render_template("login.html", error="User not found. Please sign up.")

        if not check_password_hash(users[email]["password"], password):
            return render_template("login.html", error="Invalid password")

        session["user"] = email
        session["username"] = users[email]["username"]
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password or not confirm_password:
            return render_template("signup.html", error="All fields are required")
        if len(password) < 6:
            return render_template("signup.html", error="Password must be at least 6 characters")
        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match")

        users = load_users()
        if email in users:
            return render_template("signup.html", error="Email already registered")

        users[email] = {"username": username, "password": generate_password_hash(password)}
        save_users(users)
        return redirect(url_for("login", message="Account created successfully! Please log in."))

    return render_template("signup.html")


@app.route("/dashboard")
@login_required
def dashboard():
    seed_initial_history_if_empty()
    start_serial_reader_once()
    return render_template("dashboard.html", username=session.get("username", "User"))


@app.route("/api/esp32/latest")
@login_required
def api_latest():
    with state_lock:
        snapshot = dict(latest_metrics)
    return jsonify(snapshot)


@app.route("/api/esp32/history")
@login_required
def api_history():
    limit = request.args.get("limit", default=50, type=int)
    limit = max(1, min(limit, 1000))
    return jsonify({"items": read_history(limit=limit)})


@app.route("/api/esp32/push", methods=["POST"])
def api_push():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid payload"}), 400
    record = push_metrics(data)
    return jsonify({"ok": True, "record": record})


@app.route("/api/detection/start", methods=["POST"])
@login_required
def api_detection_start():
    if get_detection_state().get("running"):
        return jsonify({"ok": True, "state": get_detection_state()})
    update_detection_state(
        running=True,
        cracked=0,
        soiled=0,
        fps=0,
        last_command="",
        message="starting",
        last_ts=now_iso(),
    )
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    return jsonify({"ok": True, "state": get_detection_state()})


@app.route("/api/detection/stop", methods=["POST"])
@login_required
def api_detection_stop():
    update_detection_state(running=False, message="stopped")
    return jsonify({"ok": True, "state": get_detection_state()})


@app.route("/api/detection/status")
@login_required
def api_detection_status():
    return jsonify(get_detection_state())


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(video_stream_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(error):
    return render_template("500.html"), 500


if __name__ == "__main__":
    ensure_storage()
    seed_initial_history_if_empty()
    app.run(debug=True, host="0.0.0.0", port=5000)
