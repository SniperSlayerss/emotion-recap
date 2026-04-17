"""
clip_viewer/app.py

Local web UI for browsing arousal clips by session.

Usage:
    python app.py [--sessions <path>] [--port <port>]

    --sessions   Root directory containing session folders (default: ../sessions)
    --port       Port to serve on (default: 5000)

Then open http://localhost:5000 in your browser.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, send_file

app = Flask(__name__)
SESSIONS_DIR: Path = Path("../sessions")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_sessions() -> list[dict]:
    """Scan sessions dir and return metadata for all sessions."""
    sessions = []
    if not SESSIONS_DIR.exists():
        return sessions

    for session_dir in sorted(SESSIONS_DIR.iterdir(), reverse=True):
        if not session_dir.is_dir():
            continue

        meta_path = session_dir / "session.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        clips = get_clips(session_dir / "clips")

        parts = session_dir.name.split("_", 2)
        try:
            dt = datetime.strptime(f"{parts[0]}_{parts[1]}", "%Y%m%d_%H%M%S")
        except (ValueError, IndexError):
            dt = datetime.fromtimestamp(session_dir.stat().st_mtime)

        sessions.append({
            "id":           session_dir.name,
            "label":        meta.get("label", parts[2] if len(parts) > 2 else "unknown"),
            "date":         dt.strftime("%A, %d %B %Y"),
            "time":         dt.strftime("%H:%M"),
            "datetime":     dt.isoformat(),
            "duration_s":   meta.get("duration_s", 0),
            "n_clips":      len(clips),
            "clips":        clips,
            "has_features": (session_dir / "features.csv").exists(),
        })

    return sessions


def get_clips(session_dir: Path) -> list[dict]:
    """Return all clips for a session, sorted by time."""
    clips_dir = session_dir
    clips = []
    if not clips_dir.exists():
        return clips

    for clip_dir in sorted(clips_dir.iterdir()):
        if not clip_dir.is_dir():
            continue

        meta_path  = clip_dir / "meta.json"
        video_path = clip_dir / "video.mp4"
        audio_path = clip_dir / "audio.wav"

        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        name_parts = clip_dir.name.replace("clip_", "")
        try:
            dt = datetime.strptime(name_parts, "%Y%m%d_%H%M%S")
            time_str = dt.strftime("%H:%M:%S")
        except ValueError:
            time_str = clip_dir.name

        features = meta.get("features", {})
        clips.append({
            "id":           clip_dir.name,
            "session_id":   session_dir.name,
            "time":         time_str,
            "score":        meta.get("anomaly_score"),
            "norm":         meta.get("anomaly_norm"),
            "video_buffer": meta.get("video_buffer_s", 6),
            "audio_pre":    meta.get("audio_pre_s", 6),
            "audio_post":   meta.get("audio_post_s", 10),
            "triggered_at": meta.get("triggered_at", ""),
            "has_video":    video_path.exists(),
            "has_audio":    audio_path.exists(),
            "features": {
                "scl_mean":   features.get("gsr_scl_mean"),
                "scr_count":  features.get("gsr_scr_count"),
                "phasic_std": features.get("gsr_phasic_std"),
                "scr_amp":    features.get("gsr_scr_mean_amp"),
                "hr_mean":    features.get("hrv_hr_mean"),
                "rmssd":      features.get("hrv_rmssd"),
                "sdnn":       features.get("hrv_sdnn"),
                "pnn50":      features.get("hrv_pnn50"),
            },
        })

    return clips


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    sessions = get_sessions()
    return render_template("index.html", sessions=sessions,
                           sessions_dir=str(SESSIONS_DIR.resolve()))


@app.route("/api/sessions")
def api_sessions():
    return jsonify(get_sessions())


@app.route("/media/<session_id>/clips/<clip_id>/video")
def serve_video(session_id, clip_id):
    path = SESSIONS_DIR / session_id / "clips" / clip_id / "video.mp4"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="video/mp4")


@app.route("/media/<session_id>/clips/<clip_id>/audio")
def serve_audio(session_id, clip_id):
    path = SESSIONS_DIR / session_id / "clips" / clip_id / "audio.wav"
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="audio/wav")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", default="../sessions", help="Sessions root directory")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    global SESSIONS_DIR
    SESSIONS_DIR = Path(args.sessions)

    print(f"[VIEWER] Sessions dir : {SESSIONS_DIR.resolve()}")
    print(f"[VIEWER] Serving at   : http://localhost:{args.port}")
    print(f"[VIEWER] LAN access   : http://<pi-ip>:{args.port}")
    print(f"[VIEWER] Ctrl+C to stop")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
