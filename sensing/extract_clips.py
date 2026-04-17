"""
extract_clips.py

Given a session and a trained arousal detector, extract video clips around
detected high-arousal events.

GSR and HRV windows are merged temporally before scoring (see
session_scoring.py). Adjacent flagged windows are grouped into events, and
one clip is extracted per event.

Usage:
    python extract_clips.py <session_dir> --model models/iforest.pkl

    --pre          Seconds before event start (default: 45)
    --post         Seconds after event end    (default: 30)
    --merge-gap    Merge flags within this many seconds into one event
                   (default: 60). Set to 0 to disable merging.
    --max-duration Cap clip duration — if an event would produce a clip
                   longer than this, split into chunks (default: 180)
    --min-score    Only extract events with peak normalised score >= this
                   (default: 0.0)
    --top-n        Only extract the N highest-scoring events (default: all)
    --out-dir      Where to put clips (default: <session_dir>/clips/)
    --reencode     Re-encode clips for frame-accurate cuts (default: off)
    --tolerance    GSR/HRV merge tolerance in seconds (default: 20)
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

from arousal_detector import ArousalDetector
from session_scoring import score_session_file, DEFAULT_MERGE_TOLERANCE_S


def format_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}-{s:02d}"


def format_time_human(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Event merging / splitting
# ---------------------------------------------------------------------------


def merge_events(flags: list[dict], merge_gap: float) -> list[dict]:
    if not flags:
        return []
    if merge_gap <= 0:
        return [_flag_to_event(f) for f in flags]

    events: list[list[dict]] = [[flags[0]]]
    for f in flags[1:]:
        if f["time_s"] - events[-1][-1]["time_s"] <= merge_gap:
            events[-1].append(f)
        else:
            events.append([f])
    return [_event_summary(group) for group in events]


def _flag_to_event(f: dict) -> dict:
    return {
        "start_s":         f["time_s"],
        "end_s":           f["time_s"],
        "peak_time_s":     f["time_s"],
        "peak_normalised": f["normalised"],
        "peak_score":      f["score"],
        "n_flags":         1,
    }


def _event_summary(group: list[dict]) -> dict:
    peak = max(group, key=lambda f: f["normalised"])
    return {
        "start_s":         min(f["time_s"] for f in group),
        "end_s":           max(f["time_s"] for f in group),
        "peak_time_s":     peak["time_s"],
        "peak_normalised": peak["normalised"],
        "peak_score":      peak["score"],
        "n_flags":         len(group),
    }


def split_long_events(events: list[dict], max_duration: float,
                      pre: float, post: float) -> list[dict]:
    """
    If an event would produce a clip longer than max_duration, split it
    into chunks. Prevents a single mega-event from eating the whole session.
    """
    out = []
    for ev in events:
        clip_len = (ev["end_s"] - ev["start_s"]) + pre + post
        if clip_len <= max_duration:
            out.append(ev)
            continue

        chunk = max(30.0, max_duration - pre - post)
        t = ev["start_s"]
        while t <= ev["end_s"]:
            end = min(t + chunk, ev["end_s"])
            out.append({
                "start_s":         t,
                "end_s":           end,
                "peak_time_s":     (t + end) / 2,
                "peak_normalised": ev["peak_normalised"],
                "peak_score":      ev["peak_score"],
                "n_flags":         ev["n_flags"],
                "split":           True,
            })
            t = end + 0.001
    return out


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------


def extract_clip(video: Path, start_s: float, duration_s: float,
                 out: Path, reencode: bool) -> bool:
    if reencode:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video),
            "-ss", f"{start_s:.3f}",
            "-t",  f"{duration_s:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(out),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_s:.3f}",
            "-i", str(video),
            "-t", f"{duration_s:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(out),
        ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"      [FAIL] {result.stderr[-400:]}")
        return False
    return True


def get_video_duration(video: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True,
    )
    return float(probe.stdout.strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("session_dir", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--pre",  type=float, default=45.0)
    p.add_argument("--post", type=float, default=30.0)
    p.add_argument("--merge-gap", type=float, default=60.0)
    p.add_argument("--max-duration", type=float, default=180.0)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--reencode", action="store_true")
    p.add_argument("--tolerance", type=float, default=DEFAULT_MERGE_TOLERANCE_S)
    args = p.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("[ERROR] ffmpeg not in PATH")

    csv_path   = args.session_dir / "features.csv"
    video_path = args.session_dir / "video.mp4"
    if not csv_path.exists():
        sys.exit(f"[ERROR] No features.csv in {args.session_dir}")
    if not video_path.exists():
        sys.exit(f"[ERROR] No video.mp4 in {args.session_dir}")

    out_dir = args.out_dir or (args.session_dir / "clips")
    out_dir.mkdir(parents=True, exist_ok=True)

    video_duration = get_video_duration(video_path)

    print(f"[EXTRACT] Session:     {args.session_dir}")
    print(f"[EXTRACT] Video:       {video_path.name} ({video_duration:.1f}s)")
    print(f"[EXTRACT] Padding:     -{args.pre:.0f}s / +{args.post:.0f}s")
    print(f"[EXTRACT] Merge gap:   {args.merge_gap:.0f}s")
    print(f"[EXTRACT] Max clip:    {args.max_duration:.0f}s")
    print(f"[EXTRACT] GSR/HRV tol: {args.tolerance:.0f}s\n")

    detector = ArousalDetector(args.model)
    results  = score_session_file(csv_path, detector, tolerance_s=args.tolerance)

    scored  = int(results["scored"].sum())
    flagged = int(results["is_aroused"].sum())
    print(f"[EXTRACT] Merged windows: {len(results)} "
          f"({scored} scored, {flagged} flagged)\n")

    flags = [
        {"time_s": r["time_s"], "score": r["score"], "normalised": r["normalised"]}
        for _, r in results.iterrows()
        if r["scored"] and r["is_aroused"]
    ]
    flags.sort(key=lambda f: f["time_s"])

    events = merge_events(flags, args.merge_gap)
    events = split_long_events(events, args.max_duration, args.pre, args.post)
    print(f"[EXTRACT] {len(flags)} flags → {len(events)} event(s) after merge/split")

    events = [e for e in events if e["peak_normalised"] >= args.min_score]
    if args.top_n is not None:
        events = sorted(events, key=lambda e: e["peak_normalised"], reverse=True)[:args.top_n]
    events.sort(key=lambda e: e["peak_time_s"])

    if not events:
        print("[EXTRACT] No events matched. Nothing to extract.")
        return 0

    print(f"[EXTRACT] Extracting {len(events)} clip(s)...\n")

    index_rows = []
    ok = 0

    for i, ev in enumerate(events, 1):
        start = max(0.0, ev["start_s"] - args.pre)
        end   = min(video_duration, ev["end_s"] + args.post)
        dur   = end - start

        if dur < 1.0:
            print(f"  [{i:02d}] SKIP — clip too short after clamping")
            continue

        name = (f"clip_{i:02d}_peak-{format_time(ev['peak_time_s'])}"
                f"_norm-{ev['peak_normalised']:.2f}.mp4")
        out_path = out_dir / name

        tag = " [split]" if ev.get("split") else ""
        print(f"  [{i:02d}] event {format_time_human(ev['start_s'])}–"
              f"{format_time_human(ev['end_s'])}{tag} "
              f"(peak {format_time_human(ev['peak_time_s'])}, "
              f"{ev['n_flags']} flag{'s' if ev['n_flags'] > 1 else ''})")
        print(f"       clip: {format_time_human(start)}–{format_time_human(end)} "
              f"({dur:.1f}s) → {name}")

        if extract_clip(video_path, start, dur, out_path, args.reencode):
            ok += 1
            index_rows.append({
                "clip":              name,
                "event_start_s":     round(ev["start_s"], 2),
                "event_end_s":       round(ev["end_s"], 2),
                "peak_time_s":       round(ev["peak_time_s"], 2),
                "peak_normalised":   round(ev["peak_normalised"], 4),
                "peak_score":        round(ev["peak_score"], 4),
                "n_flags":           ev["n_flags"],
                "clip_start_s":      round(start, 2),
                "clip_end_s":        round(end, 2),
                "clip_duration_s":   round(dur, 2),
                "was_split":         bool(ev.get("split", False)),
            })

    if index_rows:
        index_path = out_dir / "clips.csv"
        with open(index_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            writer.writeheader()
            writer.writerows(index_rows)
        print(f"\n[EXTRACT] Index saved: {index_path}")

    print(f"[EXTRACT] Done — {ok}/{len(events)} clip(s) extracted to {out_dir}/")
    return 0 if ok == len(events) else 1


if __name__ == "__main__":
    sys.exit(main())
