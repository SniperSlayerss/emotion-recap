"""
fix_fps.py

Re-encode session videos at the correct framerate.

The capture script told ffmpeg the videos were 10 fps, but Picamera2 was
actually capturing at its default rate (~30 fps). So playback is 3× too
fast. This script uses the known session duration (from session.json) to
figure out the true framerate, then re-encodes at that rate.

The SPS headers in the original h264 stream have the wrong fps baked in,
so `-c copy` inherits the wrong timing. Re-encoding rebuilds the bitstream
with fresh timestamps (takes 1–3 min per session but is the only robust fix).

Usage:
    python fix_fps.py <parent_dir>                # dry-run, shows plan
    python fix_fps.py <parent_dir> --apply        # actually fix

    python fix_fps.py <single_session> --apply    # one session

    --backup        Keep original as video_original.mp4 (default: on)
    --no-backup     Overwrite original without backup
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def ffprobe_field(path: Path, entry: str) -> str | None:
    """Run ffprobe and return a single field, or None on failure."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", entry, "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def video_info(path: Path) -> tuple[int | None, float | None]:
    """Return (frame_count, duration_seconds) for a video."""
    nb = ffprobe_field(path, "stream=nb_frames")
    dur = ffprobe_field(path, "format=duration")

    frames = None
    if nb and nb.isdigit():
        frames = int(nb)
    elif nb:
        # Container sometimes stores N/A; count manually as fallback
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-count_frames",
                 "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, check=True,
            )
            val = result.stdout.strip()
            if val.isdigit():
                frames = int(val)
        except subprocess.CalledProcessError:
            pass

    duration = None
    if dur:
        try:
            duration = float(dur)
        except ValueError:
            pass

    return frames, duration


def plan_session(session_dir: Path) -> dict | None:
    """
    Work out what to do for one session.
    Returns a plan dict or None if session can't be fixed.
    """
    mp4  = session_dir / "video.mp4"
    h264 = session_dir / "video.h264"
    meta = session_dir / "session.json"

    if not mp4.exists() and not h264.exists():
        return {"dir": session_dir, "status": "no_video"}

    if not meta.exists():
        return {"dir": session_dir, "status": "no_meta"}

    try:
        with open(meta) as f:
            session_duration = float(json.load(f).get("duration_s", 0))
    except (json.JSONDecodeError, OSError, ValueError):
        return {"dir": session_dir, "status": "bad_meta"}

    if session_duration <= 0:
        return {"dir": session_dir, "status": "zero_duration"}

    # Prefer h264 if available (original, untouched)
    source = h264 if h264.exists() else mp4
    frames, current_duration = video_info(source)

    if frames is None:
        return {"dir": session_dir, "status": "no_frame_count", "source": source}

    true_fps = frames / session_duration

    return {
        "dir":              session_dir,
        "status":           "fixable",
        "source":           source,
        "frames":           frames,
        "session_duration": session_duration,
        "current_duration": current_duration,
        "true_fps":         true_fps,
    }


def apply_fix(plan: dict, backup: bool) -> bool:
    """Re-mux the source at the correct fps. Returns True on success."""
    session_dir = plan["dir"]
    source      = plan["source"]
    true_fps    = plan["true_fps"]
    target      = session_dir / "video.mp4"

    if backup and target.exists() and source != target:
        # Source is h264, target mp4 already exists — back it up
        backup_path = session_dir / "video_original.mp4"
        if not backup_path.exists():
            shutil.copy2(target, backup_path)
            print(f"    Backed up existing mp4 → {backup_path.name}")

    if backup and source == target:
        # Re-muxing the mp4 itself — must back up first
        backup_path = session_dir / "video_original.mp4"
        if not backup_path.exists():
            shutil.move(target, backup_path)
            source = backup_path
            print(f"    Backed up original → {backup_path.name}")
        else:
            source = backup_path

    # Re-encode with the correct framerate. The SPS headers in the original
    # h264 stream have 10fps baked in, so any -c copy approach inherits the
    # wrong timing. Re-encoding rebuilds the bitstream with fresh timestamps.
    tmp_out = session_dir / "video.fixed.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-r", f"{true_fps:.4f}",     # input rate — tells decoder how to time frames
        "-i", str(source),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",                # visually ~lossless, reasonable file size
        "-r", f"{true_fps:.4f}",     # output rate — ensures container matches
        "-pix_fmt", "yuv420p",       # maximum compatibility
        "-c:a", "copy",              # audio unchanged if present
        "-movflags", "+faststart",   # playable before fully downloaded
        str(tmp_out),
    ]
    print(f"    Re-encoding at {true_fps:.2f} fps (this takes ~1-3 min)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [FAIL] ffmpeg error:\n{result.stderr[-800:]}")
        return False

    tmp_out.replace(target)
    print(f"    [OK] Re-encoded at {true_fps:.2f} fps → {target.name}")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="Session dir or parent containing sessions")
    p.add_argument("--apply", action="store_true", help="Actually fix (default: dry-run)")
    p.add_argument("--backup", dest="backup", action="store_true", default=True)
    p.add_argument("--no-backup", dest="backup", action="store_false")
    args = p.parse_args()

    # Check ffmpeg / ffprobe exist
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"[ERROR] {tool} not found in PATH.")

    if not args.path.is_dir():
        sys.exit(f"[ERROR] Not a directory: {args.path}")

    # Single session vs parent?
    if (args.path / "session.json").exists() or (args.path / "features.csv").exists():
        sessions = [args.path]
    else:
        sessions = sorted(
            d for d in args.path.iterdir()
            if d.is_dir() and ((d / "video.mp4").exists() or (d / "video.h264").exists())
        )

    if not sessions:
        sys.exit(f"[ERROR] No sessions with video found in {args.path}")

    print(f"\n[FPS-FIX] Found {len(sessions)} session(s) with video.")
    print(f"[FPS-FIX] Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"[FPS-FIX] Backup originals: {args.backup}\n")

    plans = [plan_session(s) for s in sessions]

    # Report
    fixable = [p for p in plans if p and p["status"] == "fixable"]
    skipped = [p for p in plans if p and p["status"] != "fixable"]

    print("─" * 90)
    print(f"{'session':<40} {'frames':>8} {'sess dur':>10} {'cur dur':>10} {'true fps':>10}")
    print("─" * 90)
    for plan in plans:
        if plan is None:
            continue
        name = plan["dir"].name
        if plan["status"] != "fixable":
            print(f"{name[:40]:<40}   [SKIP: {plan['status']}]")
            continue
        cur = f"{plan['current_duration']:.1f}s" if plan["current_duration"] else "?"
        print(
            f"{name[:40]:<40} "
            f"{plan['frames']:>8} "
            f"{plan['session_duration']:>9.1f}s "
            f"{cur:>10} "
            f"{plan['true_fps']:>9.2f}"
        )
    print("─" * 90)
    print(f"Fixable: {len(fixable)}   Skipped: {len(skipped)}\n")

    if not args.apply:
        print("[DRY-RUN] No changes made. Add --apply to re-mux.\n")
        return 0

    if not fixable:
        return 0

    # Apply
    ok = 0
    for plan in fixable:
        print(f"[FIX] {plan['dir'].name}")
        if apply_fix(plan, backup=args.backup):
            ok += 1

    print(f"\n[FPS-FIX] Done — {ok}/{len(fixable)} fixed.")
    return 0 if ok == len(fixable) else 1


if __name__ == "__main__":
    sys.exit(main())
