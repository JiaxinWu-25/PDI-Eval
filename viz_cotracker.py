"""Visualize CoTracker tracking results from cached .npz files.

Usage:
    python viz_cotracker.py --npz output/cache/VIDEO_ID_cotracker.npz \
                            --video path/to/video.mp4 \
                            --output cotracker_viz.mp4 \
                            [--trail 20] [--show_bg]
"""
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def load_cache(npz_path: str) -> dict:
    with np.load(npz_path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def make_color_palette(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(60, 255, size=(n, 3)).tolist()


def render(
    video_path: str,
    tracks: np.ndarray,
    visibility: np.ndarray,
    output_path: str,
    trail: int = 20,
    bg_tracks: np.ndarray = None,
    bg_visibility: np.ndarray = None,
    show_bg: bool = False,
):
    """Render tracking visualization onto video frames.

    Args:
        tracks:     (T, N, 2) foreground xy coordinates (original resolution)
        visibility: (T, N)    foreground visibility scores
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    T, N, _ = tracks.shape
    fg_colors = make_color_palette(N, seed=42)

    has_bg = show_bg and bg_tracks is not None and bg_tracks.size > 0
    if has_bg:
        M = bg_tracks.shape[1]
        bg_colors = make_color_palette(M, seed=7)

    for t in tqdm(range(T), desc="Rendering"):
        ret, frame = cap.read()
        if not ret:
            break

        # background tracks (dim, thin lines)
        if has_bg:
            for i in range(M):
                if bg_visibility[t, i] < 0.5:
                    continue
                for s in range(max(0, t - trail), t):
                    if bg_visibility[s, i] < 0.5:
                        continue
                    p1 = (int(bg_tracks[s, i, 0]), int(bg_tracks[s, i, 1]))
                    p2 = (int(bg_tracks[s + 1, i, 0]), int(bg_tracks[s + 1, i, 1]))
                    cv2.line(frame, p1, p2, bg_colors[i], 1, cv2.LINE_AA)
                curr = (int(bg_tracks[t, i, 0]), int(bg_tracks[t, i, 1]))
                cv2.circle(frame, curr, 2, bg_colors[i], -1)

        # foreground tracks (bright, thick)
        for i in range(N):
            if visibility[t, i] < 0.5:
                continue
            color = fg_colors[i]
            # trail lines with fade effect
            for s in range(max(0, t - trail), t):
                if visibility[s, i] < 0.5:
                    continue
                alpha = (s - (t - trail)) / trail  # 0 -> 1 (older -> newer)
                faded = [int(c * (0.3 + 0.7 * alpha)) for c in color]
                p1 = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                p2 = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                cv2.line(frame, p1, p2, faded, 3, cv2.LINE_AA)
            curr = (int(tracks[t, i, 0]), int(tracks[t, i, 1]))
            cv2.circle(frame, curr, 7, color, -1)
            cv2.circle(frame, curr, 7, (255, 255, 255), 2)  # white border

        cv2.putText(frame, f"CoTracker | FG pts: {N}  Frame: {t}/{T-1}",
                    (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Path to *_cotracker.npz cache file")
    parser.add_argument("--video", required=True, help="Corresponding input video path")
    parser.add_argument("--output", default="cotracker_viz.mp4")
    parser.add_argument("--trail", type=int, default=50, help="Trail length in frames")
    parser.add_argument("--show_bg", action="store_true", help="Also draw background tracks")
    args = parser.parse_args()

    print(f"Loading cache: {args.npz}")
    data = load_cache(args.npz)

    tracks = data["tracks"]          # (T, N, 2)
    visibility = data["visibility"]  # (T, N)
    bg_tracks = data.get("bg_tracks")
    bg_visibility = data.get("bg_visibility")

    T, N, _ = tracks.shape
    print(f"  Foreground: {N} pts x {T} frames")
    if bg_tracks is not None and bg_tracks.size > 0:
        print(f"  Background: {bg_tracks.shape[1]} pts x {T} frames")

    render(
        video_path=args.video,
        tracks=tracks,
        visibility=visibility,
        output_path=args.output,
        trail=args.trail,
        bg_tracks=bg_tracks,
        bg_visibility=bg_visibility,
        show_bg=args.show_bg,
    )


if __name__ == "__main__":
    main()
