#!/usr/bin/env python3
"""Overlay H.264 decoder macroblock type information.

This script uses FFmpeg's native H.264 decoder debug output:

    ffmpeg -debug mb_type ...

The parsed map is codec-side macroblock information, not an image-derived
edge/brightness proxy. For H.264, the debug map is 16x16 macroblock-level:
"i" denotes an intra4x4-coded macroblock and "I" denotes an intra16x16-coded
macroblock. The exact internal 4x4 sub-block positions inside an "i" macroblock
are not exposed by stock FFmpeg's debug text, so the full 16x16 macroblock is
painted when it matches the requested symbols.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from subtitle_event_codec_probe import ffprobe_frames, get_video_shape


NEW_FRAME_RE = re.compile(r"New frame, type:\s*([IPB])")
ROW_RE = re.compile(r"\[h264[^\]]*\]\s+(\d+)\s+(.+)$")


@dataclass
class MBTypeMetric:
    idx: int
    time: float
    pict_type: str
    ffmpeg_debug_type: str
    pkt_size: int
    selected_mb_count: int
    selected_mb_density: float
    image_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize original H.264 mb_type debug maps.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--symbols", default="i,I", help="Comma-separated mb_type symbols to paint, e.g. i,I or i.")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--color", default="0,230,255", help="B,G,R overlay color.")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--write-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-log", action="store_true", help="Save raw ffmpeg debug log.")
    return parser.parse_args()


def parse_color(value: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--color must be B,G,R")
    return tuple(max(0, min(255, x)) for x in parts)


def parse_mbtype_maps(video: Path, width: int, height: int, max_frames: int | None, keep_log_path: Path | None) -> tuple[list[np.ndarray], list[str]]:
    mb_w = (width + 15) // 16
    mb_h = (height + 15) // 16
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "debug",
        "-threads",
        "1",
        "-debug",
        "mb_type",
        "-i",
        str(video),
        "-an",
        "-sn",
        "-dn",
        "-map",
        "0:v:0",
    ]
    if max_frames is not None:
        cmd.extend(["-frames:v", str(max_frames)])
    cmd.extend(["-f", "null", "-"])

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1)
    maps: list[np.ndarray] = []
    frame_types: list[str] = []
    current_rows: list[list[str]] = []
    current_type = "?"
    raw_log = []

    def flush_current() -> None:
        nonlocal current_rows, current_type
        if len(current_rows) == mb_h:
            mask = np.zeros((mb_h, mb_w), dtype=object)
            for y, row in enumerate(current_rows):
                mask[y, : min(mb_w, len(row))] = row[:mb_w]
            maps.append(mask)
            frame_types.append(current_type)
        current_rows = []

    assert proc.stderr is not None
    for line in proc.stderr:
        if keep_log_path is not None:
            raw_log.append(line)
        new_match = NEW_FRAME_RE.search(line)
        if new_match:
            flush_current()
            current_type = new_match.group(1)
            continue

        row_match = ROW_RE.match(line)
        if not row_match:
            continue
        y = int(row_match.group(1))
        if y % 16 != 0:
            continue
        tokens = row_match.group(2).split()
        if len(tokens) < mb_w:
            continue
        if all(token.isdigit() for token in tokens[: min(len(tokens), mb_w)]):
            continue
        row_idx = y // 16
        if row_idx < 0 or row_idx >= mb_h:
            continue
        if len(current_rows) == row_idx:
            current_rows.append(tokens[:mb_w])
        elif len(current_rows) < row_idx:
            while len(current_rows) < row_idx:
                current_rows.append([""] * mb_w)
            current_rows.append(tokens[:mb_w])
        else:
            current_rows[row_idx] = tokens[:mb_w]

    flush_current()
    ret = proc.wait()
    if keep_log_path is not None:
        keep_log_path.write_text("".join(raw_log), encoding="utf-8", errors="replace")
    if ret != 0:
        raise RuntimeError(f"ffmpeg mb_type debug failed with exit code {ret}")
    return maps, frame_types


def symbol_mask(mb_symbols: np.ndarray, selected_symbols: set[str]) -> np.ndarray:
    out = np.zeros(mb_symbols.shape, dtype=bool)
    for symbol in selected_symbols:
        out |= mb_symbols == symbol
    return out


def overlay_mb_mask(frame: np.ndarray, mask: np.ndarray, alpha: float, color: tuple[int, int, int]) -> np.ndarray:
    out = frame.copy()
    layer = out.copy()
    mb_h, mb_w = mask.shape
    h, w = frame.shape[:2]
    ys, xs = np.where(mask)
    for my, mx in zip(ys, xs):
        x0 = int(mx) * 16
        y0 = int(my) * 16
        x1 = min(w, x0 + 16)
        y1 = min(h, y0 + 16)
        layer[y0:y1, x0:x1] = color
    return cv2.addWeighted(layer, alpha, out, 1.0 - alpha, 0)


def write_csv(path: Path, rows: list[MBTypeMetric]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    args = parse_args()
    video = Path(args.video).resolve()
    out_dir = Path(args.out_dir).resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    width, height, fps, _ = get_video_shape(video)
    frame_infos = ffprobe_frames(video)
    selected_symbols = {s.strip() for s in args.symbols.split(",") if s.strip()}
    color = parse_color(args.color)

    log_path = out_dir / "ffmpeg_mb_type_debug.log" if args.keep_log else None
    mb_maps, debug_types = parse_mbtype_maps(video, width, height, args.max_frames, log_path)
    frame_limit = len(mb_maps)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    writer = None
    if args.write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_dir / "mbtype_intra_overlay.mp4"), fourcc, fps or 25.0, (width, height))
        if not writer.isOpened():
            writer = None

    rows: list[MBTypeMetric] = []
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
    idx = 0
    while idx < frame_limit:
        ok, frame = cap.read()
        if not ok:
            break
        mask = symbol_mask(mb_maps[idx], selected_symbols)
        overlay = overlay_mb_mask(frame, mask, args.alpha, color)
        image_path = ""
        if idx % max(1, args.save_every) == 0:
            out_path = frames_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out_path), overlay, params)
            image_path = str(out_path)
        if writer is not None:
            writer.write(overlay)
        info = frame_infos[idx] if idx < len(frame_infos) else None
        rows.append(
            MBTypeMetric(
                idx=idx,
                time=float(info.time if info else idx / max(1.0, fps)),
                pict_type=str(info.pict_type if info else "?"),
                ffmpeg_debug_type=debug_types[idx] if idx < len(debug_types) else "?",
                pkt_size=int(info.pkt_size if info else 0),
                selected_mb_count=int(mask.sum()),
                selected_mb_density=float(mask.mean()),
                image_path=image_path,
            ),
        )
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    write_csv(out_dir / "mbtype_metrics.csv", rows)
    summary = {
        "video": str(video),
        "width": width,
        "height": height,
        "fps": fps,
        "decoded_frame_count": len(rows),
        "parsed_mbtype_frame_count": len(mb_maps),
        "symbols": sorted(selected_symbols),
        "macroblock_size": 16,
        "note": "Stock FFmpeg mb_type debug is 16x16 macroblock-level; symbol 'i' denotes intra4x4-coded MB.",
        "max_selected_mb_count": max((r.selected_mb_count for r in rows), default=0),
        "mean_selected_mb_count": float(np.mean([r.selected_mb_count for r in rows])) if rows else 0.0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"video={video}")
    print(f"frames={len(rows)} parsed_mb_maps={len(mb_maps)} symbols={','.join(sorted(selected_symbols))}")
    print(f"frames_dir={frames_dir}")
    print(f"video_out={out_dir / 'mbtype_intra_overlay.mp4' if args.write_video else ''}")
    print(f"csv={out_dir / 'mbtype_metrics.csv'}")


if __name__ == "__main__":
    main()
