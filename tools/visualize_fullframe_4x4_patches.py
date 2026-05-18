#!/usr/bin/env python3
"""Overlay 4x4 patch mask on every decoded frame.

This is intentionally minimal: decode each frame, compute the 4x4 text-like
patch proxy on the full image, paint those patches, and save the result. It
does not draw subtitle boxes, clusters, labels, or other guide graphics.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from subtitle_event_codec_probe import ffprobe_frames, get_video_shape, text_block_mask


@dataclass
class FramePatchMetric:
    idx: int
    time: float
    pict_type: str
    pkt_size: int
    patch_count: int
    patch_density: float
    image_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-frame 4x4 patch overlay for every decoded frame.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--block", type=int, default=4)
    parser.add_argument("--text-edge-thr", type=float, default=34.0)
    parser.add_argument("--bright-thr", type=int, default=155)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--write-video", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def overlay_mask(frame: np.ndarray, mask: np.ndarray, block: int, alpha: float) -> np.ndarray:
    out = frame.copy()
    bh, bw = mask.shape
    h = bh * block
    w = bw * block
    pixel_mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    patch_layer = out[:h, :w]
    yellow = np.zeros_like(patch_layer)
    yellow[:, :] = (0, 230, 255)
    patch_layer[pixel_mask] = cv2.addWeighted(yellow, alpha, patch_layer, 1.0 - alpha, 0)[pixel_mask]
    out[:h, :w] = patch_layer
    return out


def write_csv(path: Path, rows: list[FramePatchMetric]) -> None:
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

    frame_infos = ffprobe_frames(video)
    width, height, fps, _ = get_video_shape(video)
    writer = None
    if args.write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_dir / "patch_overlay.mp4"), fourcc, fps or 25.0, (width, height))
        if not writer.isOpened():
            writer = None

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    rows: list[FramePatchMetric] = []
    idx = 0
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames is not None and idx >= args.max_frames:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = text_block_mask(gray, args.block, args.text_edge_thr, args.bright_thr)
        overlay = overlay_mask(frame, mask, args.block, args.alpha)
        image_path = ""
        if idx % max(1, args.save_every) == 0:
            out_path = frames_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out_path), overlay, params)
            image_path = str(out_path)
        if writer is not None:
            writer.write(overlay)
        info = frame_infos[idx] if idx < len(frame_infos) else None
        rows.append(
            FramePatchMetric(
                idx=idx,
                time=float(info.time if info else idx / max(1.0, fps)),
                pict_type=str(info.pict_type if info else "?"),
                pkt_size=int(info.pkt_size if info else 0),
                patch_count=int(mask.sum()),
                patch_density=float(mask.mean()),
                image_path=image_path,
            ),
        )
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    write_csv(out_dir / "patch_metrics.csv", rows)
    print(f"video={video}")
    print(f"frames={len(rows)} saved_images={sum(1 for row in rows if row.image_path)}")
    print(f"patch_overlay_video={out_dir / 'patch_overlay.mp4' if args.write_video else ''}")
    print(f"frames_dir={frames_dir}")
    print(f"csv={out_dir / 'patch_metrics.csv'}")


if __name__ == "__main__":
    main()
