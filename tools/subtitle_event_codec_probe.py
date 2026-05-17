#!/usr/bin/env python3
"""Probe subtitle on/off/change frames with codec-order inspired cues.

The detector uses display-order frames, but follows a codec-side workflow:

1. Treat large local changes between consecutive non-B references as P-frame
   anchors.
2. For each P anchor, look back through the B frames between the previous
   reference and that P frame.
3. Pick the first B/P frame whose subtitle-band 4x4 block state is closer to
   the P anchor than to the previous reference.

This is intentionally conservative: stock FFmpeg does not expose H.264 intra
4x4 block maps through ffprobe, so the script estimates the local 4x4 "burst"
from luma block changes and text-like edges, then verifies static behavior with
near-frame luma differences.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class FrameInfo:
    idx: int
    time: float
    pict_type: str
    pkt_size: int
    key_frame: int


@dataclass
class FrameMetric:
    idx: int
    time: float
    pict_type: str
    text_density: float
    diff4_density_prev: float
    diff4_text_density_prev: float
    diff4_mean_prev: float
    pkt_size: int


@dataclass
class Event:
    event_id: int
    candidate_idx: int
    candidate_time: float
    candidate_type: str
    anchor_idx: int
    anchor_time: float
    anchor_type: str
    prev_ref_idx: int
    prev_ref_time: float
    prev_ref_type: str
    anchor_lag_frames: int
    kind: str
    score: float
    changed4_density: float
    text_density_prev_ref: float
    text_density_anchor: float
    text_density_delta: float
    dist_to_old: float
    dist_to_new: float
    old_new_margin: float
    post_static_mean: float
    p_pkt_size: int
    p_pkt_size_z: float
    triptych_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect subtitle event candidate frames via P anchors and B-frame backtracking.",
    )
    parser.add_argument("--video", default="xinwen.mp4", help="Input video path.")
    parser.add_argument("--out-dir", default="outputs/subtitle_codec_events/xinwen", help="Output directory.")
    parser.add_argument("--x0", type=int, default=None, help="Subtitle box left x. Defaults to the full width.")
    parser.add_argument("--x1", type=int, default=None, help="Subtitle box right x. Defaults to the full width.")
    parser.add_argument("--y0", type=int, default=None, help="Subtitle band top y. Overrides auto band.")
    parser.add_argument("--y1", type=int, default=None, help="Subtitle band bottom y. Overrides auto band.")
    parser.add_argument("--search-y0-ratio", type=float, default=0.52, help="Auto band search top as a height ratio.")
    parser.add_argument("--search-y1-ratio", type=float, default=0.92, help="Auto band search bottom as a height ratio.")
    parser.add_argument("--band-height-ratio", type=float, default=0.20, help="Auto band height as a height ratio.")
    parser.add_argument("--no-refine-box", action="store_true", help="Disable text-cluster bbox refinement.")
    parser.add_argument("--refine-sample-stride", type=int, default=5, help="Frame stride for bbox refinement.")
    parser.add_argument("--refine-pad-x", type=int, default=24, help="Horizontal padding for refined text bbox.")
    parser.add_argument("--refine-pad-y", type=int, default=18, help="Vertical padding for refined text bbox.")
    parser.add_argument("--refine-min-height", type=int, default=28, help="Minimum refined text bbox height.")
    parser.add_argument("--refine-max-height", type=int, default=180, help="Maximum refined text bbox height before fallback.")
    parser.add_argument("--refine-min-width-ratio", type=float, default=0.28, help="Minimum refined text bbox width as a frame-width ratio.")
    parser.add_argument("--block", type=int, default=4, help="Block size in pixels for local state estimation.")
    parser.add_argument("--diff-thr", type=float, default=16.0, help="4x4 luma mean difference threshold.")
    parser.add_argument("--text-edge-thr", type=float, default=22.0, help="Gradient threshold for text-like blocks.")
    parser.add_argument("--bright-thr", type=int, default=115, help="Brightness threshold used in text-like blocks.")
    parser.add_argument("--anchor-min-density", type=float, default=0.012, help="Minimum changed 4x4 density for a P anchor.")
    parser.add_argument("--i-anchor-min-density", type=float, default=0.015, help="Minimum changed 4x4 density for an I anchor.")
    parser.add_argument("--anchor-mad-k", type=float, default=3.0, help="Robust P anchor threshold multiplier.")
    parser.add_argument("--new-like-margin", type=float, default=2.0, help="Minimum old-vs-new distance margin.")
    parser.add_argument("--min-gap", type=int, default=8, help="Minimum frame gap between kept events.")
    parser.add_argument("--max-events", type=int, default=120, help="Maximum events to visualize.")
    parser.add_argument("--no-auto-band", action="store_true", help="Use default ratio band if y0/y1 are not supplied.")
    parser.add_argument("--include-i-anchors", action=argparse.BooleanOptionalAction, default=True, help="Include I-frame event anchors.")
    parser.add_argument("--include-initial-i", action=argparse.BooleanOptionalAction, default=False, help="Emit frame 0 if it already contains dense text in the refined bbox.")
    parser.add_argument("--initial-text-density-min", type=float, default=0.018, help="Minimum text density for frame-0 initial I candidate.")
    parser.add_argument("--codecview", action="store_true", help="Also render ffmpeg codecview triptychs for kept events.")
    return parser.parse_args()


def run_json(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(proc.stdout)


def ffprobe_frames(video: Path) -> list[FrameInfo]:
    data = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time,pict_type,pkt_size,key_frame",
            "-of",
            "json",
            str(video),
        ],
    )
    frames: list[FrameInfo] = []
    for idx, item in enumerate(data.get("frames", [])):
        if "pict_type" not in item:
            continue
        time_s = item.get("best_effort_timestamp_time")
        frames.append(
            FrameInfo(
                idx=len(frames),
                time=float(time_s) if time_s is not None else float(len(frames)),
                pict_type=str(item.get("pict_type", "?")),
                pkt_size=int(item.get("pkt_size", 0) or 0),
                key_frame=int(item.get("key_frame", 0) or 0),
            ),
        )
    return frames


def get_video_shape(video: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return width, height, fps, nframes


def char_like_pixel_mask(gray_roi: np.ndarray, edge_thr: float, bright_thr: int) -> np.ndarray:
    gray_f = gray_roi.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.abs(gx) + np.abs(gy)
    pixel_mask = (grad > edge_thr) & (gray_roi > bright_thr)
    raw = pixel_mask.astype(np.uint8)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    kept = np.zeros_like(raw, dtype=np.uint8)
    for label in range(1, nlabels):
        x, y, w, h, area = stats[label]
        if area < 3:
            continue
        if area > 900:
            continue
        if w > 150 or h > 90:
            continue
        if w < 2 or h < 2:
            continue
        fill = area / max(1, w * h)
        if fill > 0.75:
            continue
        kept[labels == label] = 1
    return kept.astype(bool)


def text_block_mask(gray_roi: np.ndarray, block: int, edge_thr: float, bright_thr: int) -> np.ndarray:
    pixel_mask = char_like_pixel_mask(gray_roi, edge_thr, bright_thr)
    h, w = gray_roi.shape
    bh = h // block
    bw = w // block
    if bh <= 0 or bw <= 0:
        raise ValueError("Subtitle band is smaller than the block size.")
    cropped = pixel_mask[: bh * block, : bw * block].astype(np.float32)
    density = cv2.resize(cropped, (bw, bh), interpolation=cv2.INTER_AREA)
    return density > 0.08


def block_mean(gray_roi: np.ndarray, block: int) -> np.ndarray:
    h, w = gray_roi.shape
    bh = h // block
    bw = w // block
    cropped = gray_roi[: bh * block, : bw * block]
    return cv2.resize(cropped, (bw, bh), interpolation=cv2.INTER_AREA).astype(np.float32)


def estimate_subtitle_band(
    video: Path,
    height: int,
    width: int,
    args: argparse.Namespace,
) -> tuple[int, int]:
    if args.y0 is not None and args.y1 is not None:
        return max(0, args.y0), min(height, args.y1)

    default_y0 = int(height * 0.62)
    default_y1 = int(height * 0.86)
    if args.no_auto_band:
        return default_y0, default_y1

    search_y0 = int(height * args.search_y0_ratio)
    search_y1 = int(height * args.search_y1_ratio)
    band_h = max(64, int(height * args.band_height_ratio))
    search_y0 = max(0, min(search_y0, height - band_h))
    search_y1 = max(search_y0 + band_h, min(height, search_y1))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    row_score = np.zeros(search_y1 - search_y0, dtype=np.float64)
    prev_roi = None
    sampled = 0
    frame_idx = 0
    stride = 5
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[search_y0:search_y1, :]
            textish = char_like_pixel_mask(roi, args.text_edge_thr, args.bright_thr).astype(np.float32)
            row_score += textish.mean(axis=1)
            if prev_roi is not None:
                diff = cv2.absdiff(roi, prev_roi)
                row_score += 0.25 * (diff > args.diff_thr).mean(axis=1)
            prev_roi = roi
            sampled += 1
        frame_idx += 1
    cap.release()

    if sampled == 0:
        return default_y0, default_y1

    smooth_win = max(9, band_h // 8)
    kernel = np.ones(smooth_win, dtype=np.float64) / smooth_win
    smoothed = np.convolve(row_score / sampled, kernel, mode="same")
    window = np.ones(band_h, dtype=np.float64)
    band_score = np.convolve(smoothed, window, mode="valid")
    best_rel = int(np.argmax(band_score))
    y0 = search_y0 + best_rel
    y1 = min(height, y0 + band_h)

    y0 = max(0, (y0 // args.block) * args.block)
    y1 = min(height, (y1 // args.block) * args.block)
    if y1 - y0 < 64:
        return default_y0, default_y1
    return y0, y1


def _smooth_1d(values: np.ndarray, win: int) -> np.ndarray:
    win = max(3, int(win))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(values.astype(np.float64), kernel, mode="same")


def _segments_from_projection(proj: np.ndarray, threshold: float, min_len: int, merge_gap: int) -> list[tuple[int, int, float]]:
    active = proj >= threshold
    segments: list[tuple[int, int, float]] = []
    start = None
    for idx, flag in enumerate(active):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_len:
                segments.append((start, idx, float(proj[start:idx].sum())))
            start = None
    if start is not None and len(proj) - start >= min_len:
        segments.append((start, len(proj), float(proj[start:].sum())))

    if not segments:
        return []

    merged = [segments[0]]
    for s, e, score in segments[1:]:
        ps, pe, pscore = merged[-1]
        if s - pe <= merge_gap:
            merged[-1] = (ps, e, pscore + score)
        else:
            merged.append((s, e, score))
    return merged


def refine_changed_mask(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask
    h, w = mask.shape
    row_proj = _smooth_1d(mask.mean(axis=1), 3)
    row_thr = max(float(np.percentile(row_proj, 70)), float(np.median(row_proj) + 1.0 * 1.4826 * np.median(np.abs(row_proj - np.median(row_proj)))), 0.015)
    row_segments = _segments_from_projection(row_proj, row_thr, min_len=2, merge_gap=2)
    if not row_segments:
        return mask
    row_segments.sort(key=lambda seg: seg[2] * (1.0 + 0.15 * ((seg[0] + seg[1]) * 0.5 / max(1, h))), reverse=True)
    rs, re, _ = row_segments[0]

    col_proj = _smooth_1d(mask[rs:re, :].mean(axis=0), 5)
    col_thr = max(float(np.percentile(col_proj, 68)), float(np.median(col_proj) + 0.8 * 1.4826 * np.median(np.abs(col_proj - np.median(col_proj)))), 0.01)
    col_segments = _segments_from_projection(col_proj, col_thr, min_len=max(4, w // 20), merge_gap=max(3, w // 35))
    if not col_segments:
        refined = np.zeros_like(mask)
        refined[rs:re, :] = mask[rs:re, :]
        return refined

    scored_cols = []
    for cs, ce, score in col_segments:
        width_ratio = (ce - cs) / max(1, w)
        center = (cs + ce) * 0.5
        center_bias = 1.0 - 0.25 * min(1.0, abs(center - w * 0.5) / max(1.0, w * 0.5))
        scored_cols.append((score * max(width_ratio, 0.05) * center_bias, cs, ce, score))
    scored_cols.sort(reverse=True)
    _, cs, ce, base_score = scored_cols[0]
    for _, s, e, score in scored_cols[1:]:
        gap = min(abs(s - ce), abs(cs - e))
        if gap <= max(4, w // 25) and score >= base_score * 0.3:
            cs = min(cs, s)
            ce = max(ce, e)

    pad_y = 1
    pad_x = max(2, w // 80)
    rs = max(0, rs - pad_y)
    re = min(h, re + pad_y)
    cs = max(0, cs - pad_x)
    ce = min(w, ce + pad_x)
    refined = np.zeros_like(mask)
    refined[rs:re, cs:ce] = mask[rs:re, cs:ce]
    return refined if np.any(refined) else mask


def estimate_text_box(
    video: Path,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> tuple[int, int, int, int, dict]:
    fallback_y0, fallback_y1 = estimate_subtitle_band(video, height, width, args)
    fallback_x0, fallback_x1 = 0, width

    search_y0 = int(height * args.search_y0_ratio)
    search_y1 = int(height * args.search_y1_ratio)
    search_y0 = max(0, min(search_y0, height - args.block))
    search_y1 = max(search_y0 + args.block, min(height, search_y1))
    search_h = search_y1 - search_y0

    text_accum = np.zeros((search_h, width), dtype=np.float32)
    change_accum = np.zeros((search_h, width), dtype=np.float32)
    prev_roi = None
    sampled = 0
    representative_idx = 0
    representative_score = -1.0

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    frame_idx = 0
    stride = max(1, int(args.refine_sample_stride))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[search_y0:search_y1, :]
            textish = char_like_pixel_mask(roi, args.text_edge_thr, args.bright_thr).astype(np.float32)
            text_accum += textish
            if prev_roi is not None:
                diff = (cv2.absdiff(roi, prev_roi) > args.diff_thr).astype(np.float32)
                change_accum += diff * textish
            score = float(textish.mean())
            if score > representative_score:
                representative_score = score
                representative_idx = frame_idx
            prev_roi = roi
            sampled += 1
        frame_idx += 1
    cap.release()

    if sampled == 0:
        return fallback_x0, fallback_x1, fallback_y0, fallback_y1, {"mode": "fallback_empty"}

    text_heat = text_accum / sampled
    change_heat = change_accum / max(1, sampled - 1)
    combined = text_heat + 0.35 * change_heat
    row_proj = combined.mean(axis=1)
    row_proj = _smooth_1d(row_proj, max(7, args.block * 3))
    med = float(np.median(row_proj))
    mad = float(np.median(np.abs(row_proj - med)))
    row_thr = max(float(np.percentile(row_proj, 82)), med + 1.8 * 1.4826 * mad)
    row_segments = _segments_from_projection(
        row_proj,
        row_thr,
        max(4, args.refine_min_height // 3),
        merge_gap=max(4, args.refine_pad_y // 2),
    )

    if not row_segments:
        return fallback_x0, fallback_x1, fallback_y0, fallback_y1, {"mode": "fallback_no_rows", "sampled": sampled}

    scored_rows = []
    for s, e, score in row_segments:
        y0 = search_y0 + s
        y1 = search_y0 + e
        h = y1 - y0
        if h > args.refine_max_height:
            continue
        row_slice = combined[s:e, :]
        col_proj = _smooth_1d(row_slice.mean(axis=0), max(7, args.block * 3))
        col_thr = max(float(np.percentile(col_proj, 78)), float(np.median(col_proj) + 1.2 * 1.4826 * np.median(np.abs(col_proj - np.median(col_proj)))))
        col_segments = _segments_from_projection(col_proj, col_thr, max(8, int(width * 0.05)), merge_gap=max(8, args.refine_pad_x))
        if col_segments:
            good_cols = [(cs, ce, cscore) for cs, ce, cscore in col_segments if (ce - cs) >= width * 0.08]
            if good_cols:
                scored_cols = []
                for cs, ce, cscore in good_cols:
                    center = (cs + ce) * 0.5
                    center_bias = 1.0 - 0.35 * min(1.0, abs(center - width * 0.5) / max(1.0, width * 0.5))
                    scored_cols.append((cscore * center_bias, cs, ce, cscore))
                scored_cols.sort(reverse=True)
                _, x0, x1, base_score = scored_cols[0]
                for _, cs, ce, cscore in scored_cols[1:]:
                    gap = min(abs(cs - x1), abs(x0 - ce))
                    if gap <= args.refine_pad_x * 2 and cscore >= base_score * 0.35:
                        x0 = min(x0, cs)
                        x1 = max(x1, ce)
            else:
                x0, x1 = fallback_x0, fallback_x1
        else:
            x0, x1 = fallback_x0, fallback_x1
        width_ratio = (x1 - x0) / max(1, width)
        if width_ratio < args.refine_min_width_ratio:
            continue
        bottom_bias = 0.65 + 0.35 * ((y0 + y1) * 0.5 / height)
        height_penalty = 1.0 if args.refine_min_height <= h <= args.refine_max_height else 0.7
        scored_rows.append((score * width_ratio * bottom_bias * height_penalty, s, e, x0, x1))

    if not scored_rows:
        return fallback_x0, fallback_x1, fallback_y0, fallback_y1, {"mode": "fallback_no_valid_rows", "sampled": sampled}

    scored_rows.sort(reverse=True)
    _, best_s, best_e, best_x0, best_x1 = scored_rows[0]

    # Pull in nearby strong text rows so two-line subtitles stay in one box.
    keep_s, keep_e = best_s, best_e
    best_score = scored_rows[0][0]
    for score, s, e, _, _ in scored_rows[1:]:
        if score < best_score * 0.42:
            continue
        vertical_gap = min(abs(s - keep_e), abs(keep_s - e))
        if vertical_gap <= max(24, args.refine_pad_y * 2):
            keep_s = min(keep_s, s)
            keep_e = max(keep_e, e)

    y0 = search_y0 + keep_s - args.refine_pad_y
    y1 = search_y0 + keep_e + args.refine_pad_y
    if y1 - y0 < args.refine_min_height:
        extra = (args.refine_min_height - (y1 - y0)) // 2 + 1
        y0 -= extra
        y1 += extra
    x0 = best_x0 - args.refine_pad_x
    x1 = best_x1 + args.refine_pad_x

    x0 = max(0, (x0 // args.block) * args.block)
    x1 = min(width, math.ceil(x1 / args.block) * args.block)
    y0 = max(0, (y0 // args.block) * args.block)
    y1 = min(height, math.ceil(y1 / args.block) * args.block)

    diagnostics = {
        "mode": "refined_text_box",
        "sampled": sampled,
        "search_y0": search_y0,
        "search_y1": search_y1,
        "row_threshold": row_thr,
        "representative_idx": representative_idx,
        "row_segments": [
            {"y0": int(search_y0 + s), "y1": int(search_y0 + e), "score": float(score)}
            for s, e, score in row_segments
        ],
    }
    return x0, x1, y0, y1, diagnostics


def resolve_x_band(width: int, args: argparse.Namespace) -> tuple[int, int]:
    x0 = 0 if args.x0 is None else args.x0
    x1 = width if args.x1 is None else args.x1
    x0 = max(0, min(width - args.block, x0))
    x1 = max(x0 + args.block, min(width, x1))
    x0 = (x0 // args.block) * args.block
    x1 = (x1 // args.block) * args.block
    return x0, x1


def read_roi_blocks(
    video: Path,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    args: argparse.Namespace,
    frame_infos: list[FrameInfo],
) -> tuple[list[np.ndarray], list[np.ndarray], list[FrameMetric]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    blocks: list[np.ndarray] = []
    text_masks: list[np.ndarray] = []
    metrics: list[FrameMetric] = []
    prev_block = None
    prev_text = None
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[y0:y1, x0:x1]
        bmean = block_mean(roi, args.block)
        tmask = text_block_mask(roi, args.block, args.text_edge_thr, args.bright_thr)
        text_density = float(tmask.mean())

        if prev_block is None:
            diff_density = 0.0
            diff_text_density = 0.0
            diff_mean = 0.0
        else:
            diff = np.abs(bmean - prev_block)
            diff_blocks = diff > args.diff_thr
            text_union = tmask | prev_text
            diff_density = float(diff_blocks.mean())
            diff_text_density = float((diff_blocks & text_union).mean())
            diff_mean = float(diff.mean())

        info = frame_infos[idx] if idx < len(frame_infos) else FrameInfo(idx, float(idx), "?", 0, 0)
        blocks.append(bmean)
        text_masks.append(tmask)
        metrics.append(
            FrameMetric(
                idx=idx,
                time=info.time,
                pict_type=info.pict_type,
                text_density=text_density,
                diff4_density_prev=diff_density,
                diff4_text_density_prev=diff_text_density,
                diff4_mean_prev=diff_mean,
                pkt_size=info.pkt_size,
            ),
        )
        prev_block = bmean
        prev_text = tmask
        idx += 1

    cap.release()
    return blocks, text_masks, metrics


def robust_threshold(values: Iterable[float], k: float, floor: float) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return floor, 0.0, 0.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = 1.4826 * mad
    return max(floor, med + k * sigma), med, sigma


def pkt_size_zscores(frame_infos: list[FrameInfo]) -> dict[int, float]:
    by_type: dict[str, list[int]] = {}
    for info in frame_infos:
        by_type.setdefault(info.pict_type, []).append(info.pkt_size)
    stats = {}
    for typ, sizes in by_type.items():
        arr = np.asarray(sizes, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        stats[typ] = (med, max(1.0, 1.4826 * mad))
    out = {}
    for info in frame_infos:
        med, sigma = stats.get(info.pict_type, (0.0, 1.0))
        out[info.idx] = float((info.pkt_size - med) / sigma)
    return out


def anchor_score_for_p(
    idx: int,
    prev_ref: int,
    blocks: list[np.ndarray],
    text_masks: list[np.ndarray],
    diff_thr: float,
) -> tuple[float, np.ndarray]:
    diff = np.abs(blocks[idx] - blocks[prev_ref])
    text_union = text_masks[idx] | text_masks[prev_ref]
    changed = refine_changed_mask((diff > diff_thr) & text_union)
    return float(changed.mean()), changed


def classify_kind(text_delta: float, density_eps: float = 0.0025) -> str:
    if text_delta > density_eps:
        return "onset"
    if text_delta < -density_eps:
        return "offset"
    return "replace"


def select_events(
    frame_infos: list[FrameInfo],
    blocks: list[np.ndarray],
    text_masks: list[np.ndarray],
    args: argparse.Namespace,
) -> list[dict]:
    non_b_indices = [info.idx for info in frame_infos if info.pict_type in {"I", "P"}]
    prev_ref_by_idx: dict[int, int] = {}
    prev = None
    for idx in non_b_indices:
        if prev is not None:
            prev_ref_by_idx[idx] = prev
        prev = idx

    raw_p_scores = []
    raw_anchor_data = {}
    for info in frame_infos:
        if info.pict_type != "P" or info.idx not in prev_ref_by_idx:
            continue
        prev_ref = prev_ref_by_idx[info.idx]
        score, changed = anchor_score_for_p(info.idx, prev_ref, blocks, text_masks, args.diff_thr)
        raw_p_scores.append(score)
        raw_anchor_data[info.idx] = (score, prev_ref, changed)

    threshold, med, sigma = robust_threshold(raw_p_scores, args.anchor_mad_k, args.anchor_min_density)
    candidates = []

    if args.include_initial_i and frame_infos and frame_infos[0].pict_type == "I":
        initial_density = float(text_masks[0].mean())
        if initial_density >= args.initial_text_density_min:
            candidates.append(
                {
                    "anchor_idx": 0,
                    "prev_ref": 0,
                    "candidate_idx": 0,
                    "score": initial_density,
                    "changed_mask": text_masks[0],
                    "kind": "initial",
                    "text_delta": initial_density,
                    "dist_old": 0.0,
                    "dist_new": 0.0,
                    "margin": 0.0,
                    "post_static": float(np.mean(np.abs(blocks[1][text_masks[0]] - blocks[0][text_masks[0]]))) if len(blocks) > 1 and np.any(text_masks[0]) else 0.0,
                    "threshold": args.initial_text_density_min,
                    "threshold_median": 0.0,
                    "threshold_sigma": 0.0,
                },
            )

    for anchor_idx, (score, prev_ref, changed_mask) in raw_anchor_data.items():
        if score < threshold:
            continue
        if not np.any(changed_mask):
            continue

        old_block = blocks[prev_ref]
        new_block = blocks[anchor_idx]
        candidate_idx = anchor_idx
        candidate_dist_old = 0.0
        candidate_dist_new = 0.0
        candidate_margin = 0.0

        for j in range(prev_ref + 1, anchor_idx + 1):
            dist_old = float(np.mean(np.abs(blocks[j][changed_mask] - old_block[changed_mask])))
            dist_new = float(np.mean(np.abs(blocks[j][changed_mask] - new_block[changed_mask])))
            margin = dist_old - dist_new
            if margin >= args.new_like_margin or dist_new < dist_old * 0.82:
                candidate_idx = j
                candidate_dist_old = dist_old
                candidate_dist_new = dist_new
                candidate_margin = margin
                break

        if candidate_idx == anchor_idx:
            candidate_dist_old = float(np.mean(np.abs(blocks[anchor_idx][changed_mask] - old_block[changed_mask])))
            candidate_dist_new = 0.0
            candidate_margin = candidate_dist_old

        if candidate_idx + 1 < len(blocks):
            post_static = float(np.mean(np.abs(blocks[candidate_idx + 1][changed_mask] - blocks[candidate_idx][changed_mask])))
        else:
            post_static = 0.0

        text_delta = float(text_masks[anchor_idx].mean() - text_masks[prev_ref].mean())
        candidates.append(
            {
                "anchor_idx": anchor_idx,
                "prev_ref": prev_ref,
                "candidate_idx": candidate_idx,
                "score": score,
                "changed_mask": changed_mask,
                "kind": classify_kind(text_delta),
                "text_delta": text_delta,
                "dist_old": candidate_dist_old,
                "dist_new": candidate_dist_new,
                "margin": candidate_margin,
                "post_static": post_static,
                "threshold": threshold,
                "threshold_median": med,
                "threshold_sigma": sigma,
            },
        )

    if args.include_i_anchors:
        raw_i_scores = []
        raw_i_data = {}
        for info in frame_infos:
            if info.pict_type != "I" or info.idx <= 0:
                continue
            prev_ref = info.idx - 1
            score, changed = anchor_score_for_p(info.idx, prev_ref, blocks, text_masks, args.diff_thr)
            raw_i_scores.append(score)
            raw_i_data[info.idx] = (score, prev_ref, changed)

        i_threshold, i_med, i_sigma = robust_threshold(raw_i_scores, args.anchor_mad_k, args.i_anchor_min_density)
        for anchor_idx, (score, prev_ref, changed_mask) in raw_i_data.items():
            if score < i_threshold or not np.any(changed_mask):
                continue
            dist_old = float(np.mean(np.abs(blocks[anchor_idx][changed_mask] - blocks[prev_ref][changed_mask])))
            post_static = float(np.mean(np.abs(blocks[anchor_idx + 1][changed_mask] - blocks[anchor_idx][changed_mask]))) if anchor_idx + 1 < len(blocks) else 0.0
            text_delta = float(text_masks[anchor_idx].mean() - text_masks[prev_ref].mean())
            candidates.append(
                {
                    "anchor_idx": anchor_idx,
                    "prev_ref": prev_ref,
                    "candidate_idx": anchor_idx,
                    "score": score,
                    "changed_mask": changed_mask,
                    "kind": classify_kind(text_delta),
                    "text_delta": text_delta,
                    "dist_old": dist_old,
                    "dist_new": 0.0,
                    "margin": dist_old,
                    "post_static": post_static,
                    "threshold": i_threshold,
                    "threshold_median": i_med,
                    "threshold_sigma": i_sigma,
                },
            )

    candidates.sort(key=lambda item: (item["candidate_idx"], -item["score"]))
    kept = []
    for item in candidates:
        if kept and item["candidate_idx"] - kept[-1]["candidate_idx"] < args.min_gap:
            if item["score"] > kept[-1]["score"]:
                kept[-1] = item
            continue
        kept.append(item)

    kept.sort(key=lambda item: item["score"], reverse=True)
    kept = kept[: args.max_events]
    kept.sort(key=lambda item: item["candidate_idx"])
    return kept


def read_frame(video: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {idx}")
    return frame


def overlay_change_mask(
    frame: np.ndarray,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    mask4: np.ndarray,
    block: int,
) -> np.ndarray:
    out = frame.copy()
    h_roi = y1 - y0
    w_roi = x1 - x0
    mask = cv2.resize(mask4.astype(np.uint8) * 255, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(out[y0:y1, x0:x1])
    color[:, :, 2] = 255
    color[:, :, 1] = 50
    roi = out[y0:y1, x0:x1]
    alpha = (mask.astype(np.float32) / 255.0) * 0.35
    roi[:] = (roi.astype(np.float32) * (1.0 - alpha[:, :, None]) + color.astype(np.float32) * alpha[:, :, None]).astype(np.uint8)
    out[y0:y1, x0:x1] = roi

    ys, xs = np.where(mask4)
    if len(xs) > 0:
        bx0 = int(x0 + xs.min() * block)
        bx1 = int(min(frame.shape[1] - 1, x0 + (xs.max() + 1) * block))
        my0 = int(y0 + ys.min() * block)
        my1 = int(min(frame.shape[0] - 1, y0 + (ys.max() + 1) * block))
        cv2.rectangle(out, (bx0, my0), (bx1, my1), (0, 0, 255), 2)
    return out


def label_frame(
    frame: np.ndarray,
    label: str,
    info: FrameInfo,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    active: bool = False,
) -> np.ndarray:
    out = frame.copy()
    color = (0, 0, 255) if active else (0, 220, 255)
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1), color, 2)
    header_h = 62
    cv2.rectangle(out, (0, 0), (out.shape[1], header_h), (0, 0, 0), -1)
    text1 = f"{label}  frame={info.idx}  type={info.pict_type}  t={info.time:.3f}s"
    text2 = f"pkt={info.pkt_size}"
    cv2.putText(out, text1, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text2, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    return out


def write_bbox_debug(video: Path, out_path: Path, frame_idx: int, x0: int, x1: int, y0: int, y1: int) -> None:
    frame = read_frame(video, frame_idx)
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1), (0, 0, 255), 3)
    label = f"refined text bbox: x[{x0}:{x1}] y[{y0}:{y1}] frame={frame_idx}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), frame)


def make_triptych(
    video: Path,
    out_path: Path,
    frame_infos: list[FrameInfo],
    event: dict,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    args: argparse.Namespace,
) -> None:
    cand = int(event["candidate_idx"])
    indices = [max(0, cand - 1), cand, min(len(frame_infos) - 1, cand + 1)]
    labels = ["pre", "candidate", "next"]
    panels = []
    for idx, label in zip(indices, labels, strict=True):
        frame = read_frame(video, idx)
        if idx == cand:
            frame = overlay_change_mask(frame, x0, x1, y0, y1, event["changed_mask"], args.block)
        panels.append(label_frame(frame, label, frame_infos[idx], x0, x1, y0, y1, active=(idx == cand)))
    triptych = np.concatenate(panels, axis=1)
    footer_h = 64
    footer = np.zeros((footer_h, triptych.shape[1], 3), dtype=np.uint8)
    summary = (
        f"event={event['event_id']} kind={event['kind']} "
        f"P_anchor={event['anchor_idx']} lag={event['anchor_idx'] - cand} "
        f"changed4={event['score']:.4f} old-new-margin={event['margin']:.2f} "
        f"post-static={event['post_static']:.2f}"
    )
    cv2.putText(footer, summary, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        footer,
        "Red overlay = changed 4x4 blocks in subtitle band between previous reference and P anchor.",
        (14, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    triptych = np.concatenate([triptych, footer], axis=0)
    cv2.imwrite(str(out_path), triptych)


def make_overview(triptych_paths: list[Path], out_path: Path, max_width: int = 2400) -> None:
    thumbs = []
    for path in triptych_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        scale = min(1.0, 720.0 / img.shape[1])
        thumb = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)
    if not thumbs:
        return

    rows = []
    current = []
    current_w = 0
    pad = 8
    for thumb in thumbs:
        if current and current_w + thumb.shape[1] + pad > max_width:
            rows.append(current)
            current = []
            current_w = 0
        current.append(thumb)
        current_w += thumb.shape[1] + pad
    if current:
        rows.append(current)

    row_imgs = []
    for row in rows:
        h = max(img.shape[0] for img in row)
        w = sum(img.shape[1] for img in row) + pad * (len(row) - 1)
        canvas = np.full((h, w, 3), 18, dtype=np.uint8)
        x = 0
        for img in row:
            canvas[: img.shape[0], x : x + img.shape[1]] = img
            x += img.shape[1] + pad
        row_imgs.append(canvas)

    total_h = sum(row.shape[0] for row in row_imgs) + pad * (len(row_imgs) - 1)
    total_w = max(row.shape[1] for row in row_imgs)
    canvas = np.full((total_h, total_w, 3), 18, dtype=np.uint8)
    y = 0
    for row in row_imgs:
        canvas[y : y + row.shape[0], : row.shape[1]] = row
        y += row.shape[0] + pad
    cv2.imwrite(str(out_path), canvas)


def render_codecview_triptych(video: Path, out_dir: Path, event: Event, fps: float) -> str:
    if fps <= 0:
        fps = 30.0
    start = max(0.0, (event.candidate_idx - 1) / fps)
    out_path = out_dir / f"event_{event.event_id:03d}_frame_{event.candidate_idx:06d}_codecview.jpg"
    vf = (
        "codecview=mv=pf+bf+bb:block=1,"
        "scale=360:-1,"
        "drawtext=text='%{n}':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6,"
        "tile=3x1"
    )
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-flags2",
        "+export_mvs",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(video),
        "-frames:v",
        "3",
        "-vf",
        vf,
        "-y",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        return ""
    return str(out_path)


def write_csv(path: Path, events: list[Event]) -> None:
    if not events:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(events[0]).keys()))
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def write_metrics_csv(path: Path, metrics: list[FrameMetric]) -> None:
    if not metrics:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        for metric in metrics:
            writer.writerow(asdict(metric))


def main() -> None:
    args = parse_args()
    video = Path(args.video).resolve()
    out_dir = Path(args.out_dir).resolve()
    triptych_dir = out_dir / "triptychs"
    codecview_dir = out_dir / "codecview"
    triptych_dir.mkdir(parents=True, exist_ok=True)
    codecview_dir.mkdir(parents=True, exist_ok=True)

    frame_infos = ffprobe_frames(video)
    width, height, fps, cap_frames = get_video_shape(video)
    if args.no_refine_box:
        x0, x1 = resolve_x_band(width, args)
        y0, y1 = estimate_subtitle_band(video, height, width, args)
        bbox_diagnostics = {"mode": "manual_or_band"}
    else:
        refined_x0, refined_x1, refined_y0, refined_y1, bbox_diagnostics = estimate_text_box(video, width, height, args)
        manual_x = args.x0 is not None and args.x1 is not None
        manual_y = args.y0 is not None and args.y1 is not None
        if manual_x:
            x0, x1 = resolve_x_band(width, args)
        else:
            x0, x1 = refined_x0, refined_x1
        if manual_y:
            y0, y1 = estimate_subtitle_band(video, height, width, args)
        else:
            y0, y1 = refined_y0, refined_y1
    print(f"video={video}")
    print(f"shape={width}x{height} fps={fps:.3f} ffprobe_frames={len(frame_infos)} cv_frames={cap_frames}")
    print(f"subtitle_box=x[{x0}:{x1}] y[{y0}:{y1}] size={x1-x0}x{y1-y0}")
    if bbox_diagnostics.get("representative_idx") is not None:
        write_bbox_debug(video, out_dir / "bbox_debug.jpg", int(bbox_diagnostics["representative_idx"]), x0, x1, y0, y1)

    blocks, text_masks, metrics = read_roi_blocks(video, x0, x1, y0, y1, args, frame_infos)
    if len(blocks) < len(frame_infos):
        frame_infos = frame_infos[: len(blocks)]
    elif len(frame_infos) < len(blocks):
        for idx in range(len(frame_infos), len(blocks)):
            frame_infos.append(FrameInfo(idx, idx / fps if fps else float(idx), "?", 0, 0))

    pkt_z = pkt_size_zscores(frame_infos)
    raw_events = select_events(frame_infos, blocks, text_masks, args)

    event_rows: list[Event] = []
    triptych_paths: list[Path] = []
    for event_id, raw in enumerate(raw_events, start=1):
        raw["event_id"] = event_id
        cand_idx = int(raw["candidate_idx"])
        anchor_idx = int(raw["anchor_idx"])
        prev_ref = int(raw["prev_ref"])
        triptych_path = triptych_dir / f"event_{event_id:03d}_frame_{cand_idx:06d}_anchor_{anchor_idx:06d}.jpg"
        make_triptych(video, triptych_path, frame_infos, raw, x0, x1, y0, y1, args)
        triptych_paths.append(triptych_path)

        row = Event(
            event_id=event_id,
            candidate_idx=cand_idx,
            candidate_time=frame_infos[cand_idx].time,
            candidate_type=frame_infos[cand_idx].pict_type,
            anchor_idx=anchor_idx,
            anchor_time=frame_infos[anchor_idx].time,
            anchor_type=frame_infos[anchor_idx].pict_type,
            prev_ref_idx=prev_ref,
            prev_ref_time=frame_infos[prev_ref].time,
            prev_ref_type=frame_infos[prev_ref].pict_type,
            anchor_lag_frames=anchor_idx - cand_idx,
            kind=str(raw["kind"]),
            score=float(raw["score"]),
            changed4_density=float(raw["score"]),
            text_density_prev_ref=float(text_masks[prev_ref].mean()),
            text_density_anchor=float(text_masks[anchor_idx].mean()),
            text_density_delta=float(raw["text_delta"]),
            dist_to_old=float(raw["dist_old"]),
            dist_to_new=float(raw["dist_new"]),
            old_new_margin=float(raw["margin"]),
            post_static_mean=float(raw["post_static"]),
            p_pkt_size=frame_infos[anchor_idx].pkt_size,
            p_pkt_size_z=float(pkt_z.get(anchor_idx, 0.0)),
            triptych_path=str(triptych_path),
        )
        if args.codecview:
            render_codecview_triptych(video, codecview_dir, row, fps)
        event_rows.append(row)

    make_overview(triptych_paths, out_dir / "overview_triptychs.jpg")
    write_csv(out_dir / "events.csv", event_rows)
    write_metrics_csv(out_dir / "frame_metrics.csv", metrics)
    summary = {
        "video": str(video),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": len(frame_infos),
        "subtitle_box": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
        "bbox_diagnostics": bbox_diagnostics,
        "args": vars(args),
        "events": [asdict(row) for row in event_rows],
    }
    (out_dir / "events.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"events={len(event_rows)}")
    print(f"events_csv={out_dir / 'events.csv'}")
    print(f"events_json={out_dir / 'events.json'}")
    print(f"triptychs={triptych_dir}")
    print(f"overview={out_dir / 'overview_triptychs.jpg'}")


if __name__ == "__main__":
    main()
