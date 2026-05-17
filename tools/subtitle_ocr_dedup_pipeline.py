#!/usr/bin/env python3
"""Build an OCR frame list with codec-assisted subtitle-frame deduplication.

This script is meant to run after subtitle_event_codec_probe.py. It keeps the
P/B/I event candidates from that detector, then collapses stable runs inside the
subtitle bbox so OCR only sees representative frames plus boundary candidates.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from subtitle_event_codec_probe import (
    FrameInfo,
    block_mean,
    char_like_pixel_mask,
    ffprobe_frames,
    get_video_shape,
    pkt_size_zscores,
    read_frame,
    text_block_mask,
)


@dataclass
class DedupRow:
    idx: int
    time: float
    pict_type: str
    pkt_size: int
    pkt_size_z: float
    keep: int
    reason: str
    group_id: int
    representative_idx: int
    is_event_context: int
    is_event_candidate: int
    luma_diff_mean: float
    luma_diff_density: float
    text_mask_iou: float
    text_mask_delta: float
    text_density: float
    ref_text_density: float
    exact_hash_match: int
    codec_dup_hint: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codec-assisted duplicate-frame filter for subtitle OCR.")
    parser.add_argument("--video", required=True, help="Input video.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--event-json", default=None, help="events.json from subtitle_event_codec_probe.py.")
    parser.add_argument("--x0", type=int, default=None)
    parser.add_argument("--x1", type=int, default=None)
    parser.add_argument("--y0", type=int, default=None)
    parser.add_argument("--y1", type=int, default=None)
    parser.add_argument("--scope", choices=["subtitle", "roi", "full"], default="subtitle", help="Dedup signal scope.")
    parser.add_argument(
        "--compare-mode",
        choices=["hybrid", "text", "exact"],
        default="hybrid",
        help=(
            "hybrid compares text mask plus local luma, text compares subtitle text state only, "
            "exact only drops decoded-identical frames."
        ),
    )
    parser.add_argument("--block", type=int, default=4)
    parser.add_argument("--diff-thr", type=float, default=16.0, help="4x4 luma diff threshold.")
    parser.add_argument("--mean-thr", type=float, default=5.0, help="Mean luma diff threshold against current representative.")
    parser.add_argument("--density-thr", type=float, default=0.035, help="Changed 4x4 density threshold against current representative.")
    parser.add_argument("--mask-iou-thr", type=float, default=0.82, help="Text-mask IoU threshold for subtitle-scope duplicates.")
    parser.add_argument("--mask-delta-thr", type=float, default=0.055, help="Text-mask XOR density threshold.")
    parser.add_argument("--text-density-delta-thr", type=float, default=0.06, help="Text density change threshold.")
    parser.add_argument("--text-edge-thr", type=float, default=30.0)
    parser.add_argument("--bright-thr", type=int, default=145)
    parser.add_argument("--codec-dup-z-thr", type=float, default=-0.8, help="Packet-size z-score hint for cheap duplicate candidates.")
    parser.add_argument("--keep-i", action=argparse.BooleanOptionalAction, default=True, help="Always keep I frames.")
    parser.add_argument("--event-context", type=int, default=1, help="Keep candidate +/- N frames from event detector.")
    parser.add_argument("--min-run-len", type=int, default=2, help="Only call a run duplicate after at least this many frames.")
    parser.add_argument("--max-kept-vis", type=int, default=120)
    parser.add_argument("--max-groups-vis", type=int, default=36)
    parser.add_argument("--group-frames-vis", type=int, default=8)
    return parser.parse_args()


def load_event_context(event_json: Path | None, nframes: int, context: int) -> tuple[set[int], set[int], dict, list[dict]]:
    if event_json is None:
        return set(), set(), {}, []
    data = json.loads(event_json.read_text(encoding="utf-8"))
    events = data.get("events", [])
    candidates = {int(event["candidate_idx"]) for event in events if "candidate_idx" in event}
    protected: set[int] = set()
    for idx in candidates:
        for j in range(max(0, idx - context), min(nframes, idx + context + 1)):
            protected.add(j)
    return candidates, protected, data.get("subtitle_box", {}), events


def resolve_box(args: argparse.Namespace, event_box: dict, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = args.x0 if args.x0 is not None else event_box.get("x0", 0)
    x1 = args.x1 if args.x1 is not None else event_box.get("x1", width)
    y0 = args.y0 if args.y0 is not None else event_box.get("y0", int(height * 0.62))
    y1 = args.y1 if args.y1 is not None else event_box.get("y1", int(height * 0.86))
    x0 = max(0, min(width - args.block, int(x0)))
    x1 = max(x0 + args.block, min(width, int(x1)))
    y0 = max(0, min(height - args.block, int(y0)))
    y1 = max(y0 + args.block, min(height, int(y1)))
    x0 = (x0 // args.block) * args.block
    x1 = (x1 // args.block) * args.block
    y0 = (y0 // args.block) * args.block
    y1 = (y1 // args.block) * args.block
    return x0, x1, y0, y1


def crop_for_scope(gray: np.ndarray, box: tuple[int, int, int, int], scope: str) -> np.ndarray:
    if scope == "full":
        return gray
    x0, x1, y0, y1 = box
    return gray[y0:y1, x0:x1]


def exact_hash(roi: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(roi), digest_size=16).hexdigest()


def signature_for_roi(roi: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, str]:
    roi_hash = exact_hash(roi)
    if args.compare_mode == "exact":
        return np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=bool), 0.0, roi_hash

    luma_blocks = block_mean(roi, args.block)
    if args.scope == "subtitle":
        mask = text_block_mask(roi, args.block, args.text_edge_thr, args.bright_thr)
    else:
        # For full/roi mode, use char-like pixels only for reporting; luma blocks drive dedup.
        pix = char_like_pixel_mask(roi, args.text_edge_thr, args.bright_thr).astype(np.float32)
        h, w = roi.shape
        bh, bw = h // args.block, w // args.block
        mask = cv2.resize(pix[: bh * args.block, : bw * args.block], (bw, bh), interpolation=cv2.INTER_AREA) > 0.08
    return luma_blocks, mask, float(mask.mean()), roi_hash


def compare_to_ref(
    luma: np.ndarray,
    mask: np.ndarray,
    ref_luma: np.ndarray,
    ref_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, float, float, float]:
    if args.scope == "subtitle":
        union = mask | ref_mask
        inter = mask & ref_mask
        union_count = int(union.sum())
        text_iou = float(inter.sum() / union_count) if union_count else 1.0
        text_delta = float(np.mean(mask != ref_mask))
        if np.any(union):
            diff = np.abs(luma[union] - ref_luma[union])
            diff_mean = float(diff.mean())
            diff_density = float((diff > args.diff_thr).mean())
        else:
            diff_mean = 0.0
            diff_density = 0.0
        return diff_mean, diff_density, text_iou, text_delta

    diff = np.abs(luma - ref_luma)
    diff_mean = float(diff.mean())
    diff_density = float((diff > args.diff_thr).mean())
    union = mask | ref_mask
    inter = mask & ref_mask
    union_count = int(union.sum())
    text_iou = float(inter.sum() / union_count) if union_count else 1.0
    text_delta = float(np.mean(mask != ref_mask))
    return diff_mean, diff_density, text_iou, text_delta


def is_duplicate(
    diff_mean: float,
    diff_density: float,
    text_iou: float,
    text_delta: float,
    text_density: float,
    ref_text_density: float,
    exact_match: bool,
    args: argparse.Namespace,
) -> bool:
    if args.compare_mode == "exact":
        return bool(exact_match)

    luma_stable = diff_mean <= args.mean_thr and diff_density <= args.density_thr
    if args.scope == "subtitle":
        text_stable = (
            text_iou >= args.mask_iou_thr
            and text_delta <= args.mask_delta_thr
            and abs(text_density - ref_text_density) <= args.text_density_delta_thr
        )
        if args.compare_mode == "text":
            return bool(text_stable)
        return bool(luma_stable and text_stable)
    return bool(luma_stable)


def draw_crop_panel(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    keep: bool,
    width: int = 360,
) -> np.ndarray:
    x0, x1, y0, y1 = box
    crop = frame[y0:y1, x0:x1].copy()
    if crop.size == 0:
        crop = frame.copy()
    scale = width / max(1, crop.shape[1])
    out = cv2.resize(crop, (width, max(48, int(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    color = (30, 170, 30) if keep else (40, 40, 220)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), color, 3)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_grid(panels: list[np.ndarray], out_path: Path, cols: int = 4, pad: int = 8) -> None:
    if not panels:
        return
    rows = []
    for start in range(0, len(panels), cols):
        row = panels[start : start + cols]
        h = max(panel.shape[0] for panel in row)
        w = sum(panel.shape[1] for panel in row) + pad * (len(row) - 1)
        canvas = np.full((h, w, 3), 20, dtype=np.uint8)
        x = 0
        for panel in row:
            canvas[: panel.shape[0], x : x + panel.shape[1]] = panel
            x += panel.shape[1] + pad
        rows.append(canvas)
    h = sum(row.shape[0] for row in rows) + pad * (len(rows) - 1)
    w = max(row.shape[1] for row in rows)
    canvas = np.full((h, w, 3), 20, dtype=np.uint8)
    y = 0
    for row in rows:
        canvas[y : y + row.shape[0], : row.shape[1]] = row
        y += row.shape[0] + pad
    cv2.imwrite(str(out_path), canvas)


def write_csv(path: Path, rows: list[DedupRow]) -> None:
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
    out_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = out_dir / "duplicate_groups"
    groups_dir.mkdir(exist_ok=True)

    frame_infos = ffprobe_frames(video)
    width, height, fps, cap_frames = get_video_shape(video)
    event_json = Path(args.event_json).resolve() if args.event_json else None
    event_candidates, event_context, event_box, events = load_event_context(event_json, len(frame_infos), args.event_context)
    box = resolve_box(args, event_box, width, height)
    pkt_z = pkt_size_zscores(frame_infos)

    rows: list[DedupRow] = []
    groups: list[list[int]] = []
    group_id = -1
    ref_idx = None
    ref_luma = None
    ref_mask = None
    ref_hash = None
    ref_text_density = 0.0
    run_len = 0

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = crop_for_scope(gray, box, args.scope)
        luma, mask, text_density, roi_hash = signature_for_roi(roi, args)
        info = frame_infos[idx] if idx < len(frame_infos) else FrameInfo(idx, idx / max(1.0, fps), "?", 0, 0)
        codec_dup_hint = info.pict_type in {"B", "P"} and float(pkt_z.get(idx, 0.0)) <= args.codec_dup_z_thr

        event_candidate = idx in event_candidates
        protected = idx in event_context
        if ref_luma is None:
            keep = True
            reason = "first"
            diff_mean = diff_density = text_iou = text_delta = 0.0
            hash_match = False
        else:
            diff_mean, diff_density, text_iou, text_delta = compare_to_ref(luma, mask, ref_luma, ref_mask, args)
            hash_match = roi_hash == ref_hash
            duplicate = is_duplicate(diff_mean, diff_density, text_iou, text_delta, text_density, ref_text_density, hash_match, args)
            force_keep = protected or event_candidate or (args.keep_i and info.pict_type == "I")
            if force_keep:
                keep = True
                if event_candidate:
                    reason = "event_candidate"
                elif protected:
                    reason = "event_context"
                else:
                    reason = "i_frame"
            elif duplicate and run_len + 1 >= args.min_run_len:
                keep = False
                reason = "exact_duplicate" if args.compare_mode == "exact" else "duplicate"
            elif duplicate:
                keep = False
                reason = "warmup_exact_duplicate" if args.compare_mode == "exact" else "warmup_duplicate"
            else:
                keep = True
                reason = "changed"

        if keep:
            group_id += 1
            groups.append([idx])
            ref_idx = idx
            ref_luma = luma
            ref_mask = mask
            ref_hash = roi_hash
            ref_text_density = text_density
            run_len = 0
        else:
            groups[group_id].append(idx)
            run_len += 1

        rows.append(
            DedupRow(
                idx=idx,
                time=info.time,
                pict_type=info.pict_type,
                pkt_size=info.pkt_size,
                pkt_size_z=float(pkt_z.get(idx, 0.0)),
                keep=1 if keep else 0,
                reason=reason,
                group_id=group_id,
                representative_idx=int(ref_idx),
                is_event_context=1 if protected else 0,
                is_event_candidate=1 if event_candidate else 0,
                luma_diff_mean=float(diff_mean),
                luma_diff_density=float(diff_density),
                text_mask_iou=float(text_iou),
                text_mask_delta=float(text_delta),
                text_density=float(text_density),
                ref_text_density=float(ref_text_density),
                exact_hash_match=1 if hash_match else 0,
                codec_dup_hint=1 if codec_dup_hint else 0,
            ),
        )
        idx += 1

    cap.release()

    kept = [row for row in rows if row.keep]
    dropped = [row for row in rows if not row.keep]
    write_csv(out_dir / "dedup_frames.csv", rows)
    (out_dir / "ocr_frame_ids.txt").write_text("\n".join(str(row.idx) for row in kept) + "\n", encoding="utf-8")

    summary = {
        "video": str(video),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": len(rows),
        "subtitle_box": {"x0": box[0], "x1": box[1], "y0": box[2], "y1": box[3]},
        "scope": args.scope,
        "compare_mode": args.compare_mode,
        "event_json": str(event_json) if event_json else None,
        "event_count": len(events),
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "keep_ratio": len(kept) / max(1, len(rows)),
        "duplicate_groups": sum(1 for group in groups if len(group) > 1),
        "reason_counts": dict(Counter(row.reason for row in rows)),
        "kept_reason_counts": dict(Counter(row.reason for row in kept)),
        "dropped_reason_counts": dict(Counter(row.reason for row in dropped)),
        "codec_dup_hint_count": int(sum(row.codec_dup_hint for row in rows)),
        "exact_hash_match_count": int(sum(row.exact_hash_match for row in rows)),
        "args": vars(args),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    kept_panels = []
    for row in kept[: args.max_kept_vis]:
        frame = read_frame(video, row.idx)
        label = f"K {row.idx} {row.pict_type} {row.reason}"
        kept_panels.append(draw_crop_panel(frame, box, label, keep=True))
    make_grid(kept_panels, out_dir / "kept_ocr_frames_contact.jpg", cols=4)

    group_panels = []
    shown_groups = 0
    for group in groups:
        if len(group) <= 1:
            continue
        sample = group[: args.group_frames_vis]
        if group[-1] not in sample:
            sample.append(group[-1])
        for pos, frame_idx in enumerate(sample):
            frame = read_frame(video, frame_idx)
            keep = pos == 0
            tag = "REP" if keep else "DROP"
            label = f"G{shown_groups:03d} {tag} f={frame_idx}"
            group_panels.append(draw_crop_panel(frame, box, label, keep=keep, width=260))
        shown_groups += 1
        if shown_groups >= args.max_groups_vis:
            break
    make_grid(group_panels, out_dir / "duplicate_groups_contact.jpg", cols=max(2, args.group_frames_vis + 1))

    print(f"video={video}")
    print(f"shape={width}x{height} fps={fps:.3f} frames={len(rows)}")
    print(f"subtitle_box=x[{box[0]}:{box[1]}] y[{box[2]}:{box[3]}] scope={args.scope}")
    print(f"kept={len(kept)} dropped={len(dropped)} keep_ratio={summary['keep_ratio']:.3f}")
    print(f"dedup_csv={out_dir / 'dedup_frames.csv'}")
    print(f"ocr_frame_ids={out_dir / 'ocr_frame_ids.txt'}")
    print(f"kept_contact={out_dir / 'kept_ocr_frames_contact.jpg'}")
    print(f"duplicate_contact={out_dir / 'duplicate_groups_contact.jpg'}")


if __name__ == "__main__":
    main()
