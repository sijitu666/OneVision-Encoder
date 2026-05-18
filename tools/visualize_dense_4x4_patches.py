#!/usr/bin/env python3
"""Visualize frames that contain dense 4x4 subtitle-like patches.

Stock ffprobe does not expose per-frame H.264 4x4 CU maps, so this script uses
the same compression-domain proxy as subtitle_event_codec_probe.py: text-like
4x4 luma patches inside the subtitle bbox. It scans every frame, records patch
density and cluster stats, then overlays the detected 4x4 patches on selected
frames for visual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from subtitle_event_codec_probe import ffprobe_frames, get_video_shape, read_frame, text_block_mask


@dataclass
class PatchMetric:
    idx: int
    time: float
    pict_type: str
    pkt_size: int
    text_density: float
    patch_count: int
    max_cluster_patch_count: int
    max_cluster_area_blocks: int
    max_cluster_width_px: int
    max_cluster_height_px: int
    cluster_x0: int
    cluster_y0: int
    cluster_x1: int
    cluster_y1: int
    selected: int
    saved: int
    image_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay dense subtitle 4x4 patches on candidate frames.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--event-json", default=None, help="Use subtitle_box and default thresholds from an event JSON.")
    parser.add_argument("--x0", type=int, default=None)
    parser.add_argument("--x1", type=int, default=None)
    parser.add_argument("--y0", type=int, default=None)
    parser.add_argument("--y1", type=int, default=None)
    parser.add_argument("--block", type=int, default=4)
    parser.add_argument("--text-edge-thr", type=float, default=None)
    parser.add_argument("--bright-thr", type=int, default=None)
    parser.add_argument("--density-min", type=float, default=0.030)
    parser.add_argument("--cluster-patches-min", type=int, default=36)
    parser.add_argument("--cluster-width-min", type=int, default=96)
    parser.add_argument("--cluster-height-max", type=int, default=180)
    parser.add_argument("--close-kernel-x", type=int, default=11, help="Morphology close kernel width in 4x4 blocks.")
    parser.add_argument("--close-kernel-y", type=int, default=3, help="Morphology close kernel height in 4x4 blocks.")
    parser.add_argument("--save-top-k", type=int, default=80, help="Always save the top-K densest selected frames.")
    parser.add_argument("--save-run-reps", type=int, default=3, help="Save first/middle/last representatives from each selected run.")
    parser.add_argument("--max-save", type=int, default=220)
    parser.add_argument("--max-contact", type=int, default=120)
    parser.add_argument("--panel-width", type=int, default=360)
    return parser.parse_args()


def load_event_defaults(event_json: Path | None) -> tuple[dict, dict]:
    if event_json is None:
        return {}, {}
    data = json.loads(event_json.read_text(encoding="utf-8"))
    return data.get("subtitle_box", {}), data.get("args", {})


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


def dense_cluster_stats(mask: np.ndarray, block: int, close_x: int, close_y: int) -> tuple[int, int, int, int, int, int, int, int]:
    if mask.size == 0 or int(mask.sum()) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0
    kernel = np.ones((max(1, close_y), max(1, close_x)), dtype=np.uint8)
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    best = (0, 0, 0, 0, 0, 0, 0, 0)
    for label in range(1, nlabels):
        x, y, w, h, area = stats[label]
        component = labels == label
        patch_count = int(mask[component].sum())
        if patch_count > best[0]:
            best = (
                patch_count,
                int(area),
                int(w * block),
                int(h * block),
                int(x * block),
                int(y * block),
                int((x + w) * block),
                int((y + h) * block),
            )
    return best


def is_selected(metric: PatchMetric, args: argparse.Namespace) -> bool:
    return (
        metric.text_density >= args.density_min
        and metric.max_cluster_patch_count >= args.cluster_patches_min
        and metric.max_cluster_width_px >= args.cluster_width_min
        and metric.max_cluster_height_px <= args.cluster_height_max
    )


def selected_save_ids(metrics: list[PatchMetric], args: argparse.Namespace) -> set[int]:
    selected = [m for m in metrics if m.selected]
    save_ids = {m.idx for m in sorted(selected, key=lambda m: m.text_density, reverse=True)[: args.save_top_k]}
    run: list[PatchMetric] = []
    for metric in metrics + [None]:  # type: ignore[list-item]
        if metric is not None and metric.selected:
            run.append(metric)
            continue
        if run:
            positions = [0]
            if args.save_run_reps >= 2 and len(run) > 2:
                positions.append(len(run) // 2)
            if args.save_run_reps >= 3 and len(run) > 1:
                positions.append(len(run) - 1)
            for pos in sorted(set(positions)):
                save_ids.add(run[pos].idx)
            run = []
    if len(save_ids) > args.max_save:
        top = [m.idx for m in sorted(selected, key=lambda m: m.text_density, reverse=True)[: args.max_save]]
        save_ids = set(top)
    return save_ids


def overlay_patch_mask(frame: np.ndarray, box: tuple[int, int, int, int], mask: np.ndarray, metric: PatchMetric, block: int) -> np.ndarray:
    x0, x1, y0, y1 = box
    out = frame.copy()
    overlay = out.copy()
    ys, xs = np.where(mask)
    for by, bx in zip(ys, xs):
        px0 = x0 + int(bx) * block
        py0 = y0 + int(by) * block
        cv2.rectangle(overlay, (px0, py0), (px0 + block - 1, py0 + block - 1), (0, 230, 255), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (255, 160, 0), 2)
    if metric.max_cluster_patch_count > 0:
        cx0 = x0 + metric.cluster_x0
        cy0 = y0 + metric.cluster_y0
        cx1 = x0 + metric.cluster_x1
        cy1 = y0 + metric.cluster_y1
        cv2.rectangle(out, (cx0, cy0), (cx1 - 1, cy1 - 1), (40, 220, 40), 2)
    label = (
        f"f={metric.idx} {metric.pict_type} dens={metric.text_density:.3f} "
        f"patch={metric.patch_count} cluster={metric.max_cluster_patch_count}"
    )
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 760), 34), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def crop_panel(overlay: np.ndarray, box: tuple[int, int, int, int], metric: PatchMetric, width: int) -> np.ndarray:
    x0, x1, y0, y1 = box
    crop = overlay[y0:y1, x0:x1]
    scale = width / max(1, crop.shape[1])
    panel = cv2.resize(crop, (width, max(48, int(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 32), (0, 0, 0), -1)
    text = f"f={metric.idx} {metric.pict_type} d={metric.text_density:.3f} c={metric.max_cluster_patch_count}"
    cv2.putText(panel, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def make_grid(panels: list[np.ndarray], out_path: Path, cols: int = 4, pad: int = 8) -> None:
    if not panels:
        return
    rows = []
    for start in range(0, len(panels), cols):
        row = panels[start : start + cols]
        h = max(panel.shape[0] for panel in row)
        w = sum(panel.shape[1] for panel in row) + pad * (len(row) - 1)
        canvas = np.full((h, w, 3), 24, dtype=np.uint8)
        x = 0
        for panel in row:
            canvas[: panel.shape[0], x : x + panel.shape[1]] = panel
            x += panel.shape[1] + pad
        rows.append(canvas)
    h = sum(row.shape[0] for row in rows) + pad * (len(rows) - 1)
    w = max(row.shape[1] for row in rows)
    canvas = np.full((h, w, 3), 24, dtype=np.uint8)
    y = 0
    for row in rows:
        canvas[y : y + row.shape[0], : row.shape[1]] = row
        y += row.shape[0] + pad
    cv2.imwrite(str(out_path), canvas)


def write_csv(path: Path, metrics: list[PatchMetric]) -> None:
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
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    event_box, event_args = load_event_defaults(Path(args.event_json).resolve() if args.event_json else None)
    if args.text_edge_thr is None:
        args.text_edge_thr = float(event_args.get("text_edge_thr", 30.0))
    if args.bright_thr is None:
        args.bright_thr = int(event_args.get("bright_thr", 145))

    frame_infos = ffprobe_frames(video)
    width, height, fps, _ = get_video_shape(video)
    box = resolve_box(args, event_box, width, height)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    metrics: list[PatchMetric] = []
    masks: dict[int, np.ndarray] = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x0, x1, y0, y1 = box
        roi = gray[y0:y1, x0:x1]
        mask = text_block_mask(roi, args.block, float(args.text_edge_thr), int(args.bright_thr))
        patch_count = int(mask.sum())
        cluster = dense_cluster_stats(mask, args.block, args.close_kernel_x, args.close_kernel_y)
        info = frame_infos[idx] if idx < len(frame_infos) else None
        metric = PatchMetric(
            idx=idx,
            time=float(info.time if info else idx / max(1.0, fps)),
            pict_type=str(info.pict_type if info else "?"),
            pkt_size=int(info.pkt_size if info else 0),
            text_density=float(mask.mean()),
            patch_count=patch_count,
            max_cluster_patch_count=int(cluster[0]),
            max_cluster_area_blocks=int(cluster[1]),
            max_cluster_width_px=int(cluster[2]),
            max_cluster_height_px=int(cluster[3]),
            cluster_x0=int(cluster[4]),
            cluster_y0=int(cluster[5]),
            cluster_x1=int(cluster[6]),
            cluster_y1=int(cluster[7]),
            selected=0,
            saved=0,
            image_path="",
        )
        metric.selected = 1 if is_selected(metric, args) else 0
        metrics.append(metric)
        if metric.selected:
            masks[idx] = mask
        idx += 1
    cap.release()

    save_ids = selected_save_ids(metrics, args)
    contact_panels: list[np.ndarray] = []
    for metric in metrics:
        if metric.idx not in save_ids:
            continue
        frame = read_frame(video, metric.idx)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x0, x1, y0, y1 = box
        mask = masks.get(metric.idx)
        if mask is None:
            mask = text_block_mask(gray[y0:y1, x0:x1], args.block, float(args.text_edge_thr), int(args.bright_thr))
        overlay = overlay_patch_mask(frame, box, mask, metric, args.block)
        image_path = overlays_dir / f"patch_overlay_f{metric.idx:06d}.jpg"
        cv2.imwrite(str(image_path), overlay)
        metric.saved = 1
        metric.image_path = str(image_path)
        if len(contact_panels) < args.max_contact:
            contact_panels.append(crop_panel(overlay, box, metric, args.panel_width))

    write_csv(out_dir / "patch_metrics.csv", metrics)
    selected = [m for m in metrics if m.selected]
    write_csv(out_dir / "selected_patch_frames.csv", selected)
    make_grid(contact_panels, out_dir / "patch_overlay_contact.jpg", cols=4)

    summary = {
        "video": str(video),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": len(metrics),
        "subtitle_box": {"x0": box[0], "x1": box[1], "y0": box[2], "y1": box[3]},
        "block": args.block,
        "text_edge_thr": args.text_edge_thr,
        "bright_thr": args.bright_thr,
        "density_min": args.density_min,
        "cluster_patches_min": args.cluster_patches_min,
        "cluster_width_min": args.cluster_width_min,
        "cluster_height_max": args.cluster_height_max,
        "selected_count": len(selected),
        "saved_count": int(sum(m.saved for m in metrics)),
        "max_density": max((m.text_density for m in metrics), default=0.0),
        "max_cluster_patch_count": max((m.max_cluster_patch_count for m in metrics), default=0),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"video={video}")
    print(f"frames={len(metrics)} selected={len(selected)} saved={summary['saved_count']}")
    print(f"box=x[{box[0]}:{box[1]}] y[{box[2]}:{box[3]}] block={args.block}")
    print(f"thr=edge {args.text_edge_thr} bright {args.bright_thr} density {args.density_min}")
    print(f"csv={out_dir / 'selected_patch_frames.csv'}")
    print(f"contact={out_dir / 'patch_overlay_contact.jpg'}")
    print(f"overlays={overlays_dir}")


if __name__ == "__main__":
    main()
