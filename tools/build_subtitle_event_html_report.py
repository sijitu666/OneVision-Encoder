#!/usr/bin/env python3
"""Build an HTML report for subtitle event candidates.

The report is a static page that references generated JPEG assets. It is meant
to make frame-level subtitle onset/offset candidates easier to audit than a
single full-frame triptych.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from subtitle_event_codec_probe import ffprobe_frames, get_video_shape


@dataclass
class FramePanel:
    idx: int
    frame: np.ndarray
    pict_type: str
    time: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an HTML audit report for subtitle event candidates.")
    parser.add_argument("--event-json", required=True, help="events.json from subtitle_event_codec_probe.py.")
    parser.add_argument("--video", default=None, help="Input video. Defaults to the path stored in events.json.")
    parser.add_argument("--out-dir", default=None, help="Report directory. Defaults to <event-dir>/html_report.")
    parser.add_argument("--codec-overlay-frames", default=None, help="Optional H.264 mb_type overlay frames directory.")
    parser.add_argument("--window", type=int, default=2, help="Show candidate +/- window frames.")
    parser.add_argument("--context-pad-y", type=int, default=80, help="Extra vertical pixels around the subtitle ROI.")
    parser.add_argument("--full-height", type=int, default=420, help="Panel height for full-frame strips.")
    parser.add_argument("--crop-width", type=int, default=520, help="Panel width for subtitle crop strips.")
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--max-events", type=int, default=None)
    return parser.parse_args()


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def read_video_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {idx}")
    return frame


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / max(1, image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    scale = width / max(1, image.shape[1])
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def draw_label(image: np.ndarray, title: str, subtitle: str, selected: bool = False) -> np.ndarray:
    out = image.copy()
    bar_h = 44
    color = (30, 30, 220) if selected else (30, 30, 30)
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), color, -1)
    cv2.putText(out, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, subtitle, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    border = (30, 30, 230) if selected else (120, 120, 120)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), border, 3 if selected else 1)
    return out


def hstack_panels(panels: list[np.ndarray], pad: int = 8, bg: int = 18) -> np.ndarray:
    if not panels:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    height = max(panel.shape[0] for panel in panels)
    width = sum(panel.shape[1] for panel in panels) + pad * (len(panels) - 1)
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    x = 0
    for panel in panels:
        canvas[: panel.shape[0], x : x + panel.shape[1]] = panel
        x += panel.shape[1] + pad
    return canvas


def make_full_strip(
    panels: list[FramePanel],
    candidate_idx: int,
    subtitle_box: tuple[int, int, int, int],
    full_height: int,
) -> np.ndarray:
    x0, x1, y0, y1 = subtitle_box
    out_panels: list[np.ndarray] = []
    for panel in panels:
        frame = panel.frame.copy()
        selected = panel.idx == candidate_idx
        roi_color = (0, 0, 230) if selected else (0, 215, 255)
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), roi_color, 3 if selected else 2)
        resized = resize_to_height(frame, full_height)
        scale = full_height / max(1, frame.shape[0])
        rx0, ry0 = int(round(x0 * scale)), int(round(y0 * scale))
        rx1, ry1 = int(round(x1 * scale)), int(round(y1 * scale))
        cv2.rectangle(resized, (rx0, ry0), (rx1 - 1, ry1 - 1), roi_color, 2)
        title = f"{'candidate' if selected else 'frame'} {panel.idx}"
        subtitle = f"{panel.pict_type}  t={panel.time:.3f}s"
        out_panels.append(draw_label(resized, title, subtitle, selected=selected))
    return hstack_panels(out_panels)


def crop_with_context(
    frame: np.ndarray,
    subtitle_box: tuple[int, int, int, int],
    context_box: tuple[int, int, int, int],
) -> np.ndarray:
    x0, x1, y0, y1 = context_box
    crop = frame[y0:y1, x0:x1].copy()
    sx0, sx1, sy0, sy1 = subtitle_box
    cv2.rectangle(crop, (sx0 - x0, sy0 - y0), (sx1 - x0 - 1, sy1 - y0 - 1), (0, 215, 255), 2)
    return crop


def make_crop_strip(
    panels: list[FramePanel],
    candidate_idx: int,
    subtitle_box: tuple[int, int, int, int],
    context_box: tuple[int, int, int, int],
    crop_width: int,
) -> np.ndarray:
    out_panels: list[np.ndarray] = []
    for panel in panels:
        selected = panel.idx == candidate_idx
        crop = crop_with_context(panel.frame, subtitle_box, context_box)
        resized = resize_to_width(crop, crop_width)
        title = f"{'candidate' if selected else 'frame'} {panel.idx}"
        subtitle = f"{panel.pict_type}  t={panel.time:.3f}s"
        out_panels.append(draw_label(resized, title, subtitle, selected=selected))
    return hstack_panels(out_panels)


def make_diff_panel(
    frames_by_idx: dict[int, np.ndarray],
    event: dict[str, Any],
    context_box: tuple[int, int, int, int],
    crop_width: int,
) -> np.ndarray:
    comparisons = [
        ("prev -> cand", safe_int(event["candidate_idx"]) - 1, safe_int(event["candidate_idx"])),
        ("cand -> next", safe_int(event["candidate_idx"]), safe_int(event["candidate_idx"]) + 1),
        ("prev_ref -> anchor", safe_int(event["prev_ref_idx"]), safe_int(event["anchor_idx"])),
    ]
    x0, x1, y0, y1 = context_box
    out_panels: list[np.ndarray] = []
    for title, a_idx, b_idx in comparisons:
        if a_idx not in frames_by_idx or b_idx not in frames_by_idx:
            continue
        a = cv2.cvtColor(frames_by_idx[a_idx][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(frames_by_idx[b_idx][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(a, b)
        heat = cv2.applyColorMap(np.clip(diff * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        resized = resize_to_width(heat, crop_width)
        subtitle = f"f{a_idx} to f{b_idx}  mean={float(diff.mean()):.2f}"
        out_panels.append(draw_label(resized, title, subtitle, selected=False))
    return hstack_panels(out_panels)


def read_codec_overlay(codec_dir: Path, idx: int) -> np.ndarray | None:
    path = codec_dir / f"frame_{idx:06d}.jpg"
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image


def make_codec_strip(
    frame_ids: list[int],
    candidate_idx: int,
    codec_dir: Path,
    subtitle_box: tuple[int, int, int, int],
    context_box: tuple[int, int, int, int],
    crop_width: int,
) -> np.ndarray | None:
    out_panels: list[np.ndarray] = []
    for idx in frame_ids:
        image = read_codec_overlay(codec_dir, idx)
        if image is None:
            continue
        selected = idx == candidate_idx
        crop = crop_with_context(image, subtitle_box, context_box)
        resized = resize_to_width(crop, crop_width)
        title = f"{'candidate' if selected else 'frame'} {idx}"
        out_panels.append(draw_label(resized, title, "H.264 i/I mb_type overlay", selected=selected))
    if not out_panels:
        return None
    return hstack_panels(out_panels)


def infer_codec_overlay_dir(video: Path, event_dir: Path) -> Path | None:
    candidate = event_dir.parents[1] / "h264_mbtype_intra_overlays" / video.stem / "frames"
    if candidate.exists():
        return candidate
    candidate = Path("outputs") / "h264_mbtype_intra_overlays" / video.stem / "frames"
    if candidate.exists():
        return candidate
    return None


def event_command(data: dict[str, Any]) -> str:
    args = data.get("args", {})
    video = args.get("video") or data.get("video") or "<video>"
    out_dir = args.get("out_dir") or "<out-dir>"
    parts = [
        "/opt/anaconda3/bin/python tools/subtitle_event_codec_probe.py",
        f"  --video {video}",
        f"  --out-dir {out_dir}",
    ]
    for key in ("x0", "x1", "y0", "y1"):
        if args.get(key) is not None:
            parts.append(f"  --{key} {args[key]}")
    if args.get("max_events") is not None:
        parts.append(f"  --max-events {args['max_events']}")
    return " \\\n".join(parts)


def css_text() -> str:
    return """
:root {
  color-scheme: dark;
  --bg: #101215;
  --panel: #171a1f;
  --panel2: #20242b;
  --text: #e9edf2;
  --muted: #9ca7b5;
  --line: #333a44;
  --accent: #e64b3c;
  --good: #59b36a;
  --warn: #e6b14a;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--text);
}
header {
  position: sticky;
  top: 0;
  z-index: 3;
  background: rgba(16, 18, 21, 0.96);
  border-bottom: 1px solid var(--line);
  padding: 18px 24px 14px;
}
h1 { margin: 0 0 8px; font-size: 22px; }
h2 { margin: 28px 0 12px; font-size: 17px; }
a { color: #8cc8ff; text-decoration: none; }
code, pre {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: #0c0e11;
  border: 1px solid var(--line);
  border-radius: 6px;
}
pre { padding: 12px; overflow-x: auto; }
.summary {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 10px;
}
.pill {
  background: var(--panel2);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 6px 10px;
  color: var(--muted);
  font-size: 13px;
}
.controls {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 12px;
}
.controls input, .controls select {
  background: var(--panel2);
  color: var(--text);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px 10px;
}
main { padding: 22px 24px 48px; }
.timeline {
  position: relative;
  height: 42px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: linear-gradient(90deg, #161a20, #1c222b);
  margin: 16px 0 22px;
}
.marker {
  position: absolute;
  top: 5px;
  width: 6px;
  height: 30px;
  border-radius: 4px;
  transform: translateX(-3px);
  background: var(--accent);
}
.marker.offset { background: var(--warn); }
.marker.onset { background: var(--good); }
.event {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
  margin: 16px 0;
}
.event.hidden { display: none; }
.event-title {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
}
.event-title h3 {
  margin: 0;
  font-size: 18px;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 8px;
  margin: 10px 0 14px;
}
.metric {
  background: #111419;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 8px;
}
.metric b { display: block; font-size: 13px; color: var(--muted); font-weight: 500; margin-bottom: 3px; }
.metric span { font-size: 15px; }
.images { display: grid; gap: 12px; }
.image-block {
  background: #0b0d10;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
}
.image-block strong {
  display: block;
  margin: 0 0 8px;
  color: var(--muted);
  font-size: 13px;
}
.image-block img {
  display: block;
  max-width: 100%;
  height: auto;
  border-radius: 4px;
}
details {
  margin-top: 10px;
}
summary {
  cursor: pointer;
  color: var(--muted);
  user-select: none;
}
.muted { color: var(--muted); }
"""


def js_text() -> str:
    return """
const filterInput = document.getElementById('filter');
const kindSelect = document.getElementById('kind');
const events = Array.from(document.querySelectorAll('.event'));
function applyFilter() {
  const q = filterInput.value.trim().toLowerCase();
  const kind = kindSelect.value;
  for (const card of events) {
    const text = card.dataset.search;
    const kindOk = kind === 'all' || card.dataset.kind === kind;
    const textOk = !q || text.includes(q);
    card.classList.toggle('hidden', !(kindOk && textOk));
  }
}
filterInput.addEventListener('input', applyFilter);
kindSelect.addEventListener('change', applyFilter);
document.addEventListener('keydown', (event) => {
  if (event.key !== 'j' && event.key !== 'k') return;
  const visible = events.filter(card => !card.classList.contains('hidden'));
  const current = visible.findIndex(card => card.getBoundingClientRect().top > 8);
  let next = current;
  if (event.key === 'j') next = Math.min(visible.length - 1, Math.max(0, current));
  if (event.key === 'k') next = Math.max(0, current - 2);
  visible[next]?.scrollIntoView({behavior: 'smooth', block: 'start'});
});
"""


def relpath(path: Path, base: Path) -> str:
    return html.escape(os.path.relpath(path, base).replace("\\", "/"))


def metric_html(label: str, value: Any) -> str:
    return f"<div class=\"metric\"><b>{html.escape(label)}</b><span>{html.escape(str(value))}</span></div>"


def build_html(
    data: dict[str, Any],
    events: list[dict[str, Any]],
    out_dir: Path,
    report_dir: Path,
    asset_rows: list[dict[str, str]],
    video: Path,
    subtitle_box: tuple[int, int, int, int],
    codec_overlay_dir: Path | None,
) -> str:
    frame_count = safe_int(data.get("frame_count"))
    x0, x1, y0, y1 = subtitle_box
    markers = []
    for event in events:
        idx = safe_int(event["candidate_idx"])
        pct = 100.0 * idx / max(1, frame_count - 1)
        kind = html.escape(str(event.get("kind", "unknown")))
        markers.append(
            f"<a class=\"marker {kind}\" href=\"#event-{event['event_id']:03d}\" "
            f"style=\"left:{pct:.3f}%\" title=\"event {event['event_id']} f{idx} {kind}\"></a>"
        )

    cards = []
    for event, assets in zip(events, asset_rows):
        event_id = safe_int(event["event_id"])
        kind = str(event.get("kind", "unknown"))
        candidate = safe_int(event["candidate_idx"])
        anchor = safe_int(event["anchor_idx"])
        lag = safe_int(event.get("anchor_lag_frames"))
        search = " ".join(
            [
                str(event_id),
                kind,
                str(candidate),
                str(anchor),
                str(event.get("candidate_type", "")),
                str(event.get("anchor_type", "")),
            ],
        ).lower()
        original_triptych = Path(str(event.get("triptych_path", "")))
        triptych_link = ""
        if original_triptych.exists():
            triptych_link = (
                f"<a href=\"{relpath(original_triptych, report_dir)}\">"
                "原始 triptych</a>"
            )
        blocks = [
            ("字幕区域放大窗口", assets.get("crop")),
            ("前后帧差分：定位哪一帧发生变化", assets.get("diff")),
            ("全画面上下文", assets.get("full")),
        ]
        if assets.get("codec"):
            blocks.append(("H.264 原始 i/I 宏块 overlay 对照", assets.get("codec")))

        image_html = []
        for title, src in blocks:
            if not src:
                continue
            image_html.append(
                "<div class=\"image-block\">"
                f"<strong>{html.escape(title)}</strong>"
                f"<img loading=\"lazy\" src=\"{src}\" alt=\"event {event_id} {html.escape(title)}\">"
                "</div>",
            )

        cards.append(
            f"""
<section class="event" id="event-{event_id:03d}" data-kind="{html.escape(kind)}" data-search="{html.escape(search)}">
  <div class="event-title">
    <h3>#{event_id:03d} {html.escape(kind)} · candidate f{candidate} ({html.escape(str(event.get('candidate_type', '?')))})</h3>
    <div class="muted">anchor f{anchor} ({html.escape(str(event.get('anchor_type', '?')))}) · lag {lag} frames · {triptych_link}</div>
  </div>
  <div class="metrics">
    {metric_html('candidate time', f"{safe_float(event.get('candidate_time')):.3f}s")}
    {metric_html('prev ref', f"f{safe_int(event.get('prev_ref_idx'))} {event.get('prev_ref_type', '?')}")}
    {metric_html('changed4 density', f"{safe_float(event.get('changed4_density')):.4f}")}
    {metric_html('old-new margin', f"{safe_float(event.get('old_new_margin')):.2f}")}
    {metric_html('P pkt z-score', f"{safe_float(event.get('p_pkt_size_z')):.2f}")}
    {metric_html('post static mean', f"{safe_float(event.get('post_static_mean')):.2f}")}
  </div>
  <div class="images">
    {''.join(image_html)}
  </div>
</section>
""",
        )

    command = html.escape(event_command(data))
    codec_note = (
        f"<span class=\"pill\">codec overlay: {html.escape(str(codec_overlay_dir))}</span>"
        if codec_overlay_dir
        else "<span class=\"pill\">codec overlay: not found</span>"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Subtitle Event Report - {html.escape(video.name)}</title>
  <style>{css_text()}</style>
</head>
<body>
  <header>
    <h1>字幕切换帧可视化报告 · {html.escape(video.name)}</h1>
    <div class="summary">
      <span class="pill">events: {len(events)}</span>
      <span class="pill">frames: {frame_count}</span>
      <span class="pill">ROI: x[{x0}:{x1}] y[{y0}:{y1}]</span>
      {codec_note}
    </div>
    <div class="controls">
      <input id="filter" type="search" placeholder="搜索 event id / frame id / type">
      <select id="kind">
        <option value="all">全部类型</option>
        <option value="onset">onset</option>
        <option value="offset">offset</option>
        <option value="change">change</option>
      </select>
    </div>
  </header>
  <main>
    <h2>这个目录是怎么得到的</h2>
    <pre>{command}</pre>
    <p class="muted">算法先用 P/I 帧字幕 ROI 内的 4x4 变化做 anchor，再围绕 anchor 回看附近 B/P/I 帧，选择更接近新字幕状态的第一帧作为 candidate。下面每个卡片都把 candidate 放在中心。</p>
    <h2>候选时间轴</h2>
    <div class="timeline">{''.join(markers)}</div>
    {''.join(cards)}
  </main>
  <script>{js_text()}</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    event_json = Path(args.event_json).resolve()
    event_dir = event_json.parent
    data = json.loads(event_json.read_text(encoding="utf-8"))
    video = Path(args.video or data.get("video") or data.get("args", {}).get("video")).resolve()
    report_dir = Path(args.out_dir).resolve() if args.out_dir else event_dir / "html_report"
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    width, height, _, _ = get_video_shape(video)
    box = data.get("subtitle_box") or {}
    subtitle_box = (
        clamp(safe_int(box.get("x0", 0)), 0, width - 1),
        clamp(safe_int(box.get("x1", width)), 1, width),
        clamp(safe_int(box.get("y0", int(height * 0.62))), 0, height - 1),
        clamp(safe_int(box.get("y1", int(height * 0.86))), 1, height),
    )
    x0, x1, y0, y1 = subtitle_box
    context_box = (
        x0,
        x1,
        clamp(y0 - args.context_pad_y, 0, height - 1),
        clamp(y1 + args.context_pad_y, 1, height),
    )

    events = list(data.get("events", []))
    if args.max_events is not None:
        events = events[: args.max_events]

    frame_infos = ffprobe_frames(video)
    codec_overlay_dir = Path(args.codec_overlay_frames).resolve() if args.codec_overlay_frames else infer_codec_overlay_dir(video, event_dir)
    if codec_overlay_dir is not None and not codec_overlay_dir.exists():
        codec_overlay_dir = None

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    frame_cache: dict[int, np.ndarray] = {}

    def get_frame(idx: int) -> np.ndarray:
        idx = clamp(idx, 0, len(frame_infos) - 1)
        if idx not in frame_cache:
            frame_cache[idx] = read_video_frame(cap, idx)
        return frame_cache[idx]

    asset_rows: list[dict[str, str]] = []
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
    for event in events:
        candidate = safe_int(event["candidate_idx"])
        frame_ids = list(range(max(0, candidate - args.window), min(len(frame_infos), candidate + args.window + 1)))
        needed_ids = set(frame_ids)
        needed_ids.update(
            [
                candidate - 1,
                candidate,
                candidate + 1,
                safe_int(event.get("prev_ref_idx")),
                safe_int(event.get("anchor_idx")),
            ],
        )
        frames_by_idx = {idx: get_frame(idx) for idx in needed_ids if 0 <= idx < len(frame_infos)}
        panels = [
            FramePanel(
                idx=idx,
                frame=frames_by_idx[idx],
                pict_type=frame_infos[idx].pict_type if idx < len(frame_infos) else "?",
                time=frame_infos[idx].time if idx < len(frame_infos) else 0.0,
            )
            for idx in frame_ids
        ]

        event_id = safe_int(event["event_id"])
        prefix = f"event_{event_id:03d}_frame_{candidate:06d}"
        full_path = assets_dir / f"{prefix}_full.jpg"
        crop_path = assets_dir / f"{prefix}_crop.jpg"
        diff_path = assets_dir / f"{prefix}_diff.jpg"
        codec_path = assets_dir / f"{prefix}_codec.jpg"

        cv2.imwrite(str(full_path), make_full_strip(panels, candidate, subtitle_box, args.full_height), params)
        cv2.imwrite(str(crop_path), make_crop_strip(panels, candidate, subtitle_box, context_box, args.crop_width), params)
        cv2.imwrite(str(diff_path), make_diff_panel(frames_by_idx, event, context_box, args.crop_width), params)
        row = {
            "full": relpath(full_path, report_dir),
            "crop": relpath(crop_path, report_dir),
            "diff": relpath(diff_path, report_dir),
        }
        if codec_overlay_dir is not None:
            codec = make_codec_strip(frame_ids, candidate, codec_overlay_dir, subtitle_box, context_box, args.crop_width)
            if codec is not None:
                cv2.imwrite(str(codec_path), codec, params)
                row["codec"] = relpath(codec_path, report_dir)
        asset_rows.append(row)

    cap.release()

    html_text = build_html(data, events, event_dir, report_dir, asset_rows, video, subtitle_box, codec_overlay_dir)
    report_path = report_dir / "index.html"
    report_path.write_text(html_text, encoding="utf-8")
    (report_dir / "report_manifest.json").write_text(
        json.dumps(
            {
                "event_json": str(event_json),
                "video": str(video),
                "event_count": len(events),
                "subtitle_box": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
                "context_box": {
                    "x0": context_box[0],
                    "x1": context_box[1],
                    "y0": context_box[2],
                    "y1": context_box[3],
                },
                "codec_overlay_frames": str(codec_overlay_dir) if codec_overlay_dir else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"report={report_path}")
    print(f"assets={assets_dir}")
    print(f"events={len(events)}")


if __name__ == "__main__":
    main()
