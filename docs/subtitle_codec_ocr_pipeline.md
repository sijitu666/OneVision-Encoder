# 基于编码信息的字幕 OCR 实验记录

本文档记录围绕 `xinwen.mp4` 和 `duanju.mp4` 实现的字幕起止帧候选检测、OCR 前重复帧过滤、以及 patch/宏块可视化工具。

需要先明确一个关键边界：

- `subtitle_event_codec_probe.py` 和 `subtitle_ocr_dedup_pipeline.py` 是 codec-inspired 规则：它们利用帧类型、packet size、P/B/I 上下文、解码后亮度块和文字形态 proxy 来生成候选和去重。
- `visualize_h264_mbtype_intra.py` 是当前真正的 H.264 编码侧可视化：它解析 FFmpeg H.264 decoder 的 `mb_type` debug 输出，不使用图像边缘 proxy。

## 目标

实验假设是：字幕出现、消失、换句时，字幕区域经常会产生小块 intra 编码的突增。这个突增可能落在 P 帧上，但观众实际看到的字幕边界可能位于附近 B 帧。因此完整流程应当：

1. 先找出字幕区域剧烈变化的粗 anchor。
2. 再围绕 anchor 检索附近 B/P/I 帧，避免直接把 P 帧当作可见起止点。
3. 产出高召回候选帧列表，交给 OCR 做后续语义筛选。
4. 在 OCR 前过滤字幕稳定区间的重复帧。
5. 生成可视化，方便逐帧核查规则是否符合原始编码现象。

## 已实现工具

### 1. 字幕事件候选检测

脚本：

```bash
tools/subtitle_event_codec_probe.py
```

核心思路：

- 用 `ffprobe` 读取每帧 metadata：frame id、显示时间、`pict_type`、packet size、keyframe。
- 使用显式 `--x0 --x1 --y0 --y1` 或自动 y 轴搜索来确定字幕 ROI。
- 在字幕 ROI 内构建 4x4 亮度块和文字形态 proxy。
- 把字幕区变化突出的 P/I 帧视为 anchor。
- 在 anchor 附近比较“更像旧 reference”还是“更像新 anchor”。
- 选择第一个更接近新状态的附近帧作为字幕可见边界候选。

典型用法：

```bash
/opt/anaconda3/bin/python tools/subtitle_event_codec_probe.py \
  --video duanju.mp4 \
  --out-dir outputs/subtitle_codec_events/duanju_ocrbox_codec \
  --x0 84 --x1 632 --y0 807 --y1 899 \
  --max-events 120
```

主要输出：

- `events.csv`：候选帧、anchor 帧、帧类型、score、变化密度、时间戳。
- `events.json`：候选帧和运行参数，后续 dedup 会读取它。
- `triptychs/`：`pre + candidate + next` 三联图。
- `overview_triptychs.jpg`：候选三联图总览。

本轮实验结果：

- `outputs/subtitle_codec_events/xinwen_subtitlebox/events.json`：2499 帧，56 个候选，ROI 为 `x[40:680] y[860:1020]`。
- `outputs/subtitle_codec_events/duanju_ocrbox_codec/events.json`：2790 帧，87 个候选，ROI 为 `x[84:632] y[807:899]`。

这个列表定位为高召回初筛，不负责最终语义判定，后面仍应接 OCR。

### 2. OCR 前重复帧过滤

脚本：

```bash
tools/subtitle_ocr_dedup_pipeline.py
```

核心思路：

- 读取 `events.json` 中的候选帧。
- 保护 event candidate 以及可配置的 `candidate +/- N` 上下文帧。
- 在字幕 ROI 内把当前帧和当前稳定段代表帧做比较。
- 支持三种比较模式：
  - `exact`：解码 ROI hash 完全一致才去重。
  - `text`：文字形态 4x4 mask 稳定则去重。
  - `hybrid`：文字 mask 和局部亮度都稳定才去重。
- 默认保留 I 帧，因为 I 帧对重置状态和 OCR 校验有价值。
- 输出给 OCR 使用的 frame id 列表。

典型用法：

```bash
/opt/anaconda3/bin/python tools/subtitle_ocr_dedup_pipeline.py \
  --video duanju.mp4 \
  --out-dir outputs/subtitle_ocr_pipeline/duanju_textmask_strict \
  --event-json outputs/subtitle_codec_events/duanju_ocrbox_codec/events.json \
  --scope subtitle \
  --compare-mode text \
  --mask-iou-thr 0.90 \
  --mask-delta-thr 0.035 \
  --text-density-delta-thr 0.035 \
  --event-context 1
```

主要输出：

- `dedup_frames.csv`：每帧 keep/drop 结果和原因。
- `ocr_frame_ids.txt`：送给 OCR 的 frame id。
- `kept_ocr_frames_contact.jpg`：保留帧可视化。
- `duplicate_groups_contact.jpg`：重复组可视化。
- `summary.json`：保留比例和原因统计。

`duanju.mp4` 已观察到的效果：

- `hybrid`：保留 2186 / 2790 帧，保留比例 0.784。
- `text`：保留 1542 / 2790 帧，保留比例 0.553。
- `text` strict：保留 973 / 2790 帧，保留比例 0.349。
- `exact full`：保留 2790 / 2790 帧，说明解码后完全重复帧在这个视频里几乎没有，直接 exact hash 对 OCR 加速帮助不大。

当前更实用的是 `text` 或 strict `text`，它对高帧率字幕 OCR 的降帧更明显。

### 3. 密集 4x4 patch 候选可视化

脚本：

```bash
tools/visualize_dense_4x4_patches.py
```

状态：

- 这是解码图像 proxy，不是原始 codec CU/MB 数据。
- 它可以辅助找文字密集区域、调 bbox、看字幕 ROI 是否覆盖合理。
- 它不能证明图中每个高亮 4x4 patch 都来自原始码流。

典型用法：

```bash
/opt/anaconda3/bin/python tools/visualize_dense_4x4_patches.py \
  --video duanju.mp4 \
  --out-dir outputs/dense_4x4_patch_overlays/duanju \
  --event-json outputs/subtitle_codec_events/duanju_ocrbox_codec/events.json
```

主要输出：

- `selected_patch_frames.csv`：每帧 proxy 密度和 cluster 信息。
- `patch_overlay_contact.jpg`：可视化总览。
- `overlays/`：被选中帧的 overlay。
- `summary.json`。

注意：

这个脚本会画 bbox 和 cluster 辅助框。它适合调试，不适合作为“原始全画面编码 patch 可视化”的最终版本。

### 4. 全画面 4x4 proxy overlay

脚本：

```bash
tools/visualize_fullframe_4x4_patches.py
```

状态：

- 这也是解码图像 proxy。
- 它不画蓝框、绿框、文字 label，只把全画面 text-like 4x4 proxy block 涂出来。
- 因为依据的是图像边缘和亮度，静态字幕会在每帧持续高亮。这是 proxy 的预期行为，不是原始编码证据。

典型用法：

```bash
/opt/anaconda3/bin/python tools/visualize_fullframe_4x4_patches.py \
  --video xinwen.mp4 \
  --out-dir outputs/fullframe_4x4_patch_overlays/xinwen \
  --save-every 1 \
  --write-video
```

主要输出：

- `frames/frame_*.jpg`
- `patch_overlay.mp4`
- `patch_metrics.csv`

这个工具只用于观察解码图像中文字形态密度，不应用它判断原始码流里的 patch 是否每帧存在。

### 5. H.264 原始宏块类型可视化

脚本：

```bash
tools/visualize_h264_mbtype_intra.py
```

这是当前用于核查“原始编码里哪些帧真的有 intra4x4/intra16x16 宏块”的正确可视化。

核心思路：

- 调用 FFmpeg H.264 decoder：

```bash
ffmpeg -hide_banner -nostats -loglevel debug -threads 1 -debug mb_type -i <video> -f null -
```

- 解析 `New frame, type: I/P/B` 和 H.264 macroblock type 网格。
- 只绘制选中的原始 decoder 符号。
- 默认绘制 `i,I`：
  - `i`：intra4x4-coded 16x16 macroblock。
  - `I`：intra16x16-coded 16x16 macroblock。

典型用法：

```bash
/opt/anaconda3/bin/python tools/visualize_h264_mbtype_intra.py \
  --video duanju.mp4 \
  --out-dir outputs/h264_mbtype_intra_overlays/duanju \
  --symbols i,I \
  --save-every 1 \
  --write-video
```

主要输出：

- `frames/frame_*.jpg`：每帧原图上覆盖真实 `i/I` 宏块。
- `mbtype_intra_overlay.mp4`：完整 overlay 视频。
- `mbtype_metrics.csv`：每帧 `i/I` 宏块数量和密度。
- `summary.json`。

本轮实验结果：

- `outputs/h264_mbtype_intra_overlays/xinwen/summary.json`：2499 帧，解析到 2499 张 mb-type map，平均 `i/I` 宏块数 63.85。
- `outputs/h264_mbtype_intra_overlays/duanju/summary.json`：2790 帧，解析到 2790 张 mb-type map，平均 `i/I` 宏块数 312.38。
- 两个视频中，ffprobe `pict_type` 和 FFmpeg debug frame type 的 mismatch 均为 0。
- `xinwen` 有 1384 帧没有任何 `i/I` 覆盖，`duanju` 有 276 帧没有任何 `i/I` 覆盖，说明这版不会再把静态字幕每帧都错误高亮。

当前限制：

stock FFmpeg 暴露的是 H.264 16x16 macroblock type。`i` 表示这个 16x16 宏块内部采用 intra4x4 模式，但 stock FFmpeg 不会暴露宏块内部具体哪几个 4x4 sub-block，所以当前只能涂满整个 16x16 MB。

如果要拿到严格的 HEVC CU/TU 4x4 map，或者 H.264 内部 4x4 sub-block 级别 map，需要 patched decoder 或仓库里的自定义 codec feature decoder。当前仓库里的 `dataloader/decoder/bin/hevc` 是 Linux ELF，在这台 macOS 工作区不能直接运行。

## 推荐工作流

高召回字幕 OCR 初筛可以按下面流程跑。

1. 用较准的字幕 ROI 做事件候选：

```bash
/opt/anaconda3/bin/python tools/subtitle_event_codec_probe.py \
  --video duanju.mp4 \
  --out-dir outputs/subtitle_codec_events/duanju_ocrbox_codec \
  --x0 84 --x1 632 --y0 807 --y1 899
```

2. 用字幕 ROI 内文字 mask 做 OCR 前去重：

```bash
/opt/anaconda3/bin/python tools/subtitle_ocr_dedup_pipeline.py \
  --video duanju.mp4 \
  --out-dir outputs/subtitle_ocr_pipeline/duanju_textmask_strict \
  --event-json outputs/subtitle_codec_events/duanju_ocrbox_codec/events.json \
  --scope subtitle \
  --compare-mode text \
  --mask-iou-thr 0.90 \
  --mask-delta-thr 0.035 \
  --text-density-delta-thr 0.035 \
  --event-context 1
```

3. 将 `ocr_frame_ids.txt` 交给 OCR。

4. 用 H.264 mb-type overlay 核查候选附近是否有真实 intra 宏块突增：

```bash
/opt/anaconda3/bin/python tools/visualize_h264_mbtype_intra.py \
  --video duanju.mp4 \
  --out-dir outputs/h264_mbtype_intra_overlays/duanju \
  --symbols i,I \
  --save-every 1 \
  --write-video
```

## 解释和后续方向

- P 帧 intra 突增适合做 anchor，但不能直接等同于字幕可见起止帧。
- 附近 B 帧仍需按显示顺序参与判断。
- 当前事件检测器用旧状态/新状态相似度近似这个回溯过程。
- 精确的 B 帧 RefList/MV 依赖分析还没有完全实现，stock FFmpeg/ffprobe 不够用，需要 decoder 侧导出 MV 和 reference-list。
- 对 OCR 加速而言，完全相同帧过滤过于保守；字幕 ROI 内 text mask 稳定性过滤更有效。
- 判断“原始编码里是否真的有 intra4x4/intra16x16 宏块”时，应优先看 `visualize_h264_mbtype_intra.py` 的输出，而不是 full-frame 4x4 proxy 输出。

