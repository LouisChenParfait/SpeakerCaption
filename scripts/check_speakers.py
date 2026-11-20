#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分人诊断：为什么这一段音频没分成不同的人？

完全照抄实时听录链路的分人步骤（同样的 silero VAD 参数 → CAM++ 声纹 →
`OnlineSpeakerTracker.assign`），把每段的时长、标签、分数和段间余弦相似度
打出来。分不出人时先用它定位到底是「阈值太严」「段太短」还是「长段把
好几个人平均成了一个声纹」。

用法：
    python scripts/check_speakers.py <音频.wav>
    python scripts/check_speakers.py <音频.wav> --threshold 0.5 --num-speakers 2
    python scripts/check_speakers.py <音频.wav> --chunk 13   # 模拟长段场景
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speakercaption.diarize import build_vad  # noqa: E402
from speakercaption.live_speaker import (      # noqa: E402
    OnlineSpeakerTracker, build_extractor)

EMB_MODEL = ROOT / "models" / "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"


def load_audio(path: Path):
    from speakercaption.audio import load_any
    audio, sr = load_any(path, target_sr=16000)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), sr


def get_segments(audio, sr, chunk: float):
    """要么按 VAD 真实断句，要么按固定秒数切段（模拟广播里的长段）。"""
    if chunk > 0:
        step = int(chunk * sr)
        return [audio[o:o + step] for o in range(0, len(audio), step)
                if len(audio[o:o + step]) >= 1600]
    vad = build_vad()
    step = 1600  # 100ms，和实时链路一致
    for off in range(0, len(audio), step):
        vad.accept_waveform(audio[off:off + step])
    vad.flush()
    segs = []
    while not vad.empty():
        seg = vad.front
        segs.append(np.ascontiguousarray(
            np.asarray(seg.samples, dtype=np.float32)))
        vad.pop()
    return segs


def main():
    ap = argparse.ArgumentParser(description="说话人分人诊断")
    ap.add_argument("wav")
    ap.add_argument("--threshold", type=float, default=0.60,
                    help="声纹相似度阈值（默认 0.60）")
    ap.add_argument("--num-speakers", type=int, default=0,
                    help="已知人数上限，0=自动")
    ap.add_argument("--chunk", type=float, default=0.0,
                    help="把音频按固定秒数切段，模拟广播音频里的长 VAD 段")
    args = ap.parse_args()

    if not EMB_MODEL.is_file():
        print("缺声纹模型 %s\n运行: python scripts/download_models.py --with-live"
              % EMB_MODEL)
        return 2

    audio, sr = load_audio(Path(args.wav))
    print("音频: %d 采样 @ %d Hz = %.1f 秒" % (len(audio), sr, len(audio) / sr))

    segs = get_segments(audio, sr, args.chunk)
    print("分段数: %d\n" % len(segs))

    ext = build_extractor(EMB_MODEL)
    tracker = OnlineSpeakerTracker(ext,
                                   max_speakers=args.num_speakers or None,
                                   threshold=args.threshold)

    print("%-4s %8s %6s %8s %6s  %s" % ("#", "时长", "标签", "分数", "新?", "说明"))
    print("-" * 78)
    embs = []
    for i, s in enumerate(segs, 1):
        e = tracker.embedding(s)
        embs.append(e)
        label, score, is_new = tracker.assign(s)
        note = "新说话人" if is_new else ("短句沿用" if score < 0 else "匹配已有")
        print("%-4d %7.2fs %6d %8.3f %6s  %s"
              % (i, len(s) / sr, label, score, "是" if is_new else "否", note))

    idx = [i for i, s in enumerate(segs) if len(s) >= 16000 and embs[i] is not None]
    if idx:
        print("\n--- 段间余弦相似度（只看 >=1 秒的段）---")
        print("     " + "".join("%7d" % (j + 1) for j in idx))
        for i in idx:
            row = [float(np.dot(embs[i], embs[j])) for j in idx]
            print("%3d  " % (i + 1) + "".join("%7.3f" % v for v in row))
        print("\n判读：同一说话人的相似度一般 >= 0.7，不同人 < 0.5。")
        print("     若出现「明明相似度很低却判成同一人」，说明该段太长、")
        print("     整段一个声纹把好几个人平均掉了（用 --chunk 复现）。")
    print("\n分出 %d 位说话人。" % tracker.num_speakers())
    return 0


if __name__ == "__main__":
    sys.exit(main())
