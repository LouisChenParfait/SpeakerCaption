#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Smoke test for the sherpa-onnx diarization API + downloaded models.

Run after scripts/download_models.py --with-test-audio:
    python tests/test_smoke.py

It verifies that:
  * both ONNX files exist and the config validates,
  * sd.process() returns (start, end, speaker) tuples,
  * the 2-speaker test wave yields exactly 2 speaker labels,
  * the 4-speaker test wave yields exactly 4 speaker labels.

Also runnable through the unittest suite (``python -m unittest discover tests``):
before it was a bare script, so ``unittest discover`` never picked it up and
"are the models still working?" had no regression protection at all.
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import sherpa_onnx  # noqa: E402

from speakercaption.audio import load_mono  # noqa: E402

# fp32 is the default: int8 merges two speakers on the 4-speaker sample,
# see docs/DEVLOG.md for the measurement.
SEG = ROOT / "models" / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMB = ROOT / "models" / "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"

CASES = [
    ("2 speakers / known count", "1-two-speakers-en.wav", 2, 0.7, 2),
    ("4 speakers / known count", "0-four-speakers-zh.wav", 4, 0.7, 4),
    ("4 speakers / auto estimate", "0-four-speakers-zh.wav", -1, 0.7, 4),
]


def make_config(num_speakers, threshold=0.7):
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(SEG), window_shift_ratio=0.1
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB)),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers, threshold=threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    return config


def run(wav, num_speakers, limit_sec=None, threshold=0.7, expected=0):
    audio, sr = load_mono(wav, target_sr=16000)
    if limit_sec:
        audio = audio[: int(limit_sec * sr)]
    cfg = make_config(num_speakers, threshold)
    if not cfg.validate():
        raise RuntimeError("config.validate() failed -- check model paths")
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    if sr != sd.sample_rate:
        raise RuntimeError("sample rate mismatch: %s vs %s" % (sr, sd.sample_rate))
    t0 = time.time()
    result = sd.process(audio).sort_by_start_time()
    elapsed = time.time() - t0
    speakers = sorted({r.speaker for r in result})
    print("  wav             : %s" % Path(wav).name)
    print("  audio           : %.2fs @ %d Hz" % (len(audio) / sr, sr))
    print("  segments        : %d" % len(result))
    print("  speaker labels  : %s (expected %d different labels)" %
          (speakers, expected or num_speakers))
    print("  diarize time    : %.2fs (%.1fx realtime)" %
          (elapsed, (len(audio) / sr) / max(elapsed, 1e-6)))
    for r in list(result)[:5]:
        print("    %7.2f -- %7.2f  speaker_%02d" % (r.start, r.end, r.speaker))
    return speakers


class TestSmoke(unittest.TestCase):
    def test_models_exist(self):
        for p in (SEG, EMB):
            self.assertTrue(p.is_file(),
                            "缺模型 %s；先跑 scripts/download_models.py" % p)

    def test_diarization_labels(self):
        d = ROOT / "tests" / "data"
        for label, name, n, threshold, expected in CASES:
            with self.subTest(case=label):
                wav = d / name
                if not wav.is_file():
                    self.skipTest("缺测试音频 %s" % wav)
                spk = run(wav, n, threshold=threshold, expected=expected)
                self.assertEqual(len(spk), expected,
                                 "%s：期望 %d 个标签，实际 %s" % (label, expected, spk))


def main():
    for p in (SEG, EMB):
        if not p.is_file():
            print("MISSING MODEL: %s" % p)
            print("Run: python scripts/download_models.py --with-test-audio")
            return 2
        print("model: %s (%.2f MB)" % (p.name, p.stat().st_size / 1048576.0))

    d = ROOT / "tests" / "data"
    bad = 0
    for label, name, n, threshold, expected in CASES:
        print("-" * 66)
        print("case: %s" % label)
        wav = d / name
        if not wav.is_file():
            print("  MISSING TEST WAV: %s" % wav)
            bad += 1
            continue
        try:
            spk = run(wav, n, threshold=threshold, expected=expected)
            if len(spk) != expected:
                print("  !! expected %d speakers, got %d" % (expected, len(spk)))
                bad += 1
        except Exception as e:
            print("  !! FAILED: %s: %s" % (type(e).__name__, e))
            bad += 1
    print("-" * 66)
    print("SMOKE %s" % ("FAILED" if bad else "PASSED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
