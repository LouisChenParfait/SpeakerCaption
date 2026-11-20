#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Environment self-check for SpeakerCaption.

Usage:
    python scripts/check_env.py

Prints Python/package versions, the sherpa-onnx diarization API actually
available on this machine, and whether optional ASR backends are importable.
No models are downloaded and no network requests are made.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import inspect
import os
import platform
import sys
from pathlib import Path

PKGS = [
    "sherpa-onnx",
    "onnxruntime",
    "numpy",
    "soundfile",
    "faster-whisper",
    "gradio",
    "requests",
    "httpx",
    "opencc-python-reimplemented",
    "torch",
]


def ver(name):
    try:
        return md.version(name)
    except Exception:
        return "NOT INSTALLED"


def main():
    print("=" * 66)
    print("SpeakerCaption / environment self-check")
    print("=" * 66)
    print("Python      :", sys.version.split()[0], platform.platform())
    print("Executable  :", sys.executable)
    print("CWD         :", os.getcwd())
    print("-" * 66)
    for p in PKGS:
        print("%-16s %s" % (p, ver(p)))
    print("-" * 66)

    try:
        import sherpa_onnx

        print("sherpa_onnx file :", sherpa_onnx.__file__)
        has = hasattr(sherpa_onnx, "OfflineSpeakerDiarization")
        print("OfflineSpeakerDiarization available :", has)
        if has:
            cls = sherpa_onnx.OfflineSpeakerDiarization
            try:
                print("  __init__ :", inspect.signature(cls.__init__))
            except Exception as e:
                print("  __init__ signature unavailable:", e)
            print("  doc      :", (cls.__doc__ or "")[:600].strip())
            methods = [
                m for m in dir(cls)
                if not m.startswith("_")
            ]
            print("  methods  :", ", ".join(methods))
        for cfg in [
            "OfflineSpeakerDiarizationConfig",
            "OfflineSpeakerSegmentationModelConfig",
            "OfflineSpeakerSegmentationPyannoteModelConfig",
            "SpeakerEmbeddingExtractorConfig",
            "FastClusteringConfig",
        ]:
            obj = getattr(sherpa_onnx, cfg, None)
            if obj is None:
                print("  %-52s : MISSING" % cfg)
            else:
                try:
                    print("  %-52s : %s" % (cfg, inspect.signature(obj.__init__)))
                except Exception:
                    print("  %-52s : %s" % (cfg, obj))
    except Exception as e:
        print("sherpa_onnx import FAILED:", type(e).__name__, e)

    print("-" * 66)
    try:
        import torch

        print("torch cuda available :", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("torch device         :", torch.cuda.get_device_name(0))
    except Exception as e:
        print("torch check skipped:", e)
    print("-" * 66)
    for m in ["faster_whisper", "gradio", "soundfile", "onnxruntime", "opencc"]:
        try:
            importlib.import_module(m)
            print("import %-16s : OK" % m)
        except Exception as e:
            print("import %-16s : FAIL (%s: %s)" % (m, type(e).__name__, e))

    # ---- model assets -------------------------------------------------
    # New users hit "why is there no model in models/?" most often on the
    # Whisper weights, because those land in the HF cache, not in models/.
    # Print where each asset actually is so the answer is visible.
    print("-" * 66)
    print("model assets:")
    root = Path(__file__).resolve().parents[1]
    models = root / "models"
    required = [
        ("pyannote segmentation", models / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"),
        ("CAM++ speaker embedding", models / "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"),
    ]
    optional = [
        ("silero VAD", models / "silero_vad.onnx"),
        ("streaming paraformer", models / "streaming-paraformer" / "encoder.int8.onnx"),
    ]
    for label, path in required + optional:
        mark = "OK " if path.exists() else "-- "
        extra = "" if path.exists() else "   (缺失)"
        print("  [%s] %-26s %s%s" % (mark, label, path.name, extra))

    # Whisper: local project dir first, then HF cache.
    try:
        sys.path.insert(0, str(root / "src"))
        from speakercaption.model_paths import resolve_whisper, size_of
        found = False
        for tier in ("large-v3", "small", "medium", "tiny", "base"):
            source, origin = resolve_whisper(tier)
            if origin in ("local", "hf-cache"):
                where = "工程内" if origin == "local" else "HF 缓存"
                print("  [OK] whisper %-22s %s (%s)" % (tier, where, source))
                found = True
                break
        if not found:
            for tier in ("large-v3", "small", "medium", "tiny", "base"):
                print("  [--] whisper %-22s 未下载，约 %s；"
                      "运行 scripts/download_models.py --with-whisper %s"
                      % (tier, size_of(tier), tier))
                break
    except Exception as e:
        print("  whisper 检查跳过:", e)

    print("=" * 66)
    print("提示：Whisper 权重可选。不装它也能用 --asr none / --final none，"
          "只做分人不出文字。")
    print("=" * 66)


if __name__ == "__main__":
    main()
