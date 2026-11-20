#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download the speaker-diarization models (and optional test audio).

The model weights are NOT stored in this repository on purpose. This script
fetches them from the official sherpa-onnx GitHub release assets at first run.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --with-test-audio
    python scripts/download_models.py --with-live      # VAD + streaming ASR for live.py
    python scripts/download_models.py --with-whisper   # Whisper weights (~2.9 GB for large-v3)
    python scripts/download_models.py --with-whisper small   # a smaller tier (~484 MB)
    python scripts/download_models.py --proxy http://127.0.0.1:18081

Whisper tiers (faster-whisper / Systran, MIT licensed):
    tiny 75 MB | base 145 MB | small 484 MB | medium 1.53 GB
    large-v3 2.88 GB | large-v3-turbo 1.62 GB

Whisper weights are downloaded into ``models/whisper/<tier>/`` **inside the
project**, so the whole folder can be zipped and handed to someone else --
no download, no network, fully offline.

Proxy notes:
    The script honours HTTPS_PROXY/HTTP_PROXY from the environment. On some
    Windows machines the environment proxy cannot reach github.com while the
    system (WinHTTP) proxy can; use --proxy in that case. If the first attempt
    fails, the script automatically retries with no proxy and with a few
    common local proxy ports.

    For HuggingFace (Whisper weights) use --endpoint https://hf-mirror.com
    when huggingface.co is unreachable, or set HF_ENDPOINT yourself.
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
TESTS = ROOT / "tests" / "data"

RELEASE_SEG = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models"
RELEASE_REC = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models"
RELEASE_ASR = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
# GitHub release only ships a 1 GB archive for the streaming Paraformer.
# The three individual files (int8 encoder/decoder + tokens) are on HuggingFace;
# hf-mirror.com is reachable from mainland China without a proxy.
HF_LIVE = "https://hf-mirror.com/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en/resolve/main"

# (url, destination, expected size in bytes, description)
ASSETS = [
    (
        RELEASE_SEG + "/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        MODELS / "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        6_958_444,
        "pyannote segmentation 3.0 ONNX (archive, includes LICENSE/README)",
    ),
    (
        RELEASE_REC + "/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        MODELS / "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        28_281_164,
        "3D-Speaker CAM++ zh-en speaker embedding ONNX",
    ),
]

# Extra assets for the real-time listener (live.py). ~232 MB in total.
LIVE_ASSETS = [
    (RELEASE_ASR + "/silero_vad.onnx", MODELS / "silero_vad.onnx", 643_854,
     "Silero VAD ONNX (utterance segmentation for the live listener)"),
    (HF_LIVE + "/tokens.txt", MODELS / "streaming-paraformer" / "tokens.txt", 75_756,
     "streaming Paraformer bilingual zh-en tokens"),
    (HF_LIVE + "/encoder.int8.onnx",
     MODELS / "streaming-paraformer" / "encoder.int8.onnx", 165_462_184,
     "streaming Paraformer int8 encoder"),
    (HF_LIVE + "/decoder.int8.onnx",
     MODELS / "streaming-paraformer" / "decoder.int8.onnx", 71_664_561,
     "streaming Paraformer int8 decoder"),
]

TEST_AUDIO = [
    (RELEASE_SEG + "/1-two-speakers-en.wav", TESTS / "1-two-speakers-en.wav", 512_044,
     "official 2-speaker English test wave"),
    (RELEASE_SEG + "/0-four-speakers-zh.wav", TESTS / "0-four-speakers-zh.wav", 1_819_586,
     "official 4-speaker Chinese test wave"),
]

FALLBACK_PROXIES = [
    "http://127.0.0.1:18081",   # typical system/WinHTTP proxy on this machine
    "http://127.0.0.1:22641",   # typical environment-variable proxy
    None,                        # direct connection
]

# ---------------------------------------------------------------------------
# Whisper (faster-whisper / Systran) -- weights live on HuggingFace, not PyPI.
# ---------------------------------------------------------------------------
WHISPER_HOME = MODELS / "whisper"

# tier -> (HF repo suffix, approx weight size in bytes)
WHISPER_TIERS = {
    "tiny": 75_000_000,
    "base": 145_000_000,
    "small": 484_000_000,
    "medium": 1_530_000_000,
    "large-v3": 3_088_000_000,
    "large-v3-turbo": 1_620_000_000,
    "distil-large-v3": 1_530_000_000,
}

# Only these files are needed; skipping the rest keeps the download lean.
WHISPER_ALLOW = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


def whisper_repo(tier: str) -> str:
    return "Systran/faster-whisper-%s" % tier


def fetch_whisper(tier, endpoint=None, proxy=None, force=False):
    """Download one faster-whisper tier into models/whisper/<tier>/.

    Uses huggingface_hub.snapshot_download with local_dir, so the files are
    real files in the project (not symlinks into a cache) -- that is what
    makes "zip models/ and share it" work.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[fail] huggingface_hub not installed: pip install huggingface_hub")
        return False

    dest = WHISPER_HOME / tier
    if dest.is_dir() and (dest / "model.bin").exists() and not force:
        print("[skip] whisper %-16s already present (%s)" % (tier, dest))
        return True

    repo = whisper_repo(tier)
    approx = WHISPER_TIERS.get(tier, 0)
    print("-" * 72)
    print("Whisper tier : %s  (~%s)" % (tier, human(approx)))
    print("  repo       : https://huggingface.co/%s" % repo)
    print("  target     : %s" % dest)
    if endpoint:
        print("  endpoint   : %s" % endpoint)

    dest.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(repo_id=repo, local_dir=str(dest),
                  allow_patterns=WHISPER_ALLOW)
    if endpoint:
        kwargs["endpoint"] = endpoint
    if proxy:
        # huggingface_hub reads the standard env vars; set them for the call.
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
    try:
        t0 = time.time()
        snapshot_download(**kwargs)
    except Exception as e:
        print("[fail] whisper %s: %s: %s" % (tier, type(e).__name__, e))
        print("       国内网络可试：--endpoint https://hf-mirror.com")
        return False
    ok = (dest / "model.bin").exists()
    print("[%s] whisper %-16s %s  %.1fs" %
          ("ok  " if ok else "fail", tier, dest, time.time() - t0))
    if not ok:
        print("       目录里没有 model.bin，权重可能没下全")
    return ok


def make_opener(proxy):
    if proxy:
        h = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    else:
        h = urllib.request.ProxyHandler({})
    return urllib.request.build_opener(h)


def candidate_proxies(explicit):
    if explicit is not None:
        return [explicit]
    seen, out = set(), []
    env = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") \
        or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    if env:
        out.append(env)
        seen.add(env)
    for p in FALLBACK_PROXIES:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def human(n):
    return "%.2f MB" % (n / 1024.0 / 1024.0)


def fetch(url, dest, expected, proxy_list, label):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == expected:
        print("[skip] %-58s %s" % (dest.name, human(expected)))
        return True
    req = urllib.request.Request(url, headers={"User-Agent": "SpeakerCaption/0.1"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    for proxy in proxy_list:
        tag = proxy or "direct"
        try:
            opener = make_opener(proxy)
            t0 = time.time()
            with opener.open(req, timeout=90) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                got, shown_pct = 0, -1
                while True:
                    buf = r.read(262144)
                    if not buf:
                        break
                    f.write(buf)
                    got += len(buf)
                    # Print at most a few lines per file: some Windows consoles
                    # do not honour \r and would flood the log with progress.
                    pct = int(100.0 * got / total) if total else 0
                    step = pct // 25
                    if total and step > shown_pct // 25:
                        print("[get ] %-34s %s / %s  %d%% (%s)" %
                              (dest.name[:34], human(got), human(total), pct, tag))
                        shown_pct = pct
            size = tmp.stat().st_size
            if expected and size != expected:
                raise RuntimeError("size mismatch: got %d, expected %d" % (size, expected))
            os.replace(tmp, dest)
            print("[ok  ] %-58s %s  %.1fs  via %s" %
                  (dest.name[:58], human(size), time.time() - t0, tag))
            print("        %s" % label)
            return True
        except Exception as e:
            sys.stdout.write("\r" + " " * 96 + "\r")
            print("[fail] %s via %s: %s: %s" % (dest.name, tag, type(e).__name__, e))
            if tmp.exists():
                tmp.unlink()
    return False


def safe_extract_bz2(archive: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    root = outdir.resolve()
    with tarfile.open(archive, "r:bz2") as tf:
        for member in tf.getmembers():
            target = (outdir / member.name).resolve()
            if not str(target).startswith(str(root)):
                raise RuntimeError("unsafe path in archive: %s" % member.name)
        try:
            tf.extractall(outdir, filter="data")   # Python >= 3.12
        except TypeError:
            tf.extractall(outdir)                  # older Python
    return sorted(str(p.relative_to(outdir)) for p in outdir.rglob("*") if p.is_file())


def main():
    ap = argparse.ArgumentParser(description="Download diarization models")
    ap.add_argument("--proxy", default=None,
                    help="explicit proxy, e.g. http://127.0.0.1:18081")
    ap.add_argument("--with-test-audio", action="store_true",
                    help="also download official 2/4-speaker test wavs")
    ap.add_argument("--with-live", action="store_true",
                    help="also download VAD + streaming Paraformer for live.py (about 232 MB)")
    ap.add_argument("--with-whisper", nargs="?", const="large-v3",
                    metavar="TIER",
                    help="also download faster-whisper weights into models/whisper/. "
                         "TIER: tiny|base|small|medium|large-v3|large-v3-turbo "
                         "(default: large-v3, about 2.9 GB)")
    ap.add_argument("--whisper-tier", nargs="*", metavar="TIER",
                    help="download several tiers at once, e.g. small medium")
    ap.add_argument("--endpoint", default=None,
                    help="HuggingFace endpoint for whisper weights, "
                         "e.g. https://hf-mirror.com (China-friendly mirror)")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    proxies = candidate_proxies(args.proxy)
    print("=" * 72)
    print("SpeakerCaption / downloading model assets")
    print("project :", ROOT)
    print("proxies :", ", ".join(str(p) for p in proxies))
    print("=" * 72)

    if args.force:
        for _, dest, _, _ in ASSETS + LIVE_ASSETS:
            if dest.exists():
                dest.unlink()

    ok = True
    for url, dest, size, label in ASSETS:
        ok = fetch(url, dest, size, proxies, label) and ok

    seg_dir = MODELS / "sherpa-onnx-pyannote-segmentation-3-0"
    archive = MODELS / "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    if archive.exists():
        if not seg_dir.exists() or not (seg_dir / "model.int8.onnx").exists():
            print("[unpack] %s" % archive.name)
            files = safe_extract_bz2(archive, MODELS)
            print("         " + ", ".join(files[:12]))
        else:
            print("[skip] segmentation archive already unpacked")

    if args.with_test_audio:
        for url, dest, size, label in TEST_AUDIO:
            ok = fetch(url, dest, size, proxies, label) and ok

    if args.with_live:
        print("-" * 72)
        print("Live listener assets (VAD + streaming Paraformer, about 232 MB)")
        for url, dest, size, label in LIVE_ASSETS:
            ok = fetch(url, dest, size, proxies, label) and ok

    # Whisper tiers: --with-whisper [TIER] and/or --whisper-tier A B C
    tiers = []
    if args.with_whisper:
        tiers.append(args.with_whisper)
    for t in (args.whisper_tier or []):
        tiers.append(t)
    if tiers:
        print("-" * 72)
        print("Whisper weights (faster-whisper / Systran, MIT)")
        seen = set()
        for tier in tiers:
            if tier in seen:
                continue
            seen.add(tier)
            if tier not in WHISPER_TIERS:
                print("[fail] unknown whisper tier: %s" % tier)
                print("       available: %s" % ", ".join(sorted(WHISPER_TIERS)))
                ok = False
                continue
            ok = fetch_whisper(tier, endpoint=args.endpoint,
                               proxy=args.proxy, force=args.force) and ok

    print("=" * 72)
    if not ok:
        print("FAILED: some assets could not be downloaded. Try --proxy <your proxy>.")
        return 1
    print("Done. Files under %s:" % MODELS)
    for p in sorted(MODELS.rglob("*")):
        if p.is_file():
            print("  %-72s %s" % (str(p.relative_to(ROOT)), human(p.stat().st_size)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
