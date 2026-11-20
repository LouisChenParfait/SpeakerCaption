"""Command line entry point for SpeakerCaption."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .audio import load_any, probe, slice_audio
from .asr import ASRError, build_asr
from .diarize import Diarizer, SEG_INT8, ROOT, merge_adjacent
from .live_core import _resolve_outdir
from .outputs import speaker_name, write_outputs
from .text_pipeline import SCRIPT_CHOICES, build_pipeline


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="diarize.py",
        description="Offline speaker-diarized transcription (zh/en, cross-platform)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="audio/video file (wav/flac/mp3/m4a/mp4 via ffmpeg)")
    p.add_argument("--outdir", default=str(ROOT / "out"),
                   help="output directory; files are recreated on every run")
    p.add_argument("--num-speakers", type=int, default=-1,
                   help="known speaker count; -1 = estimate automatically; "
                        "1 = single speaker (no diarization labels)")
    p.add_argument("--single-speaker", action="store_true",
                   help="shortcut for --num-speakers 1: single-speaker audio, "
                        "omit speaker labels in txt/srt")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="clustering threshold when --num-speakers=-1 "
                        "(larger = fewer speakers)")
    p.add_argument("--merge-gap", type=float, default=0.3,
                   help="join same-speaker segments separated by <= this many seconds")
    p.add_argument("--seg-int8", action="store_true",
                   help="use the smaller int8 segmentation model (faster, "
                        "but measured to merge speakers in 4+ speaker audio)")
    p.add_argument("--asr", choices=["none", "whisper", "confucius", "confucius-local"],
                   default="whisper",
                   help="ASR backend; confucius-local = 进程内直调 Confucius4-R2T2，"
                        "不需要先启动 7861 服务")
    p.add_argument("--language", default=None,
                   help="force ASR language, e.g. zh / en; default auto-detect")
    p.add_argument("--hot-word", default="",
                   help="comma-separated hot words (Confucius backend only)")
    p.add_argument("--confucius-url", default="http://127.0.0.1:7861",
                   help="URL of a user-deployed Confucius4-R2T2 Gradio service")
    p.add_argument("--confucius-dir", default=None,
                   help="Confucius4-R2T2 project directory (for --asr confucius-local); "
                        "auto-detected when omitted, or set CONFUCIUS_DIR")
    p.add_argument("--whisper-model", default="large-v3",
                   help="faster-whisper model name or local path")
    p.add_argument("--whisper-device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--whisper-compute", default="auto",
                   help="auto / float16 / int8_float16 / int8 / float32")
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--vad-filter", action="store_true",
                   help="apply Whisper's internal VAD (usually unnecessary here)")
    p.add_argument("--download-root", default=None,
                   help="directory for cached Whisper models")
    p.add_argument("--pad", type=float, default=0.10,
                   help="seconds of padding around each ASR segment")
    p.add_argument("--timestamps", action="store_true",
                   help="prefix each .txt line with a timestamp (off by default)")
    p.add_argument("--script", choices=list(SCRIPT_CHOICES), default="simplified",
                   help="Chinese script: simplified / traditional / as-is (default simplified)")
    p.add_argument("--traditional", action="store_true", help="shortcut for --script traditional")
    p.add_argument("--simplify", action="store_true",
                   help="shortcut for --script simplified (legacy alias)")
    p.add_argument("--translate", action="store_true",
                   help="translate non-Chinese ASR text to Chinese via local Ollama (off by default)")
    p.add_argument("--translate-model", default="auto",
                   help="Ollama model name; auto picks an installed translation model")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--translate-timeout", type=float, default=60.0)
    p.add_argument("--speaker-names", default="",
                   help='map labels to names, e.g. "1=张三,2=李四"')
    p.add_argument("--formats", default="txt,srt,json,rttm",
                   help="comma-separated output formats")
    p.add_argument("--progress", action="store_true", help="show diarization progress")
    p.add_argument("--version", action="version", version="SpeakerCaption " + __version__)
    return p.parse_args(argv)


def parse_names(spec):
    out = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[int(k.strip())] = v.strip()
        except ValueError:
            pass
    return out


def main(argv=None):
    args = parse_args(argv)
    single = bool(args.single_speaker or args.num_speakers == 1)
    if args.traditional:
        args.script = "traditional"
    elif args.simplify:
        args.script = "simplified"
    pipeline = build_pipeline(translate=args.translate, script=args.script,
                              ollama_url=args.ollama_url,
                              translate_model=args.translate_model,
                              translate_timeout=args.translate_timeout)
    src = Path(args.input).expanduser().resolve()
    if not src.is_file():
        print("ERROR: input not found: %s" % src, file=sys.stderr)
        return 2

    info = probe(src)
    print("=" * 72)
    print("SpeakerCaption %s / offline diarized transcription" % __version__)
    print("=" * 72)
    print("input      : %s" % src)
    print("duration   : %.2fs  %d Hz  %d ch  %s" %
          (info.duration, info.sample_rate, info.channels, info.subtype))
    print("speakers   : %s (threshold %.2f)" %
          ("single (no diarization)" if single else
           (args.num_speakers if args.num_speakers > 0 else "auto"), args.threshold))
    print("asr        : %s (language=%s)" % (args.asr, args.language or "auto"))
    print("text       : script=%s translate=%s speaker_labels=%s" %
          (args.script, "on" if args.translate else "off",
           "off" if single else "on"))
    print("-" * 72)

    t_all = time.time()
    names = parse_names(args.speaker_names)
    audio, sr = load_any(src, target_sr=16000)
    duration = len(audio) / sr

    dia = Diarizer(
        segmentation_model=str(SEG_INT8) if args.seg_int8 else None,
        # 单人模式仍要跑 segmentation（它同时负责断句），但聚类强制走自动，
        # 之后把所有片段统一标成 1，避免 num_clusters=1 的后端边界行为。
        num_speakers=-1 if single else args.num_speakers,
        threshold=args.threshold,
    )
    print("[diarize] segmentation=%s" % Path(dia.seg_model).name)
    print("[diarize] embedding   =%s" % Path(dia.emb_model).name)
    segments = dia.process(audio, sr, merge_gap=args.merge_gap,
                           progress=args.progress)
    if single:
        for s in segments:
            s.speaker = 1
        segments = merge_adjacent(segments, args.merge_gap)
        print("[diarize] 单人模式：已合并为 1 位说话人，txt/srt 不显示说话人")
    speakers = sorted({s.speaker for s in segments})
    print("[diarize] %d segments / %d speakers in %.2fs (%.1fx realtime)" %
          (len(segments), len(speakers), dia.elapsed, duration / max(dia.elapsed, 1e-6)))
    for s in segments:
        print("  %7.2f -- %7.2f  %s" %
              (s.start, s.end, "" if single else speaker_name(s.speaker, names)))
    if not segments:
        print("WARNING: no speech segments found.", file=sys.stderr)

    asr = None
    try:
        if args.asr != "none":
            asr = build_asr(
                args.asr,
                whisper_model=args.whisper_model,
                whisper_device=args.whisper_device,
                whisper_compute=args.whisper_compute,
                language=args.language,
                task="transcribe",
                beam_size=args.beam_size,
                vad_filter=args.vad_filter,
                download_root=args.download_root,
                confucius_url=args.confucius_url,
                confucius_dir=args.confucius_dir,
                hot_word=args.hot_word,
            )
            print("-" * 72)
            t_asr = time.time()
            for i, s in enumerate(segments, 1):
                chunk = slice_audio(audio, sr, s.start, s.end, pad=args.pad)
                try:
                    s.text = pipeline.process(asr.transcribe(chunk, sr))
                except Exception as e:
                    print("  [warn] segment %d failed: %s: %s" %
                          (i, type(e).__name__, e), file=sys.stderr)
                    s.text = ""
                preview = s.text.replace("\n", " ")[:56]
                if single:
                    print("  [%2d/%2d] %s" % (i, len(segments), preview))
                else:
                    print("  [%2d/%2d] %s: %s" %
                          (i, len(segments), speaker_name(s.speaker, names), preview))
            print("[asr] total %.2fs" % (time.time() - t_asr))
    except ASRError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        print("Hint: use --asr none to output diarization only.", file=sys.stderr)
        return 3
    finally:
        if asr is not None and hasattr(asr, "close"):
            asr.close()
        pipeline.close()

    meta = {
        "version": __version__,
        "source": str(src),
        "duration_seconds": round(duration, 3),
        "sample_rate": sr,
        "num_speakers_detected": len(speakers),
        "num_speakers_requested": args.num_speakers,
        "speaker_mode": "single" if single else "diarize",
        "clustering_threshold": args.threshold,
        "segmentation_model": Path(dia.seg_model).name,
        "embedding_model": Path(dia.emb_model).name,
        "asr_backend": args.asr,
        "language": args.language or "auto",
        "script": args.script,
        "translate": bool(args.translate),
        "diarize_seconds": round(dia.elapsed, 3),
    }
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    paths = write_outputs(segments, _resolve_outdir(args.outdir), src.stem, names=names,
                          timestamps=args.timestamps, meta=meta, formats=formats,
                          show_speaker=not single)
    print("=" * 72)
    print("done in %.2fs  ->  %d segments, %d speakers" %
          (time.time() - t_all, len(segments), len(speakers)))
    for k, v in paths.items():
        print("  %-5s %s" % (k, v))
    return 0
