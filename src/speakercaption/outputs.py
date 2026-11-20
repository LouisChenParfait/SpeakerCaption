"""Output writers: txt / srt / json / rttm.

All four files are (re)created on every run with the same stem, so one audio
file always maps to one deterministic set of results. No per-sentence files.
"""
from __future__ import annotations

import json
from pathlib import Path


def speaker_name(n: int, names=None) -> str:
    names = names or {}
    return names.get(n) or "说话人%d" % n


def clock(sec: float, comma: bool = True) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    ms = int(round((s - int(s)) * 1000))
    if ms == 1000:
        s, ms = int(s) + 1, 0
    sep = "," if comma else "."
    return "%02d:%02d:%02d%s%03d" % (h, m, int(s), sep, ms)


def stamp(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return ("%02d:%02d:%04.1f" % (h, m, s)) if h else ("%02d:%04.1f" % (m, s))


def to_txt(segments, names=None, timestamps=True, show_speaker=True) -> str:
    """单人模式下 show_speaker=False：每行只留时间戳和正文，不再写「说话人1：」。"""
    lines = []
    for s in segments:
        head = "[%s] " % stamp(s.start) if timestamps else ""
        who = "%s：" % speaker_name(s.speaker, names) if show_speaker else ""
        lines.append("%s%s%s" % (head, who, s.text.strip()))
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def to_srt(segments, names=None, show_speaker=True) -> str:
    blocks = []
    for i, s in enumerate(segments, 1):
        body = s.text.strip() or "（无转写）"
        who = "[%s] " % speaker_name(s.speaker, names) if show_speaker else ""
        blocks.append("%d\n%s --> %s\n%s%s\n" % (
            i, clock(s.start), clock(s.end), who, body
        ))
    return "\n".join(blocks)


def to_json(segments, meta=None, names=None) -> str:
    data = dict(meta or {})
    data["speakers"] = [
        {"id": n, "name": speaker_name(n, names),
         "total_seconds": round(sum(s.duration for s in segments if s.speaker == n), 3)}
        for n in sorted({s.speaker for s in segments})
    ]
    data["segments"] = [
        {"start": round(s.start, 3), "end": round(s.end, 3),
         "duration": round(s.duration, 3), "speaker": s.speaker,
         "speaker_name": speaker_name(s.speaker, names), "text": s.text}
        for s in segments
    ]
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def to_rttm(segments, file_id="audio") -> str:
    lines = []
    for s in segments:
        lines.append("SPEAKER %s 1 %.3f %.3f <NA> <NA> speaker_%02d <NA> <NA>" % (
            file_id, s.start, max(0.0, s.duration), s.speaker
        ))
    return "\n".join(lines) + ("\n" if lines else "")


def write_outputs(segments, outdir, stem, names=None, timestamps=True, meta=None,
                  formats=("txt", "srt", "json", "rttm"),
                  show_speaker=True) -> dict:
    """json / rttm 始终保留结构化 speaker 字段（便于下游处理），
    只有给人看的 txt / srt 才受 show_speaker 影响。"""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    writers = {
        "txt": lambda: to_txt(segments, names, timestamps, show_speaker),
        "srt": lambda: to_srt(segments, names, show_speaker),
        "json": lambda: to_json(segments, meta, names),
        "rttm": lambda: to_rttm(segments, stem),
    }
    paths = {}
    for fmt in formats:
        if fmt not in writers:
            continue
        p = out / ("%s.%s" % (stem, fmt))
        p.write_text(writers[fmt](), encoding="utf-8")
        paths[fmt] = str(p)
    return paths
