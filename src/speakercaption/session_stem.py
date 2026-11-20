"""每次听录生成独立的输出文件名，避免新一场直播覆盖上一场文字稿。

约定：``<base>_<YYYYmmdd_HHMMSS>``，例如 ``live_20261118_232530.txt``。
同一秒内连开两次（或手工造了同名文件）会追加 ``_2`` / ``_3``，
只要 ``outdir`` 给到，就保证返回一个**当前尚未被占用**的 stem。
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path

STAMP_FORMAT = "%Y%m%d_%H%M%S"
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MEDIA_EXT = re.compile(
    r"\.(wav|mp3|m4a|aac|flac|ogg|opus|wma|mp4|mkv|mov|avi|webm|flv|wmv|ts|m4v)$",
    re.I)


def safe_base(base, fallback: str = "live") -> str:
    """把任意文件名（含中文、带空格、从路径截取的）洗成安全的 stem 前缀。"""
    b = _ILLEGAL.sub("_", str(base or "").strip())
    b = _MEDIA_EXT.sub("", b)          # 万一传的是完整文件名，别把 .wav 带进 stem
    b = b.strip(" ._")                 # Windows 不允许以点/空格结尾
    if len(b) > 60:                    # 给时间戳和 _N 留余量
        b = b[:60].rstrip(" ._")
    return b or fallback


def make_session_stem(base="live", outdir=None, formats=("txt",),
                      when=None) -> str:
    """返回 ``<base>_<时间戳>``；若该组输出文件已存在则自动加序号。

    ``outdir=None`` 时只生成名字、不做存在性检查（方便纯逻辑测试）。
    ``formats`` 用于检查所有会写出的后缀，避免只判 .txt 而撞上 .srt。
    """
    d = datetime.fromtimestamp(time.time() if when is None else when)
    stem = "%s_%s" % (safe_base(base), d.strftime(STAMP_FORMAT))
    if outdir is None:
        return stem

    out = Path(outdir)
    fmts = tuple(f for f in (formats or ()) if f) or ("txt",)
    candidate, n = stem, 1
    while any((out / ("%s.%s" % (candidate, f))).exists() for f in fmts):
        n += 1
        candidate = "%s_%d" % (stem, n)
    return candidate


def split_stamp(stem: str) -> tuple:
    """把 ``live_20261118_232530`` 拆成 ``(base, stamp)``；不合格式返回 ``("", "")``。"""
    m = re.search(r"_((\d{8})_(\d{6}))(?:_(\d+))?$", str(stem or ""))
    if not m:
        return "", ""
    return str(stem)[:m.start()], m.group(1)
