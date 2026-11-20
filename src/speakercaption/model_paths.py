# -*- coding: utf-8 -*-
"""模型来源解析：让「没装 Whisper 的人」也能顺畅跑起来。

背景
----
faster-whisper 的**权重**不在 pip 包里，在 HuggingFace 上（MIT 许可）。
第一次用 ``WhisperModel("large-v3")`` 时它会��动下载到 HF 缓存目录：

    Windows  %USERPROFILE%\\.cache\\huggingface\\hub\\models--Systran--faster-whisper-large-v3
    Linux    ~/.cache/huggingface/hub/...
    macOS    ~/.cache/huggingface/hub/...

这带来两个问题：

1. **不在工程目录里** —— 换机器、换用户、放进容器都要重下；
2. **默认走 huggingface.co** —— 国内网络经常连不上，需要镜像。

本模块定义一套**来源优先级**，让用户有三种明确可选的方式：

    1) 工程内目录   models/whisper/<档位>/     ← 随工程走、可离线、可拷给同事
    2) 显式本地路径  任意目录                    ← 用户自己管的模型
    3) HuggingFace   Systran/faster-whisper-*    ← 兜底：自动下载（可用镜像）

解析顺序（``resolve_whisper``）：

    显式路径(用户填的) > 工程内 models/whisper/<档位> > HF 缓存 > 在线下载

关键点：只要工程内存在模型，就**完全不发网络请求**，天然离线可用。
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
WHISPER_HOME = MODELS / "whisper"

# faster-whisper 支持的档位 -> 权重体积（字节，仅用于提示）
WHISPER_SIZES = {
    "tiny": 75_000_000,
    "base": 145_000_000,
    "small": 484_000_000,
    "medium": 1_530_000_000,
    "large-v3": 3_088_000_000,
    "large-v3-turbo": 1_620_000_000,
    "distil-large-v3": 1_530_000_000,
}

# HF 仓库名
REPO_PREFIX = "models--Systran--faster-whisper-"
REPO_FALLBACK = "Systran/faster-whisper-%s"


def human(n: float) -> str:
    if n >= 1073741824:
        return "%.2f GB" % (n / 1073741824.0)
    return "%.0f MB" % (n / 1048576.0)


def size_of(name: str) -> str:
    """粗略体积提示，用于下载前告知用户。"""
    base = name.replace(".en", "")
    for key in ("large-v3-turbo", "distil-large-v3", "large-v3", "medium", "small", "base", "tiny"):
        if base.startswith(key):
            return human(WHISPER_SIZES.get(key, 0))
    return "?"


def _looks_like_path(value: str) -> bool:
    """判断用户填的是路径而不是档位名。"""
    if not value:
        return False
    if os.sep in value or "/" in value:
        return True
    # Windows 盘符，如 D:\models\whisper
    return len(value) > 1 and value[1] == ":"


def is_model_dir(path: Path) -> bool:
    """目录里是否像一个完整的 faster-whisper 模型。"""
    try:
        if not path.is_dir():
            return False
        return (path / "model.bin").exists() and (path / "config.json").exists()
    except OSError:
        return False


def hf_cache_dir() -> Path:
    """HF 缓存根目录（尊重 HF_HOME 环境变量）。"""
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def find_in_hf_cache(name: str) -> Path | None:
    """在 HF 缓存里找已下载的 faster-whisper 模型。"""
    if _looks_like_path(name):
        return None
    repo = hf_cache_dir() / (REPO_PREFIX + name)
    snaps = repo / "snapshots"
    if not snaps.is_dir():
        return None
    for snap in sorted(snaps.glob("*")):
        if is_model_dir(snap):
            return snap
    return None


def find_local(name: str, download_root: str | None = None) -> Path | None:
    """在工程内 / 指定目录里找模型。

    查找位置：
      - ``download_root`` 直接指向的目录
      - ``download_root/<档位>``
      - ``models/whisper/<档位>``（工程内默认位置）
      - ``models/whisper/<档位>.bin`` 之类的老式单文件（不适用 CTranslate2，跳过）
    """
    if _looks_like_path(name):
        p = Path(name).expanduser()
        if is_model_dir(p):
            return p.resolve()
        return None

    candidates = []
    if download_root:
        d = Path(download_root).expanduser()
        candidates.append(d)
        candidates.append(d / name)
    candidates.append(WHISPER_HOME / name)

    for c in candidates:
        if is_model_dir(c):
            return c.resolve()
    return None


def resolve_whisper(name: str, download_root: str | None = None):
    """解析 whisper 模型来源。

    返回 ``(source, origin)``：
      - ``source``：可以直接传给 ``WhisperModel(...)`` 的路径或档位名
      - ``origin``：``"local"`` / ``"hf-cache"`` / ``"download"``，用于日志与提示

    顺序：
      1. 用户填的显式路径（存在就用，不存在则**不静默降级**，由调用方报错）
      2. ``download_root`` 指定的目录
      3. 工程内 ``models/whisper/<档位>``
      4. HF 缓存里已下载的
      5. 都没有 → 交给 faster-whisper 在线下载（此时应传 ``download_root``）
    """
    name = (name or "large-v3").strip()

    # 1) 显式路径：用户明确指定了，必须尊重；不存在就报错而不是偷偷换别的模型
    if _looks_like_path(name):
        p = Path(name).expanduser()
        if is_model_dir(p):
            return str(p.resolve()), "local"
        # 有目录但内容不全 -> 可能是 HF 缓存的 snapshot 结构
        if p.is_dir():
            return str(p.resolve()), "local"
        return None, "missing"

    # 2) download_root 指向的目录本身就是一个模型
    if download_root:
        d = Path(download_root).expanduser()
        if is_model_dir(d):
            return str(d.resolve()), "local"

    # 3) 工程内 models/whisper/<档位>
    hit = find_local(name, download_root)
    if hit:
        return str(hit), "local"

    # 4) HF 缓存
    hit = find_in_hf_cache(name)
    if hit:
        return str(hit), "hf-cache"

    # 5) 在线下载
    return name, "download"


def describe(name: str = "large-v3") -> str:
    """一行式说明，给 GUI 状态栏和文档用。"""
    return "%s（约 %s）" % (name, size_of(name))
