# -*- coding: utf-8 -*-
"""查看 / 迁移 faster-whisper 模型缓存。

faster-whisper 默认把模型下到 HuggingFace 缓存目录（Windows 上是
``%USERPROFILE%\\.cache\\huggingface\\hub``），**不在工程目录里**，
所以在 models/ 下看不到它，很容易以为「模型没下」。

本脚本做三件事：
  list     列出本机已缓存的 faster-whisper 模型及大小
  path     打印某个模型的实际路径（可直接填进「Whisper 模型」选项）
  link     把模型「链接」到工程的 models/whisper/<名字>（不复制，不占额外空间）

用法：
    python scripts/whisper_models.py list
    python scripts/whisper_models.py path large-v3
    python scripts/whisper_models.py link large-v3
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
LINK_DIR = MODELS / "whisper"

# faster-whisper 的模型在 HF 上用的仓库名
REPO_PREFIX = "models--Systran--faster-whisper-"


def hf_hub_dir():
    """HF 缓存根目录（尊重 HF_HOME 环境变量）。"""
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def list_models():
    hub = hf_hub_dir()
    if not hub.is_dir():
        print("HF 缓存目录不存在: %s" % hub)
        print("（首次运行 faster-whisper 时会自动下载）")
        return []
    found = []
    for repo in sorted(hub.glob(REPO_PREFIX + "*")):
        snaps = list((repo / "snapshots").glob("*")) if (repo / "snapshots").is_dir() else []
        if not snaps:
            continue
        snap = snaps[0]
        name = repo.name.replace(REPO_PREFIX, "")
        size = 0
        for f in snap.rglob("*"):
            if f.is_file():
                try:
                    size += f.stat().st_size
                except OSError:
                    pass
        found.append((name, snap, size))
    if not found:
        print("缓存里没有 faster-whisper 模型: %s" % hub)
    return found


def cmd_list(_args):
    hub = hf_hub_dir()
    print("HF 缓存目录: %s" % hub)
    print("=" * 72)
    models = list_models()
    for name, snap, size in models:
        print("  %-16s %6.2f GB   %s" % (name, size / 1073741824.0, snap))
    print("=" * 72)
    print("提示：把上面「Whisper 模型」填成 large-v3 即可；")
    print("      也可以填上面的完整路径，直接从磁盘加载。")
    print("      工程内自建模型放 models/ 目录，脚本 download_models.py 会用。")
    return 0


def cmd_path(args):
    hub = hf_hub_dir()
    target = args.model
    for name, snap, _ in list_models():
        if name == target or target in name:
            print(snap)
            return 0
    print("没找到模型: %s" % target)
    print("先跑: python scripts/whisper_models.py list")
    return 1


def cmd_link(args):
    """在工程里建一个指向缓存的目录链接（不复制数据）。"""
    hub = hf_hub_dir()
    target = args.model
    src = None
    for name, snap, _ in list_models():
        if name == target or target in name:
            src = snap
            break
    if src is None:
        print("没找到模型: %s" % target)
        return 1

    LINK_DIR.mkdir(parents=True, exist_ok=True)
    dst = LINK_DIR / target
    if dst.exists() or dst.is_symlink():
        print("已存在: %s" % dst)
        return 0

    try:
        # Windows 需要管理员或开发者模式才能建符号链接
        os.symlink(src, dst, target_is_directory=True)
        print("已创建符号链接: %s -> %s" % (dst, src))
    except (OSError, NotImplementedError) as e:
        print("符号链接失败(%s)，改用目录联接(junction)" % e)
        try:
            subprocess_run = shutil.os.system
            r = subprocess_run('cmd /c mklink /J "%s" "%s"' % (dst, src))
            if r == 0:
                print("已创建目录联接: %s -> %s" % (dst, src))
            else:
                print("联接失败，请手动复制目录")
                return 1
        except Exception as e2:
            print("联接失败: %s" % e2)
            return 1
    print("现在可以把「Whisper 模型」填成: %s" % dst)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="faster-whisper 模型缓存管理")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("list", help="列出已缓存的模型")
    pp = sub.add_parser("path", help="打印某模型的完整路径")
    pp.add_argument("model")
    pl = sub.add_parser("link", help="在工程 models/whisper/ 建指向缓存的链接")
    pl.add_argument("model")
    args = p.parse_args(argv)
    cmd = args.cmd or "list"
    if cmd == "list":
        return cmd_list(args)
    if cmd == "path":
        return cmd_path(args)
    if cmd == "link":
        return cmd_link(args)
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
