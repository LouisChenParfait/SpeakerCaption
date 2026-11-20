# -*- coding: utf-8 -*-
"""Confucius 进程内转写桥接 worker（独立进程）。

为什么要有这个文件
------------------
`Confucius4-R2T2` 的推理核心是一个 Cython 编译模块 `cy_app.pyd`，它
**依赖 torch**。而 SpeakerCaption 自己的虚拟环境刻意不装 torch（那要
2.5 GB+ 磁盘，而且和 faster-whisper 的 ctranslate2 推理栈互相干扰）。

所以本 worker 用** Confucius 自带的那个 Python 解释器**（它的 py312/ 里
已经装好了 torch 2.8 + CUDA）跑，把 `cy_app.do_text()` 在**进程内**直接
调用——不再需要 Gradio HTTP 服务、不需要占 7861 端口、不需要用户手动
先启动另一个程序。

通信协议：走 stdin/stdout 的 JSON 行（JSONL）。
  → {"audio": "/abs/path/seg.wav", "hot_word": "雷军", "id": 1}
  ← {"id": 1, "ok": true, "text": "识别结果"}
  ← {"id": 1, "ok": false, "error": "..."}

之所以用文件路径而不是 ndarray：`do_text()` 内部会自己解码音频并
处理采样率，传路径最省事也最稳（它的 Gradio 版本收到的是 FileData，
实际也是路径）。

这个模块**只在被当作 __main__ 运行时**才启动 worker loop，
因此可以被安全地 import 来做路径探测。
"""
from __future__ import annotations

import json
import os
import sys
import traceback


def _emit(obj):
    """写一行 JSON 到 stdout 并立刻 flush（父进程要实时读到）。"""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _find_cy_app(project_dir):
    """在项目目录里定位 cy_app 模块。

    优先直接 import（它就在项目根目录），失败时按文件路径加载。
    """
    project_dir = os.path.abspath(project_dir)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    # cy_app.pyd 依赖同目录下的 config.py / mel 等资源，
    # 把工作目录也切过去，模型路径才解析得到。
    try:
        os.chdir(project_dir)
    except OSError:
        pass
    import cy_app  # noqa: E402
    return cy_app


def main():
    if len(sys.argv) < 2:
        _emit({"ok": False, "error": "usage: confucius_worker.py <project_dir>"})
        return 2
    project_dir = sys.argv[1]

    try:
        cy_app = _find_cy_app(project_dir)
    except Exception as e:
        _emit({"ok": False, "error": "cannot import cy_app from %s: %s: %s"
                              % (project_dir, type(e).__name__, e)})
        return 3

    _emit({"ok": True, "ready": True, "event": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            _emit({"ok": False, "error": "bad json: %s" % e})
            continue

        rid = req.get("id")
        audio = req.get("audio")
        hot_word = req.get("hot_word", "") or ""
        model_type = req.get("model_type", "Confucius4-R2T2-model")

        if not audio or not os.path.isfile(audio):
            _emit({"id": rid, "ok": False, "error": "audio not found: %s" % audio})
            continue

        try:
            # 契约（从 Gradio config 反推）：
            #   do_text(audio, video, input_chage, text_type, hot_word, model_type)
            # 返回的是 generator，消费后拿到 (文本, 输出文件路径) 的元组。
            gen = cy_app.do_text(audio, None, "音频", "文本", hot_word, model_type)
            result = None
            for item in gen:
                result = item
                break        # 第一个 yield 就是最终结果，后面的别等
            text = ""
            if isinstance(result, (tuple, list)) and result:
                text = result[0] or ""
            elif result is not None:
                text = str(result)
            _emit({"id": rid, "ok": True, "text": str(text).strip()})
        except Exception as e:
            _emit({"id": rid, "ok": False,
                   "error": "%s: %s" % (type(e).__name__, e),
                   "trace": traceback.format_exc()[-800:]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
