#!/usr/bin/env python3
"""SpeakerCaption 实时入口：抓电脑正在播放的声音，边听边出字并标注说话人。

    python live.py                          # Windows 默认 WASAPI 回环
    python live.py --num-speakers 2         # 双人播客
    python live.py --simulate meeting.wav   # 用文件回放验收
    python live.py --list-devices

详见 README「实时听录」章节。
"""
import os
import sys
from pathlib import Path

# 代理会劫持 127.0.0.1：任何访问本机 7861 的 HTTP 都必须绕过代理。
# （ConfuciusASR 已用 trust_env=False，这里再兜一层环境变量。）
for _k in ("NO_PROXY", "no_proxy"):
    _old = os.environ.get(_k, "")
    _add = "127.0.0.1,localhost,::1,0.0.0.0"
    os.environ[_k] = (_old + "," if _old else "") + _add

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from speakercaption.live_core import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
