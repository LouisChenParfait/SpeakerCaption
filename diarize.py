#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SpeakerCaption CLI entry point.

    python diarize.py INPUT [options]

Run `python diarize.py --help` for the full option list.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from speakercaption.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
