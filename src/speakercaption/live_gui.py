"""实时字幕窗口：设置面板 + 全部历史 + 说话人分色 + 底部电平状态条。

Tkinter 只跑在主线程；采集流水线由控制器放在工作线程里。设置面板上的
「译成中文 / 中文字形 / 时间戳 / 语言 / 精修后端 / 人数」在启动时读取，
其中字形和时间戳可以在运行中即时切换（纯文本变换，可逆）。

界面设计规范（照此改，不要临时加颜色）
--------------------------------------
Tkinter 没有 CSS，视觉质量全靠「少颜色 + 强层次 + 一致间距」堆出来。
本文件顶部的常量就是设计令牌，改一处即可全局生效：

* **层级**：背景 < 面板 < 控件 三级灰阶，靠明度差分层，不靠描边。
* **强调色只有一个**：绿（主操作）→ 红（停止/危险）→ 琥珀（提示）。
* **字体两级**：正文 10pt（Microsoft YaHei），标题 11pt bold。
  字幕区单独用 15pt，因为它才是这个软件的主角。
* **间距**：所有内边距取 4 的倍数；控件之间留 8~12px。
* **不用渐变、不用阴影、不用圆角**，改用「色块 + 明度差」表达层次。

模式：实时听录（抓系统声音）/ 听录文件（转写音视频），用顶部两个分段按钮切换。
"""
from __future__ import annotations

import math
import sys
import threading
import time
from pathlib import Path

from .outputs import stamp

ROOT = Path(__file__).resolve().parents[2]

# 说话人配色：黑底上的高对比色，label 从 1 开始轮转
SPEAKER_COLORS = ["#5ad7ff", "#ffb35c", "#c9a0ff", "#7dffa8", "#ff8fb1", "#ffe66b"]
DRAFT_COLOR = "#ffd166"
FINAL_FG = "#39ff88"
BG = "#0d0f12"
PANEL = "#161a20"
STATUS_BG = "#232a33"
FG = "#e8edf2"
MUTED = "#8c98a6"

# ---------- 设计令牌（新增）----------
# 背景层级：越靠上层越亮，靠明度差分层，不靠描边
CONTROL = "#2b3542"     # 可交互控件底
CONTROL_HI = "#3b4959"  # 控件悬停
LINE = "#0a0c0f"        # 分隔线（比面板更暗，用凹陷表达分层）
FAINT = "#6d7a89"       # 最弱信息（时间戳）

# 语义色：全局只有这三个 + 说话人配色，避免界面变成调色盘
ACCENT = "#1f7a45"      # 主操作（开始）
ACCENT_HI = "#2a9a58"
DANGER = "#a83636"      # 停止 / 危险
DANGER_HI = "#c74444"
WARN = "#ffb35c"        # 提示

# 字体两级：正文 10pt，标题 11pt bold；字幕区单独 15pt（它才是主角）
FONT_UI = ("Microsoft YaHei", 10)
FONT_UI_BOLD = ("Microsoft YaHei", 10, "bold")
FONT_TITLE = ("Microsoft YaHei", 11, "bold")
FONT_MONO = ("Consolas", 10)
FONT_CAPTION = ("Microsoft YaHei", 15)

MODE_LIVE = "live"
MODE_FILE = "file"

LANG_CHOICES = [("自动检测", "auto"), ("中文", "zh"), ("英文", "en")]
SCRIPT_CHOICES = [("简体中文", "simplified"), ("繁体中文", "traditional"),
                  ("保持原样", "as-is")]
FINAL_CHOICES = [("Whisper（推荐）", "whisper"),
                 ("Confucius 本地（免启服务）", "confucius-local"),
                 ("Confucius 服务", "confucius"),
                 ("仅流式草稿", "none")]
COUNT_CHOICES = [("自动", "0"), ("单人（不分人）", "1"),
                 ("2 人", "2"), ("3 人", "3"), ("4 人", "4"),
                 ("5 人", "5"), ("6 人", "6")]
DEVICE_CHOICES = [("自动", "auto"), ("GPU / CUDA", "cuda"), ("CPU", "cpu")]

# Whisper 档位：权重体积 vs 速度/准确率的取舍表。
#
# 为什么必须让用户自己选：faster-whisper 的权重在 HuggingFace 上，2.9 GB
# 远超 GitHub 单文件 100 MB 上限，**不能进仓库**。分享者只能告诉被分享者
# 「下哪个档位」，那就得把取舍讲清楚，而不是让对方猜 tiny 是怎么回事。
#
#   档位                体积      说明
#   tiny                72 MB     最省空间和算力，准确率最低，只适合试跑
#   base               145 MB     勉强能听
#   small              461 MB     中文可用，术语偶错，CPU 上也跑得动
#   medium            1.4 GB      折中
#   large-v3-turbo     1.6 GB     接近 large-v3 的效果，速度快数倍
#   large-v3           2.9 GB     最准；默认档位，有 GPU 时首选
WHISPER_TIERS = [
    ("large-v3", "2.9GB", "最准"),
    ("large-v3-turbo", "1.6GB", "快而准"),
    ("medium", "1.4GB", "折中"),
    ("small", "461MB", "轻量"),
    ("base", "145MB", "能用"),
    ("tiny", "72MB", "最省"),
]
WHISPER_CUSTOM = "自定义路径…"
WHISPER_CHOICES = [("%s %s" % (t, s), t) for t, s, _ in WHISPER_TIERS]
WHISPER_CHOICES.append((WHISPER_CUSTOM, "__custom__"))


class Choices:
    def __init__(self, pairs):
        self.pairs = pairs
        self.l2v = dict(pairs)
        self.v2l = {str(v): k for k, v in pairs}

    @property
    def labels(self):
        return [k for k, _ in self.pairs]

    def label(self, value):
        return self.v2l.get(str(value), self.pairs[0][0])

    def value(self, label):
        return self.l2v.get(label, self.pairs[0][1])


LANG = Choices(LANG_CHOICES)
SCRIPT = Choices(SCRIPT_CHOICES)
FINAL = Choices(FINAL_CHOICES)
COUNT = Choices(COUNT_CHOICES)
DEVICE = Choices(DEVICE_CHOICES)
WHISPER = Choices(WHISPER_CHOICES)


class LiveDisplay:
    """线程安全的显示状态。无 GUI 时也可作为纯状态容器使用。"""
    def __init__(self):
        self.lock = threading.RLock()
        self.finals = []            # list[dict(label, text, semantic, start, end)]
        self.partial = ""           # 当前这句的流式草稿
        self.partial_label = None
        self.stats = {}
        self.show_timestamps = False
        self.show_speaker = True
        self.notice = ""
        self.error = ""
        # 准备阶段（加载模型、开音频设备）的分阶段反馈；{} 表示不在准备中。
        # 与 stats 分开存：采集开始后 set_stats() 会整体替换 stats，不能把
        # 「本次输出文件名」这类会话级信息塞进去，否则会被冲掉。
        self.preparing = {}
        self.session = {}
        self.dirty = True
        self.title = "实时听录 · 系统音频 + 说话人"

    def set_preparing(self, stage, detail="", step=0, total=0):
        """上报准备进度。at 只在第一轮记录，用来显示「已等待 N 秒」。"""
        with self.lock:
            at = self.preparing.get("at") or time.monotonic()
            self.preparing = {"stage": str(stage or ""), "detail": str(detail or ""),
                              "step": int(step or 0), "total": int(total or 0),
                              "at": at}
            self.dirty = True

    def clear_preparing(self):
        with self.lock:
            self.preparing = {}
            self.dirty = True

    def set_session(self, info: dict):
        """本次会话的输出文件名/目录，供界面右下角显示，方便回找历史稿。"""
        with self.lock:
            self.session = dict(info or {})
            self.dirty = True

    def set_stats(self, snap: dict):
        with self.lock:
            self.stats = snap
            self.dirty = True

    def set_error(self, msg: str):
        with self.lock:
            self.error = msg or ""
            self.dirty = True

    def set_notice(self, msg: str):
        """状态条上的一句提示（模型下载进度之类），不影响错误显示。"""
        with self.lock:
            self.notice = msg or ""
            self.dirty = True

    def set_partial(self, text: str, label=None):
        with self.lock:
            self.partial = text or ""
            self.partial_label = label
            self.dirty = True

    def set_timestamps(self, value: bool):
        with self.lock:
            self.show_timestamps = bool(value)
            self.dirty = True

    def set_show_speaker(self, value: bool):
        """单人模式：窗口不显示「[说话人1]」前缀。"""
        with self.lock:
            self.show_speaker = bool(value)
            self.dirty = True

    def add_final(self, item: dict):
        with self.lock:
            self.finals.append(item)
            self.partial = ""
            self.partial_label = None
            self.dirty = True

    def update_final(self, uid: int, text: str, semantic=None):
        """句末精修回来后覆盖对应行的文字。"""
        with self.lock:
            for it in self.finals:
                if it.get("uid") == uid:
                    it["text"] = text
                    if semantic is not None:
                        it["semantic"] = semantic
                    it["refined"] = True
                    self.dirty = True
                    return

    def rebuild(self, render_fn, show_timestamps: bool):
        """字形/时间戳切换：用 semantic 重算已上屏的所有行。"""
        with self.lock:
            for it in self.finals:
                it["text"] = render_fn(it.get("semantic", it.get("text", "")))
            self.show_timestamps = bool(show_timestamps)
            self.dirty = True

    def reset(self):
        with self.lock:
            self.finals = []
            self.partial = ""
            self.partial_label = None
            self.stats = {}
            self.preparing = {}
            self.session = {}
            self.error = ""
            self.dirty = True

    def snapshot(self):
        with self.lock:
            return {
                "finals": [dict(it) for it in self.finals],
                "partial": self.partial,
                "partial_label": self.partial_label,
                "stats": dict(self.stats),
                "dirty": self.dirty,
                "show_timestamps": self.show_timestamps,
                "show_speaker": self.show_speaker,
                "notice": self.notice,
                "error": self.error,
                "preparing": dict(self.preparing),
                "session": dict(self.session),
            }

    def clear_dirty(self):
        with self.lock:
            self.dirty = False


def status_text(snap: dict) -> tuple:
    """返回 (状态条文字, 颜色)。"""
    if snap.get("error"):
        return "⚠ " + str(snap["error"])[:150], "#ff6b6b"
    peak = float(snap.get("peak", 0.0))
    db = 20 * math.log10(max(peak, 1e-5))
    bars = int(round(max(0.0, min(1.0, (db + 60) / 60.0)) * 12))
    bar = "▮" * bars + "▯" * (12 - bars)
    head = "电平 %s %5.1f dB" % (bar, db)
    if snap.get("warn"):
        return ("%s  ⚠ 连续 %.0fs 没收到声音 —— 多半是安全软件拦了麦克风，"
                "请点『允许』并勾选记住；或确认系统确实在播放声音"
                % (head, snap.get("silent_sec", 0)), "#ff6b6b")
    if snap.get("blocked_hint") and snap.get("sound_ratio", 1.0) < 0.01:
        return head + "  ⚠ 开流偏慢且几乎全程静音，检查授权弹窗", "#ffb35c"
    return head + "  ● 采集中  %s" % str(snap.get("device", ""))[:44], "#7dffa8"


# 高级设置字段：(key, 标签, 类型, 可选下拉)
ADVANCED_FIELDS = [
    ("whisper_model", "Whisper 模型（档位名或本地路径）", "str", None),
    ("whisper_device", "推理设备", "choice", DEVICE),
    ("whisper_compute", "计算精度", "str", None),
    ("download_root", "Whisper 权重目录（留空=工程内/自动下载）", "str", None),
    ("beam_size", "beam size", "int", None),
    ("confucius_url", "Confucius 服务地址", "str", None),
    ("confucius_dir", "Confucius 项目目录", "str", None),
    ("hot_word", "热词（逗号分隔）", "str", None),
    ("ollama_url", "Ollama 地址", "str", None),
    ("translate_model", "翻译模型（auto 自动挑）", "str", None),
    ("translate_timeout", "翻译超时(秒)", "float", None),
    ("speaker_threshold", "声纹相似度阈值", "float", None),
    ("speaker_names", "说话人命名，如 1=主持,2=嘉宾", "str", None),
    ("vad_min_silence", "断句静音(秒)", "float", None),
    ("silence_warn", "静音告警(秒，0=关)", "float", None),
    ("duration", "采集时长(秒，0=一直)", "float", None),
    ("outdir", "输出目录", "str", None),
    ("formats", "输出格式", "str", None),
    ("enroll", "载入声纹 voices.json", "str", None),
    ("save_voices", "保存声纹到", "str", None),
    ("simulate", "模拟回放 wav（调试用）", "str", None),
]


def _coerce(kind, raw):
    try:
        if kind == "int":
            return int(float(str(raw).strip() or 0))
        if kind == "float":
            return float(str(raw).strip() or 0)
        if kind == "bool":
            return str(raw).strip().lower() in ("1", "true", "yes", "on")
    except ValueError:
        return 0 if kind in ("int", "float") else False
    return str(raw)


def gui_main(display: LiveDisplay, controller, title=None):
    """controller: start(opts) / stop() / is_running() / apply_live(opts) /
    save(opts) / on_close() / settings / defaults.

    停止一律走 controller.stop()：控制器每轮听录会换一个独立的 Event，
    早期把某个 Event 对象直接传进来会在二次启动后指向上一轮，关窗口停不掉。
    """
    import tkinter as tk

    root = tk.Tk()
    root.title(title or display.title)
    root.geometry("1200x620+60+50")
    root.attributes("-topmost", True)
    root.configure(bg=BG)
    try:
        root.wm_attributes("-alpha", 0.96)
    except Exception:
        pass

    opts = dict(controller.settings)
    ready = {"v": False}
    state = {"running": False, "last_running": False, "stopping": False,
             "stop_at": 0.0, "mode": MODE_LIVE}

    # 状态条必须先 pack。若先放 expand+fill 的 Text，Tk 会把剩余空间分配完，
    # 后放的底部 Label 在某些 DPI 缩放下会被挤到不可见（实测踩过）。
    status = tk.Label(root, text="设置好选项后点『开始听录』", anchor="w",
                      font=FONT_MONO, bg=STATUS_BG, fg=MUTED,
                      padx=12, pady=6)
    status.pack(fill="x", side="bottom")

    # ---------- 顶栏：标题 + 模式切换 ----------
    header = tk.Frame(root, bg=PANEL, padx=12, pady=8)
    header.pack(fill="x", side="top")

    tk.Label(header, text="实时听录", bg=PANEL, fg=FG,
             font=FONT_TITLE).pack(side="left")

    # 竖分隔线：用 1px 宽的 Frame（不能用 pack 的 padx=(8,0)，那是间距不是线）
    tk.Frame(header, bg=LINE, width=1).pack(
        side="left", fill="y", padx=12)

    mode_frame = tk.Frame(header, bg=STATUS_BG)
    mode_frame.pack(side="left")

    mode_btns = {}

    def make_mode_btn(parent, key, caption):
        b = tk.Button(parent, text=caption, bg=STATUS_BG, fg=MUTED,
                      relief="flat", bd=0, padx=14, pady=4,
                      font=FONT_UI, cursor="hand2",
                      activebackground=CONTROL_HI, activeforeground=FG)
        b.pack(side="left")
        mode_btns[key] = b
        return b

    make_mode_btn(mode_frame, MODE_LIVE, "实时听录")
    make_mode_btn(mode_frame, MODE_FILE, "听录文件")

    tk.Label(header, text="抓电脑正在播的声音  ·  或直接转写音视频文件",
             bg=PANEL, fg=MUTED, font=FONT_UI).pack(side="left", padx=12)

    # ---------- 模式相关区域（文件路径 / 提示） ----------
    # 注意：tk.Frame 的 padx/pady 只接受整数，元组是 pack/grid 的语法。
    mode_area = tk.Frame(root, bg=BG, padx=12)

    file_bar = tk.Frame(mode_area, bg=BG)
    tk.Label(file_bar, text="媒体文件", bg=BG, fg=MUTED,
             font=FONT_UI).pack(side="left", padx=(0, 8))
    v_file = tk.StringVar(value=str(opts.get("media_file", "") or ""))
    file_entry = tk.Entry(file_bar, textvariable=v_file, bg=STATUS_BG, fg=FG,
                          insertbackground=FG, relief="flat",
                          font=FONT_UI, width=52)
    file_entry.pack(side="left", fill="x", expand=True)

    def browse_file():
        import tkinter.filedialog as fd
        p = fd.askopenfilename(
            title="选择要听录的音视频文件",
            filetypes=[
                ("音视频文件", "*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus "
                               "*.wma *.mp4 *.mkv *.mov *.avi *.webm *.flv *.wmv "
                               "*.ts *.m4v"),
                ("音频文件", "*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus *.wma"),
                ("视频文件", "*.mp4 *.mkv *.mov *.avi *.webm *.flv *.wmv *.ts *.m4v"),
                ("所有文件", "*.*"),
            ])
        if p:
            v_file.set(p)
            opts["media_file"] = p
            controller.save(dict(opts))

    browse_btn = tk.Button(file_bar, text="浏览…", bg=CONTROL, fg=FG,
                           relief="flat", bd=0, padx=12, pady=4,
                           font=FONT_UI, cursor="hand2",
                           activebackground=CONTROL_HI, command=browse_file)
    browse_btn.pack(side="left", padx=(8, 0))

    file_hint = tk.Label(mode_area,
                          text="支持 mp3 / m4a / wav / flac / mp4 / mkv 等常见格式"
                               "（视频自动抽音轨）。转写结果与实时听录一致，"
                               "按原文件名存到输出目录。",
                          bg=BG, fg=MUTED, font=FONT_UI,
                          anchor="w", justify="left")
    # 注意：这里不要立刻 pack。file_bar 稍后由 apply_mode() 统一 pack，
    # 如果这里先 pack，说明文字就会排到文件选择框**上面**（pack 顺序=上下顺序）。

    def apply_mode(mode):
        """切换实时 / 文件模式。"""
        state["mode"] = mode
        for k, b in mode_btns.items():
            active = (k == mode)
            b.configure(bg=(CONTROL if active else STATUS_BG),
                        fg=(FG if active else MUTED))
        if mode == MODE_FILE:
            # before=top：必须插在「标题栏」和「设置面板」之间。
            # Tk 的 pack 顺序决定上下位置，直接 pack 会掉到最底下。
            mode_area.pack(fill="x", side="top", before=top)
            file_bar.pack(fill="x")
            file_hint.pack(fill="x", pady=(6, 0))
            btn.configure(text="▶ 开始转写" if not state["running"] else "■ 停止")
        else:
            mode_area.pack_forget()
            btn.configure(text="▶ 开始听录" if not state["running"] else "■ 停止")
        # 记住上次用的模式，下次打开还在同一页
        if ready["v"]:
            opts["mode"] = mode
            controller.save(dict(opts))

    for k, b in mode_btns.items():
        b.configure(command=lambda kk=k: apply_mode(kk))

    # ---------- 设置面板 ----------
    top = tk.Frame(root, bg=PANEL, padx=12, pady=8)
    top.pack(fill="x", side="top")

    row1 = tk.Frame(top, bg=PANEL)
    row1.pack(fill="x")
    row2 = tk.Frame(top, bg=PANEL)
    row2.pack(fill="x", pady=(6, 0))

    widget_map = {}

    def styled_menu(parent, var, choices, width, key=None):
        om = tk.OptionMenu(parent, var, *choices.labels)
        om.configure(bg=STATUS_BG, fg=FG, activebackground=CONTROL_HI,
                     activeforeground=FG, highlightthickness=0, bd=0,
                     relief="flat", font=FONT_UI, width=width)
        om["menu"].configure(bg=STATUS_BG, fg=FG, activebackground=CONTROL_HI,
                             font=FONT_UI)
        om.pack(side="left", padx=(3, 10))
        if key:
            widget_map[key] = om
        return om

    def checkbox(parent, text, value, key=None):
        v = tk.BooleanVar(value=bool(value))
        cb = tk.Checkbutton(parent, text=text, variable=v, bg=PANEL, fg=FG,
                            selectcolor=CONTROL, activebackground=PANEL,
                            activeforeground=FG, font=FONT_UI,
                            bd=0, highlightthickness=0)
        cb.pack(side="left", padx=(3, 10))
        if key:
            widget_map[key] = cb
        return v

    def label(parent, text):
        tk.Label(parent, text=text, bg=PANEL, fg=MUTED,
                 font=FONT_UI).pack(side="left", padx=(3, 2))

    # 主操作按钮：唯一的高饱和色块，视觉锚点
    btn = tk.Button(row1, text="▶ 开始听录", font=FONT_UI_BOLD,
                    bg=ACCENT, fg="white", activebackground=ACCENT_HI,
                    relief="flat", bd=0, padx=16, pady=6, cursor="hand2")
    btn.pack(side="left", padx=(0, 16))

    v_translate = checkbox(row1, "译成中文", _coerce("bool", opts.get("translate", False)),
                           key="translate")
    label(row1, "字形")
    v_script = tk.StringVar(value=SCRIPT.label(opts.get("script", "simplified")))
    styled_menu(row1, v_script, SCRIPT, 10, key="script")
    v_timestamps = checkbox(row1, "时间戳", _coerce("bool", opts.get("timestamps", False)),
                            key="timestamps")

    label(row1, "音频语言")
    v_lang = tk.StringVar(value=LANG.label(opts.get("language") or "auto"))
    styled_menu(row1, v_lang, LANG, 9, key="language")
    label(row1, "精修后端")
    v_final = tk.StringVar(value=FINAL.label(opts.get("final", "whisper")))
    styled_menu(row1, v_final, FINAL, 22, key="final")
    label(row1, "人数")
    v_count = tk.StringVar(value=COUNT.label(str(opts.get("num_speakers", 0))))
    # 宽度要容得下最长的选项「单人（不分人）」（7 个全角字 ≈ 14 个字符宽），
    # 否则选项文字会被截断成「（不分人」，右侧还跟高级设置按钮挤在一起。
    styled_menu(row1, v_count, COUNT, 14, key="num_speakers")

    adv_btn = tk.Button(row2, text="高级设置…", bg=CONTROL, fg=FG, relief="flat",
                        bd=0, padx=12, pady=4, font=FONT_UI,
                        cursor="hand2", activebackground=CONTROL_HI)
    adv_btn.pack(side="left", padx=(3, 4))
    reset_btn = tk.Button(row2, text="恢复默认", bg=CONTROL, fg=FG, relief="flat",
                          bd=0, padx=12, pady=4, font=FONT_UI,
                          cursor="hand2", activebackground=CONTROL_HI)
    reset_btn.pack(side="left", padx=4)

    # Whisper 档位：权重不在仓库里（2.9GB 超 GitHub 单文件上限），
    # 所以必须给使用者一个明确的档位选择 + 一键下载入口，
    # 而不是让人对着「Could not find a version」猜该下哪个。
    label(row2, "Whisper 档位")
    _wm = str(opts.get("whisper_model") or "large-v3")
    v_whisper = tk.StringVar(
        value=WHISPER.label(_wm) if _wm in WHISPER.v2l else WHISPER_CUSTOM)
    styled_menu(row2, v_whisper, WHISPER, 20, key="whisper_model")
    dl_btn = tk.Button(row2, text="下载权重…", bg=CONTROL, fg=FG, relief="flat",
                       bd=0, padx=10, pady=4, font=FONT_UI,
                       cursor="hand2", activebackground=CONTROL_HI)
    dl_btn.pack(side="left", padx=(0, 12))

    hint = tk.Label(row2,
                    text="默认：英文原文、中文简体、无时间戳；人数选「单人」可跳过分人。"
                         "「译成中文」需本机 Ollama；运行中可切换字形和时间戳。",
                    bg=PANEL, fg=MUTED, font=("Microsoft YaHei", 9),
                    anchor="w", justify="left", wraplength=560)
    hint.pack(side="left", fill="x", expand=True, padx=(4, 0))

    txt = tk.Text(root, wrap="word", font=FONT_CAPTION,
                  bg=BG, fg=FINAL_FG, insertbackground=FINAL_FG,
                  padx=16, pady=14, relief="flat", spacing1=3, spacing3=6)
    txt.pack(expand=True, fill="both", side="top")
    for i, c in enumerate(SPEAKER_COLORS, 1):
        txt.tag_configure("spk%d" % i, foreground=c)
    txt.tag_configure("draft", foreground=DRAFT_COLOR)
    txt.tag_configure("stamp", foreground=FAINT)
    txt.tag_configure("hint", foreground=FAINT,
                      font=("Microsoft YaHei", 12))
    txt.tag_configure("speaker_name", font=FONT_CAPTION + ("bold",))
    txt.configure(state="disabled")

    EMPTY_HINT = ("设置好上方选项后点「▶ 开始听录」。\n"
                  "全程只有一个人讲话时，把「人数」选成「单人（不分人）」，"
                  "可跳过声纹分人、画面更干净。\n"
                  "首次采集若被安全软件拦截，请在弹窗点「允许」并勾选记住。\n"
                  "窗口会保留全部定稿文字，多人时用不同颜色区分说话人。\n")

    # ---------- 选项收集 ----------
    all_vars = [v_translate, v_script, v_timestamps, v_lang, v_final, v_count,
                v_whisper]

    def collect():
        out = dict(opts)
        out.update({
            "translate": bool(v_translate.get()),
            "script": SCRIPT.value(v_script.get()),
            "timestamps": bool(v_timestamps.get()),
            "language": LANG.value(v_lang.get()),
            "final": FINAL.value(v_final.get()),
            "num_speakers": int(COUNT.value(v_count.get())),
            "media_file": v_file.get(),
        })
        # 档位下拉选了具体档位才覆盖 whisper_model；
        # 选「自定义路径…」时保留高级设置里填的路径，不覆盖。
        wm = WHISPER.value(v_whisper.get())
        if wm and wm != "__custom__":
            out["whisper_model"] = wm
        return out

    def on_change(*_):
        if not ready["v"]:
            return
        cur = collect()
        opts.update(cur)
        if state["running"]:
            # 运行中只允许字形/时间戳变化，其余控件已禁用
            controller.apply_live(cur)
        else:
            controller.save(cur)

    for v in all_vars:
        v.trace_add("write", on_change)

    # ---------- 高级设置 ----------
    def open_advanced():
        win = tk.Toplevel(root)
        win.title("高级设置")
        win.configure(bg=BG)
        win.geometry("780x860+160+40")
        win.minsize(700, 520)
        win.transient(root)
        win.grab_set()
        tk.Label(win, text="高级选项（停止后可修改，改动会自动保存）",
                 bg=BG, fg=FG, font=("Microsoft YaHei", 11, "bold"),
                 anchor="w").pack(fill="x", padx=14, pady=(12, 8))

        # 表单字段有 20+ 个，加上模型说明后内容高度超过 800 px，
        # 在 1080p 屏幕上也放不下 -> 用 Canvas + 滚动条，
        # 这样小屏用户也能访问全部字段（而不是被「确定」按钮挤出可视区）。
        wrap = tk.Frame(win, bg=BG)
        wrap.pack(fill="both", expand=True, padx=14)
        canvas = tk.Canvas(wrap, bg=BG, highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(wrap, orient="vertical", command=canvas.yview,
                                  bg="#232a33", activebackground="#2e3844",
                                  troughcolor=BG, relief="flat", bd=0,
                                  width=12)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=BG)
        inner_win = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_config(_e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _on_inner_config)

        def _on_canvas_config(e):
            canvas.itemconfigure(inner_win, width=e.width)
        canvas.bind("<Configure>", _on_canvas_config)

        def _wheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        for w in (canvas, inner):
            w.bind("<MouseWheel>", _wheel)

        form = tk.Frame(inner, bg=BG)
        form.pack(fill="both", expand=True)
        fvars = {}
        for i, (key, caption, kind, choices) in enumerate(ADVANCED_FIELDS):
            tk.Label(form, text=caption, bg=BG, fg=MUTED,
                     font=("Microsoft YaHei", 10), anchor="w").grid(
                         row=i, column=0, sticky="w", padx=(0, 10), pady=3)
            if kind == "choice":
                var = tk.StringVar(value=choices.label(opts.get(key, "auto")))
                om = tk.OptionMenu(form, var, *choices.labels)
                om.configure(bg="#232a33", fg=FG, highlightthickness=0, bd=0,
                             relief="flat", width=34, font=("Microsoft YaHei", 10))
                om["menu"].configure(bg="#232a33", fg=FG, font=("Microsoft YaHei", 10))
                om.grid(row=i, column=1, sticky="ew", pady=3)
                fvars[key] = (var, kind, choices)
            else:
                var = tk.StringVar(value=str(opts.get(key, "") if opts.get(key) is not None else ""))
                tk.Entry(form, textvariable=var, bg="#1a2029", fg=FG,
                         insertbackground=FG, relief="flat",
                         font=("Microsoft YaHei", 10)).grid(
                             row=i, column=1, sticky="ew", pady=3)
                fvars[key] = (var, kind, None)
        form.columnconfigure(1, weight=1)

        # 模型来源说明：这是新用户最常问的问题（「Whisper 模型在哪？」），
        # 答案直接摆在设置窗口里，省得去翻文档。
        # 放在滚动区内部，跟着表单一起滚，避免在下方占死高度。
        whisper_note = tk.Frame(inner, bg=BG)
        whisper_note.pack(fill="x", pady=(10, 4))
        tk.Label(whisper_note, text="Whisper 模型从哪来？", bg=BG, fg=FG,
                 font=("Microsoft YaHei", 10, "bold"), anchor="w").pack(
                     fill="x", pady=(0, 3))
        tk.Label(whisper_note,
                 text="代码由 pip 安装；权重不在本仓库里（2.9 GB，超过 GitHub 单文件上限）。\n"
                      "· 已下到工程内 models/whisper/<档位>/ → 直接用，完全离线\n"
                      "· 否则首次使用时自动从 HuggingFace 下载；国内可设 HF_ENDPOINT=https://hf-mirror.com\n"
                      "· 想提前备好：主界面「Whisper 档位」选好，点「下载权重…」\n"
                      "  或命令行 python scripts/download_models.py --with-whisper small",
                 bg=BG, fg=MUTED, font=("Microsoft YaHei", 9),
                 anchor="w", justify="left").pack(fill="x")

        bool_frame = tk.Frame(win, bg=BG)
        bool_frame.pack(fill="x", padx=14, pady=6)
        v_nostream = tk.BooleanVar(value=bool(opts.get("no_streaming", False)))
        tk.Checkbutton(bool_frame, text="关闭流式草稿（整句说完才出字）",
                       variable=v_nostream, bg=BG, fg=FG, selectcolor="#2b3542",
                       activebackground=BG, activeforeground=FG, bd=0,
                       font=("Microsoft YaHei", 10)).pack(side="left")
        v_loop = tk.BooleanVar(value=bool(opts.get("loop_sim", False)))
        tk.Checkbutton(bool_frame, text="模拟回放循环", variable=v_loop,
                       bg=BG, fg=FG, selectcolor="#2b3542", activebackground=BG,
                       activeforeground=FG, bd=0,
                       font=("Microsoft YaHei", 10)).pack(side="left", padx=12)

        def apply(*_):
            for key, (var, kind, choices) in fvars.items():
                raw = var.get()
                opts[key] = choices.value(raw) if choices else _coerce(kind, raw)
            opts["no_streaming"] = bool(v_nostream.get())
            opts["loop_sim"] = bool(v_loop.get())
            # 高级设置里改了 Whisper 模型（可能是路径），同步回主界面的档位下拉
            _wm = str(opts.get("whisper_model") or "")
            v_whisper.set(WHISPER.label(_wm) if _wm in WHISPER.v2l
                          else WHISPER_CUSTOM)
            controller.save(dict(opts))
            win.destroy()

        tk.Button(win, text="确定", bg="#1f7a45", fg="white", relief="flat", bd=0,
                  padx=24, pady=6, font=("Microsoft YaHei", 10, "bold"),
                  command=apply).pack(pady=(4, 12))

    def restore_defaults():
        d = dict(controller.defaults)
        for key, val in d.items():
            opts[key] = val
        ready["v"] = False
        v_translate.set(bool(d.get("translate", False)))
        v_script.set(SCRIPT.label(d.get("script", "simplified")))
        v_timestamps.set(bool(d.get("timestamps", False)))
        v_lang.set(LANG.label(d.get("language") or "auto"))
        v_final.set(FINAL.label(d.get("final", "whisper")))
        v_count.set(COUNT.label(str(d.get("num_speakers", 0))))
        _wm = str(d.get("whisper_model") or "large-v3")
        v_whisper.set(WHISPER.label(_wm) if _wm in WHISPER.v2l else WHISPER_CUSTOM)
        ready["v"] = True
        controller.save(dict(opts))

    adv_btn.configure(command=open_advanced)
    reset_btn.configure(command=restore_defaults)

    # ---------- Whisper 权重下载 ----------
    # 权重不能进仓库，所以被分享者必须自己下一份。这里给一个后台下载入口，
    # 免得让人去翻 README 找命令、看不懂 HF 报错。
    dl_state = {"result": None}

    def download_whisper():
        from tkinter import messagebox

        tier = WHISPER.value(v_whisper.get())
        if not tier or tier == "__custom__":
            messagebox.showinfo(
                "Whisper 权重",
                "先在上方下拉里选一个档位（tiny / base / small / medium /\n"
                "large-v3 / large-v3-turbo），再点「下载权重…」。\n\n"
                "如果已经有一份本地模型了，去「高级设置」把「Whisper 模型」\n"
                "填成它的完整路径即可，不用下载。")
            return
        if controller.is_running():
            return
        display.set_notice("正在下载 Whisper 权重 %s …（可继续使用界面）" % tier)
        dl_btn.configure(state="disabled")

        def work():
            import subprocess
            script = ROOT / "scripts" / "download_models.py"
            cmd = [sys.executable, str(script), "--with-whisper", tier]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace",
                                   timeout=3600)
                out = (p.stdout or "") + (p.stderr or "")
                ok = p.returncode == 0
            except Exception as e:                       # noqa: BLE001
                out = "%s: %s" % (type(e).__name__, e)
                ok = False
            tail = "\n".join([ln for ln in out.strip().splitlines() if ln.strip()][-10:])
            dl_state["result"] = (ok, tier, tail)

        threading.Thread(target=work, name="whisper-dl", daemon=True).start()

    dl_btn.configure(command=download_whisper)

    # ---------- 运行控制 ----------
    LIVE_KEYS = ("script", "timestamps")   # 运行中可即时切换

    def mode_start_label():
        return "▶ 开始转写" if state["mode"] == MODE_FILE else "▶ 开始听录"

    def set_running_ui(running):
        state["running"] = running
        state["stopping"] = False
        btn.configure(text="■ 停止" if running else mode_start_label(),
                      bg=DANGER if running else ACCENT,
                      activebackground=DANGER_HI if running else ACCENT_HI,
                      state="normal")
        for key, w in widget_map.items():
            enabled = (not running) or key in LIVE_KEYS
            try:
                w.configure(state="normal" if enabled else "disabled")
            except tk.TclError:
                pass
        for b in (adv_btn, reset_btn):
            b.configure(state="disabled" if running else "normal")
        try:
            dl_btn.configure(state="disabled" if running else "normal")
        except tk.TclError:
            pass
        # 运行时锁定模式切换与文件选择，避免中途换模式导致状态错乱
        for b in mode_btns.values():
            b.configure(state="disabled" if running else "normal")
        try:
            file_entry.configure(state="disabled" if running else "normal")
            browse_btn.configure(state="disabled" if running else "normal")
        except tk.TclError:
            pass
        status.configure(fg=MUTED)

    def start_click():
        if state["running"]:
            if state["stopping"]:
                return                      # 已经在停了，别重复点
            state["stopping"] = True
            state["stop_at"] = time.monotonic()
            controller.stop()
            btn.configure(text="正在停止…", state="disabled")
            status.configure(text="正在停止…（等待最后几句定稿）", fg=MUTED)
            return

        cur = collect()
        if state["mode"] == MODE_FILE:
            path = (cur.get("media_file") or "").strip()
            if not path:
                status.configure(text="请先选择要听录的音视频文件", fg=WARN)
                return
            from pathlib import Path as _P
            if not _P(path).is_file():
                status.configure(text="文件不存在: %s" % path, fg="#ff6b6b")
                return
            controller.start_file(cur, path)
        else:
            controller.start(cur)
        set_running_ui(True)

    btn.configure(command=start_click)
    # 结束状态由 refresh() 在主线程轮询 controller.is_running() 恢复；
    # 不要从工作线程调用 root.after（Tkinter 非线程安全）。

    def on_close():
        controller.on_close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    def refresh():
        snap = display.snapshot()

        # 下载结果要弹窗，但必须回到主线程弹（Tkinter 非线程安全），
        # 所以工作线程只把结果放进 dl_state，这里轮询取走。
        if dl_state["result"] is not None:
            from tkinter import messagebox
            ok, tier, tail = dl_state["result"]
            dl_state["result"] = None
            try:
                dl_btn.configure(state="normal")
            except tk.TclError:
                pass
            if ok:
                display.set_notice("Whisper %s 已下载到 models/whisper/%s/" % (tier, tier))
                messagebox.showinfo(
                    "Whisper 权重",
                    "档位 %s 下载完成。\n\n位置：models/whisper/%s/\n\n"
                    "这个目录可以整个拷给同事，拷过去就完全离线可用。\n\n%s"
                    % (tier, tier, tail))
            else:
                display.set_notice("Whisper %s 下载失败（详见弹窗）" % tier)
                messagebox.showerror(
                    "Whisper 权重下载失败",
                    "档位 %s 下载失败。\n\n%s\n\n"
                    "国内网络可改用镜像：\n"
                    "python scripts/download_models.py --with-whisper %s "
                    "--endpoint https://hf-mirror.com" % (tier, tail, tier))

        if snap["dirty"]:
            txt.configure(state="normal")
            txt.delete("1.0", "end")
            for it in snap["finals"]:
                lab = int(it.get("label", 1))
                color_tag = "spk%d" % ((lab - 1) % len(SPEAKER_COLORS) + 1)
                if snap["show_timestamps"]:
                    txt.insert("end", "[%s] " % stamp(it.get("start", 0.0)), "stamp")
                if snap.get("show_speaker", True):
                    txt.insert("end", "[%s] " % it.get("name", "说话人%d" % lab),
                               (color_tag, "speaker_name"))
                txt.insert("end", (it.get("text") or "").strip() + "\n", color_tag)
            if snap["partial"]:
                tag = ("spk%d" % (((snap["partial_label"] or 1) - 1)
                                 % len(SPEAKER_COLORS) + 1)
                       if snap["partial_label"] else "draft")
                txt.insert("end", "▷ " + snap["partial"] + "\n", tag)
            if not snap["finals"] and not snap["partial"] and not snap["error"]:
                prep = snap.get("preparing") or {}
                if prep:
                    txt.insert("end", "⏳ 正在准备，请稍等…\n\n", "hint")
                    txt.insert("end", "(%d/%d) %s %s\n" %
                               (prep.get("step", 0), prep.get("total", 0),
                                prep.get("stage", ""), prep.get("detail", "")), "hint")
                else:
                    txt.insert("end", EMPTY_HINT, "hint")
            txt.configure(state="disabled")
            txt.see("end")
            display.clear_dirty()
        st_snap = dict(snap["stats"])
        st_snap["error"] = snap["error"]
        st, color = status_text(st_snap)
        # 文件模式的进度：显示阶段 + 细节，而不是电平条
        if snap["stats"].get("file_stage"):
            stage = snap["stats"].get("file_stage")
            detail = snap["stats"].get("file_detail", "")
            st = "文件听录 · %s%s" % (stage, ("  " + detail) if detail else "")
            color = (ACCENT_HI if stage == "done" else MUTED)
        if not state["running"] and not snap["stats"] and not snap["error"]:
            st = ("选择媒体文件后点『开始转写』" if state["mode"] == MODE_FILE
                  else "设置好选项后点『开始听录』")
            color = MUTED
        # 提示语（模型下载之类）只在空闲时显示，别盖掉运行状态
        notice = snap.get("notice", "")
        if notice and not state["running"] and not snap["stats"].get("file_stage"):
            st = notice
            color = MUTED
        # 准备阶段（加载模型 / 开音频设备）：明确告诉用户没死机
        prep = snap.get("preparing") or {}
        if prep and not snap["error"]:
            waited = time.monotonic() - prep.get("at", time.monotonic())
            st = ("⏳ 正在准备，请稍等…  (%d/%d) %s %s · 已 %.0fs" %
                  (prep.get("step", 0), prep.get("total", 0),
                   prep.get("stage", ""), prep.get("detail", ""), waited))
            color = WARN
        if state["stopping"]:
            waited = time.monotonic() - state["stop_at"]
            st = ("正在停止…（等待最后几句定稿，已 %.0fs）" % waited if waited < 12
                  else "仍在收尾，可关闭窗口强制退出（已等待 %.0fs）" % waited)
            color = MUTED
        sess = snap.get("session") or {}
        if sess.get("stem") and not snap["error"] and not prep:
            st += "   ·   本次文字稿: %s.txt" % sess["stem"]
        status.configure(text=st, fg=color)
        running = controller.is_running()
        if running != state["last_running"]:
            state["last_running"] = running
            set_running_ui(running)
        root.after(150, refresh)

    # 恢复上次用的模式（默认实时听录）
    start_mode = opts.get("mode") or MODE_LIVE
    if start_mode not in (MODE_LIVE, MODE_FILE):
        start_mode = MODE_LIVE
    apply_mode(start_mode)
    ready["v"] = True
    root.after(100, refresh)
    try:
        root.mainloop()
    finally:
        controller.stop()
