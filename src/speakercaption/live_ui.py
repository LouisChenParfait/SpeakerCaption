"""GUI controller: owns the capture worker thread and the persisted settings.

Tkinter must stay on the main thread, so the window (live_gui.gui_main) only
renders; this controller starts / stops LiveSession in a background thread.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

from .live_core import LiveSession, build_args, parse_args, _resolve_outdir
from .live_gui import LiveDisplay
from .session_stem import make_session_stem, safe_base
from .wave_meter import WaveMeter
try:
    # 曜石 × 火焰（PySide6 新版视图层）
    from .live_gui_qt import gui_main
    _UI_BACKEND = "qt"
except ImportError:
    # PySide6 未安装时回退到 Tkinter 旧版界面
    from .live_gui import gui_main
    _UI_BACKEND = "tk"
from .settings import UI_KEYS, load_settings, save_settings
from .text_pipeline import SCRIPT_CHOICES

_BOOL_KEYS = {"translate", "timestamps", "no_streaming", "loop_sim"}
_INT_KEYS = {"num_speakers", "beam_size"}
_FLOAT_KEYS = {"speaker_threshold", "vad_min_silence", "silence_warn",
               "duration", "translate_timeout"}


def _sanitize(opts: dict, defaults: dict) -> dict:
    out = {k: defaults[k] for k in UI_KEYS if k in defaults}
    for k, v in (opts or {}).items():
        if k not in UI_KEYS or v is None:
            continue
        try:
            if k in _BOOL_KEYS:
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
            elif k in _INT_KEYS:
                v = int(v)
            elif k in _FLOAT_KEYS:
                v = float(v)
            else:
                v = str(v)
        except (ValueError, TypeError):
            v = defaults.get(k)
        out[k] = v
    if out.get("script") not in SCRIPT_CHOICES:
        out["script"] = "simplified"
    if out.get("language") in ("", "None", "none", "null"):
        out["language"] = "auto"
    if out.get("final") not in ("none", "whisper", "confucius", "confucius-local"):
        out["final"] = "whisper"
    return out


class UIController:
    def __init__(self, base_args, settings, defaults, config_path,
                 cli_overrides=None, persisted=None):
        self.base_args = base_args
        # settings = 本次真正生效的值（落盘值 + 命令行当次覆盖），界面显示它。
        self.settings = dict(settings)
        self.defaults = defaults
        self.config_path = Path(config_path)
        # 命令行当次覆盖的键：只对本次启动生效，不写回 settings.json。
        # README 的承诺是「命令行参数只对当次启动覆盖」，GUI 初始化时如果把
        # 合并后的值存回去，用户下次开 GUI 就会莫名其妙接着上次的命令行参数跑
        # （例如 --simulate 指了测试 wav，之后一直回放测试音频而不是抓真实声音）。
        self._cli_overrides = dict(cli_overrides or {})
        # 真正落盘的值。调用方（run_ui）会显式传「settings.json 里的原值」；
        # 没传时兜底：命令行覆盖过的键一律回落到默认值，绝不把命令行值当落盘值。
        if persisted is not None:
            self._persisted = dict(persisted)
        else:
            self._persisted = {k: v for k, v in dict(settings).items()
                               if k not in self._cli_overrides}
            for k in self._cli_overrides:
                if k in self.defaults:
                    self._persisted[k] = self.defaults[k]
        # 本次启动时的生效值，用来判断「用户在界面上到底改没改」
        self._effective_at_start = dict(self.settings)
        self.display = LiveDisplay()
        # 真实音频波形总线：采集线程写、Qt 视图层 40ms 读。
        # 与 LiveDisplay 分开，是因为 display.snapshot() 每 150ms 才轮询一次，
        # 且 set_stats() 整体替换字典，波形这种高频数据不适合走它。
        self.wave_meter = WaveMeter()
        # 千万别叫 self.stop：实例属性会覆盖下面的 stop() 方法，
        # GUI 点「停止」时 controller.stop() 就变成调用 Event 对象，
        # 抛 TypeError: 'Event' object is not callable，采集永远停不下来。
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._session = None
        self._running = False

    def _apply(self, opts):
        """更新生效值并落盘。

        命令行当次覆盖的键：如果用户在界面上没动过（值仍等于本次启动时的
        生效值），就保持 settings.json 里原来的值不动；只有用户真的改了
        才写回。这样命令行既能在本次启动里生效，又不会污染下次的设置。
        """
        self.settings = _sanitize(opts, self.defaults)
        out = dict(self._persisted)
        for k, v in self.settings.items():
            if k in self._cli_overrides and v == self._effective_at_start.get(k):
                continue
            out[k] = v
        self._persisted = _sanitize(out, self.defaults)
        save_settings(self.config_path, self._persisted)
        self._effective_at_start = dict(self.settings)

    def start(self, opts):
        with self._lock:
            if self._running:
                return
            self._apply(opts)
            args = build_args(self.base_args, self.settings)
            args.no_gui = False
            # 每轮用独立事件，绝不 clear 复用：上一轮若还有线程没退干净，
            # clear 会把它的停止信号抹掉、让它继续跑并写同一个 display。
            self.stop_event = threading.Event()
            self.display.reset()
            self.display.set_timestamps(bool(self.settings.get("timestamps")))
            self.wave_meter.begin_preparing()
            # 先上「准备中」，否则构造会话到模型加载完这段窗口一片死寂
            self.display.set_preparing("正在初始化", "准备本次听录", 0, 5)
            try:
                session = LiveSession(args, display=self.display, stop=self.stop_event,
                                      wave_meter=self.wave_meter)
            except Exception as e:                        # noqa: BLE001
                # 构造失败也必须撤掉「准备中」，否则界面会永远卡在加载动画上
                msg = "启动失败 %s: %s" % (type(e).__name__, e)
                self.display.clear_preparing()
                self.wave_meter.finish()
                self.display.set_error(msg)
                print("[致命错误] " + msg, file=sys.stderr, flush=True)
                return
            self.display.set_session({
                "stem": session.session_stem,
                "outdir": str(session.outdir),
                "mode": "live",
                "formats": tuple(session.session_formats),
            })
            self._session = session
            self._running = True

        def target():
            try:
                session.run_pipeline()
            except Exception as e:                        # noqa: BLE001
                msg = "运行失败 %s: %s" % (type(e).__name__, e)
                self.display.clear_preparing()
                self.display.set_error(msg)
                print("[致命错误] " + msg, file=sys.stderr, flush=True)
                import traceback
                traceback.print_exc()
            finally:
                # 只改状态，不碰 Tk：窗口的 refresh() 在主线程轮询到状态变化后
                # 自己恢复控件。从工作线程调 root.after 会踩 Tkinter 线程不安全。
                self.display.clear_preparing()
                self.wave_meter.finish()
                with self._lock:
                    self._running = False

        thread = threading.Thread(target=target, name="pipeline", daemon=True)
        self._thread = thread
        thread.start()

    def stop(self):
        """请求停止采集；由 GUI 的「■ 停止」按钮调用。

        采集循环最多 0.2s 就能看到事件，但句末精修队列还需排空，
        因此窗口会先显示「正在停止…」，等 pipeline 真正结束再恢复控件。
        """
        self.stop_event.set()

    def is_running(self):
        with self._lock:
            return self._running

    def apply_live(self, opts):
        """运行中切换字形 / 时间戳：即时重算历史并重写输出文件。"""
        self._apply(opts)
        with self._lock:
            session, running = self._session, self._running
        if session and running:
            session.live_options.update(script=self.settings.get("script"),
                                        timestamps=self.settings.get("timestamps"))
            session.apply_settings()

    def save(self, opts):
        self._apply(opts)

    def on_close(self):
        self.stop()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=30)

    # ---------- 文件听录模式 ----------

    def start_file(self, opts, path):
        """听录一个媒体文件（音视频皆可）。

        复用同一个 display，文件转写出来的每句话也会上屏，
        所以实时模式和文件模式的文本区行为一致。
        """
        with self._lock:
            if self._running:
                return
            self._apply(opts)
            self.stop_event = threading.Event()
            self.display.reset()
            self.display.set_timestamps(bool(self.settings.get("timestamps")))
            self.wave_meter.begin_preparing()
            self._file_path = Path(path)
            self.display.set_preparing("正在初始化", "准备转写文件", 0, 6)
            try:
                outdir = _resolve_outdir(self.settings.get("outdir") or "live_out")
                outdir.mkdir(parents=True, exist_ok=True)
                formats = [f.strip() for f in
                           str(self.settings.get("formats", "txt,srt,json,rttm")).split(",")
                           if f.strip()]
                # 和实时模式同一套命名规则：每转写一次新建一份，绝不覆盖上次结果
                if getattr(self.base_args, "stem_suffix", "timestamp") != "timestamp":
                    stem = safe_base(self._file_path.stem)
                else:
                    stem = make_session_stem(self._file_path.stem, outdir, formats)
            except Exception as e:                        # noqa: BLE001
                msg = "启动文件听录失败 %s: %s" % (type(e).__name__, e)
                self.display.clear_preparing()
                self.wave_meter.finish()
                self.display.set_error(msg)
                print("[致命错误] " + msg, file=sys.stderr, flush=True)
                return
            self._file_outdir = outdir
            self._file_stem = stem
            self._file_formats = formats
            self.display.set_session({
                "stem": stem, "outdir": str(outdir), "mode": "file",
                "formats": tuple(formats),
            })
            self._running = True

        def target():
            try:
                self._run_file()
            except Exception as e:                        # noqa: BLE001
                msg = "文件听录失败 %s: %s" % (type(e).__name__, e)
                self.display.clear_preparing()
                self.display.set_error(msg)
                print("[致命错误] " + msg, file=sys.stderr, flush=True)
            finally:
                self.display.clear_preparing()
                self.wave_meter.finish()
                with self._lock:
                    self._running = False

        thread = threading.Thread(target=target, name="file-transcribe", daemon=True)
        self._thread = thread
        thread.start()

    def _run_file(self):
        from .file_mode import FileTranscriber

        s = self.settings
        display = self.display
        # 单人模式：窗口不显示「[说话人1]」前缀，和实时模式、txt/srt 口径一致。
        # LiveSession 会在建会话时设这个标志，但文件听录不走 LiveSession，
        # 必须在这里显式设，否则单人听录照样满屏【说话人1】。
        single = int(s.get("num_speakers", 0) or 0) == 1
        display.set_show_speaker(not single)

        STAGE_LABELS = {
            "probe": "读取文件信息", "decode": "解码音频",
            "diarize": "断句 / 分离说话人", "asr": "逐段转写",
            "write": "写出文件", "done": "完成",
        }
        STAGE_ORDER = ("probe", "decode", "diarize", "asr", "write")

        def progress(stage, detail=""):
            display.set_stats({"file_stage": stage, "file_detail": detail})
            if stage == "done":
                display.clear_preparing()
            else:
                # 文件转写是一次性跑完才上屏，中间字幕区一直是空的，
                # 必须同步推进准备动画，否则也像卡死。
                step = STAGE_ORDER.index(stage) + 1 if stage in STAGE_ORDER else 0
                display.set_preparing(
                    STAGE_LABELS.get(stage, stage), detail,
                    step, len(STAGE_ORDER))
            print("[文件] %s %s" % (stage, detail), flush=True)

        task = FileTranscriber(
            self._file_path,
            self._file_outdir,
            stem=self._file_stem,
            num_speakers=int(s.get("num_speakers", 0) or 0),
            single_speaker=bool(int(s.get("num_speakers", 0) or 0) == 1),
            asr=s.get("final", "whisper"),
            language=(None if str(s.get("language", "auto")).lower()
                      in ("auto", "none", "null", "") else s.get("language")),
            script=s.get("script", "simplified"),
            translate=bool(s.get("translate")),
            translate_model=s.get("translate_model", "auto"),
            ollama_url=s.get("ollama_url"),
            hot_word=s.get("hot_word", ""),
            confucius_dir=s.get("confucius_dir"),
            whisper_model=s.get("whisper_model", "large-v3"),
            whisper_device=s.get("whisper_device", "auto"),
            whisper_compute=s.get("whisper_compute", "auto"),
            beam_size=int(s.get("beam_size", 5) or 5),
            timestamps=bool(s.get("timestamps")),
            formats=self._file_formats,
            speaker_names=s.get("speaker_names", ""),
            progress=progress,
            wave_meter=self.wave_meter,
        )
        result = task.run()
        display.clear_preparing()   # 兜底：即使 progress("done") 没走到也能恢复界面

        # 结果上屏：每一段当作一条定稿，实时窗口那套着色逻辑直接复用
        from .outputs import speaker_name as _sname
        names = result["names"]
        for seg in result["segments"]:
            if not (seg.text or "").strip():
                continue
            display.add_final({
                "uid": id(seg),
                "label": int(seg.speaker or 1),
                "name": _sname(seg.speaker, names),
                "text": seg.text,
                "start": float(seg.start),
                "end": float(seg.end),
                "semantic": None,
            })
        paths = result["paths"]
        if result.get("single"):
            detail = "%d 段 / 单人模式 / 用时 %.1fs" % (
                len(result["segments"]), result["elapsed"])
        else:
            detail = "%d 段 / %d 位说话人 / 用时 %.1fs" % (
                len(result["segments"]), len(result["speakers"]), result["elapsed"])
        display.set_stats({
            "file_stage": "done",
            "file_detail": detail,
            "file_paths": paths,
        })
        print("[文件] 输出:", paths, flush=True)


def run_ui(base_args, explicit=None):
    if base_args.list_devices:
        from .live_audio import AudioSource
        print(AudioSource.list_devices())
        return 0
    explicit = dict(explicit or {})
    # --single-speaker 是纯命令行别名，_sanitize 只认识 UI_KEYS；
    # 转成 num_speakers=1 才能当次覆盖 settings.json 里保存的人数。
    if getattr(base_args, "single_speaker", False):
        explicit["num_speakers"] = 1
    defaults = {k: v for k, v in vars(parse_args([])).items() if k in UI_KEYS}
    saved = load_settings(base_args.config)
    # 两份值必须分开：
    #   persisted = settings.json 里真正存着的值（落盘用）
    #   effective = 本次启动真正生效的值（落盘值 + 命令行覆盖，界面显示它）
    persisted = _sanitize({**defaults, **saved}, defaults)
    effective = _sanitize({**defaults, **saved, **(explicit or {})}, defaults)
    ctl = UIController(base_args, effective, defaults, base_args.config,
                        cli_overrides=explicit, persisted=persisted)
    gui_main(ctl.display, ctl)
    return 0
