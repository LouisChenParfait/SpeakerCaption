"""在线说话人分配：VAD 切出的每一句，边到边判断「这是谁」。

与离线分离（OfflineSpeakerDiarization）的区别：离线方案要拿到整段音频再全局聚类，
这里必须**立刻**给当前这句一个标签，且不能等后面的人出现再回头改标签。

为什么不能「整段算一个声纹」
--------------------------
真实广播/直播音频里，VAD 切出来的段往往很长（实测本机一次抓取 13 段，
每段 7~19 秒）——电台节目几乎不停顿，0.45 秒的静音门槛根本触发不了，
只有节目里的换人处才有零点几秒的缝。

用整段（十几秒、可能含好几个人）算一个 192 维声纹，等于把几个人**平均**成一个
向量：这个向量既不像 A 也不像 B，后面每个人跟它比都是「离最近的中心最近」，
于是整场只剩【说话人1】。实测（tests/data/0-four-speakers-zh.wav 切 13 秒段）：
段 1 与段 4 的余弦只有 0.185（明显不同的人），却都被判成同一个说话人。

v2 的做法
---------
1. **切窗取声纹**：每段再切成 3 秒窗（步进 1.5 秒），每窗单独算一个声纹。
   3 秒正好是 CAM++ 的训练窗口长度，小窗比大段稳得多；一段里若混了两个人，
   至少有一半的窗是纯的。
2. **原型库**：每个说话人保留最多 4 个原型（同一个人的不同音色/信道/情绪），
   匹配时取「与所有说话人所有原型里最像的那个」，而不是只跟一个平均向量比。
3. **投票**：一段的标签 = 投票最多的说话人；只有过半的窗都认不出来，
   才算出现了新说话人。

实测依据（docs/DEVLOG.md §14）：官方双人音频 VAD 4 段，
同人相似度 0.72~0.77，跨人 0.30~0.38，故默认阈值 0.60 有充足余量。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 声纹窗长（秒）。3 秒是 CAM++ / 3D-Speaker 的训练窗长，短于此噪声大、长于此易混人。
DEFAULT_WIN_SEC = 3.0
DEFAULT_HOP_SEC = 1.5
# 每个说话人保留几个原型：覆盖「同一人的不同录音条件」
DEFAULT_MAX_PROTOS = 4
# 原型之间的合并阈值：>= 0.70 才认为是同一条声纹的变体，做 EMA 微调；
# 低于它但仍匹配上，就另立一个原型，避免把不同音色平均掉。
PROTO_MERGE = 0.70


class OnlineSpeakerTracker:
    def __init__(self, extractor, max_speakers: Optional[int] = None,
                 threshold: float = 0.60, min_speech_sec: float = 1.0,
                 ema: float = 0.08, names: Optional[Dict[int, str]] = None,
                 win_sec: float = DEFAULT_WIN_SEC, hop_sec: float = DEFAULT_HOP_SEC,
                 max_protos: int = DEFAULT_MAX_PROTOS):
        self.ext = extractor
        self.max_speakers = max_speakers
        self.threshold = float(threshold)
        self.min_samples = int(16000 * min_speech_sec)
        self.ema = float(ema)
        self.names = dict(names or {})
        self.win_sec = float(win_sec)
        self.hop_sec = float(hop_sec)
        self.max_protos = int(max_protos)
        #: label(int, 1-based) -> 该说话人的原型列表（L2 归一化）
        self.protos: Dict[int, List[np.ndarray]] = {}
        self.counts: Dict[int, int] = {}
        self.last_label: Optional[int] = None
        self._lock = threading.Lock()

    # ---------- 声纹 ----------

    def embedding(self, samples: np.ndarray) -> Optional[np.ndarray]:
        """返回 L2 归一化声纹；音频太短或模型未就绪返回 None。"""
        if samples is None or len(samples) < 1600:
            return None
        try:
            s = self.ext.create_stream()
            s.accept_waveform(16000, np.ascontiguousarray(samples, dtype=np.float32))
            s.input_finished()
            if not self.ext.is_ready(s):
                return None
            v = np.asarray(self.ext.compute(s), dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(v))
            if n < 1e-6:
                return None
            return v / n
        except Exception:
            return None

    def window_embeddings(self, samples: np.ndarray) -> List[np.ndarray]:
        """把一段音频切成若干窗，每窗一个声纹。

        段比窗长时按 hop 步进滑窗；最后不足一窗的尾巴丢掉（信息量太少）。
        """
        if samples is None or len(samples) < 1600:
            return []
        win = int(self.win_sec * 16000)
        hop = max(1, int(self.hop_sec * 16000))
        n = len(samples)
        if n <= win:
            e = self.embedding(samples)
            return [e] if e is not None else []
        out: List[np.ndarray] = []
        for off in range(0, n - win + 1, hop):
            e = self.embedding(samples[off:off + win])
            if e is not None:
                out.append(e)
        return out

    # ---------- 匹配 ----------

    def _match(self, e: np.ndarray) -> Tuple[Optional[int], float]:
        """在所有说话人的所有原型里找最像的一个，返回 (label, score)。"""
        best_label, best_score = None, -2.0
        for lab, protos in self.protos.items():
            for p in protos:
                s = float(np.dot(e, p))
                if s > best_score:
                    best_label, best_score = lab, s
        return best_label, best_score

    def _update(self, label: int, items: List[Tuple[float, np.ndarray]]) -> None:
        """把这一段里认得这个人的窗并进他的原型库。"""
        protos = self.protos.setdefault(label, [])
        for _score, e in items:
            best_i, best_s = -1, -2.0
            for i, p in enumerate(protos):
                s = float(np.dot(e, p))
                if s > best_s:
                    best_i, best_s = i, s
            if best_i >= 0 and best_s >= PROTO_MERGE:
                # 同一条声纹的变体 → 指数滑动平均，让原型缓慢自适应
                merged = protos[best_i] * (1 - self.ema) + e * self.ema
                protos[best_i] = merged / (np.linalg.norm(merged) + 1e-9)
            elif len(protos) < self.max_protos:
                protos.append(e)
            else:
                # 原型满了：换掉与它最不像的那个（保留最有辨识度的音色）
                worst = min(range(len(protos)),
                            key=lambda i: float(np.dot(e, protos[i])))
                protos[worst] = e
        self.counts[label] = self.counts.get(label, 0) + 1

    def _new_speaker(self, items: List[Tuple[float, np.ndarray]]) -> int:
        """新建一个说话人，用这些窗做种子原型。"""
        label = max(self.protos.keys(), default=0) + 1
        protos: List[np.ndarray] = []
        for _score, e in items:
            if not any(float(np.dot(e, p)) >= 0.9 for p in protos):
                protos.append(e)
            if len(protos) >= self.max_protos:
                break
        if not protos and items:
            protos = [items[0][1]]
        self.protos[label] = protos or []
        self.counts[label] = 1
        return label

    # ---------- 主入口 ----------

    def assign(self, samples: np.ndarray) -> Tuple[int, float, bool]:
        """给一句音频分配说话人。

        返回 (label, score, is_new)。label 从 1 开始。
        短句会沿用上一句标签并给出 score=-1（表示未做声纹判断）。
        """
        with self._lock:
            if samples is None or len(samples) < self.min_samples:
                label = self.last_label or 1
                self.last_label = label
                if label not in self.protos:
                    self.protos[label] = []       # 占位，等有效声纹来填
                return label, -1.0, False

            embs = self.window_embeddings(samples)
            if not embs:
                label = self.last_label or 1
                self.last_label = label
                if label not in self.protos:
                    self.protos[label] = []
                return label, -1.0, False

            # 每个窗投票：认得出谁就投给谁，认不出算「未知」
            votes: Dict[int, List[Tuple[float, np.ndarray]]] = {}
            unknown: List[Tuple[float, np.ndarray]] = []
            for e in embs:
                lab, score = self._match(e)
                if lab is not None and score >= self.threshold:
                    votes.setdefault(lab, []).append((score, e))
                else:
                    unknown.append((score, e))

            total = len(embs)
            top = None
            if votes:
                top = max(votes.items(),
                          key=lambda kv: (len(kv[1]), max(s for s, _ in kv[1])))

            is_new = False
            if top is not None and len(top[1]) * 2 >= total:
                # 过半的窗都认得这个人 → 就是他
                label = top[0]
                score = max(s for s, _ in top[1])
                self._update(label, top[1])
            elif self.max_speakers is None or len(self.protos) < self.max_speakers:
                # 大半的窗谁都不认 → 出现了新说话人
                seed = [e for _s, e in unknown] if unknown else list(embs)
                label = self._new_speaker([(0.0, e) for e in seed])
                score = 0.0
                is_new = True
            else:
                # 人数已满：不能再新建，强制归到最像的那个
                if top is not None:
                    label = top[0]
                    score = max(s for s, _ in top[1])
                    self._update(label, top[1])
                else:
                    label = max(self.protos.keys(), default=1)
                    score = 0.0

            self.last_label = label
            return label, (score if score > -2 else 0.0), is_new

    # ---------- 展示 / 持久化 ----------

    def display_name(self, label: int) -> str:
        return self.names.get(label, "说话人%d" % label)

    def num_speakers(self) -> int:
        return len([1 for v in self.protos.values() if v])

    @property
    def centroids(self) -> Dict[int, np.ndarray]:
        """主原型视图（第一个原型），兼容旧的 save/load 与外部引用。"""
        return {lab: v[0] for lab, v in self.protos.items() if v}

    def save(self, path) -> Path:
        """把声纹原型存成 JSON，供下次 --enroll 复用，保持跨次标签一致。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "dim": int(self.ext.dim),
            "threshold": self.threshold,
            "speakers": [
                {"label": lab, "name": self.display_name(lab),
                 "count": self.counts.get(lab, 0),
                 "embedding": [round(float(x), 7) for x in protos[0]],
                 "embeddings": [[round(float(x), 7) for x in p] for p in protos]}
                for lab, protos in sorted(self.protos.items()) if protos
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load(self, path) -> int:
        """载入已注册声纹。返回成功载入的说话人数。"""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError("声纹文件不存在: %s" % path)
        d = json.loads(path.read_text(encoding="utf-8"))
        n = 0
        for item in d.get("speakers", []):
            vectors = []
            # v2 存了多个原型；v1 只有一条
            for raw in item.get("embeddings") or []:
                v = np.asarray(raw, dtype=np.float32)
                if v.size == self.ext.dim:
                    vectors.append(v / (np.linalg.norm(v) + 1e-9))
            if not vectors:
                v = np.asarray(item.get("embedding", []), dtype=np.float32)
                if v.size == self.ext.dim:
                    vectors.append(v / (np.linalg.norm(v) + 1e-9))
            if not vectors:
                continue
            lab = int(item["label"])
            self.protos[lab] = vectors
            self.counts[lab] = int(item.get("count", 0))
            if item.get("name"):
                self.names[lab] = item["name"]
            n += 1
        return n


class SingleSpeakerTracker:
    """单人直播 / 单人播客：完全不做分人，每句恒定 label=1。

    与 OnlineSpeakerTracker 接口对齐，但不加载 CAM++ 模型、不算 embedding，
    也就不会出现「同一个人的短句被误判成新说话人」。适合全程只有一个人讲话
    的直播、讲座、录屏旁白——这也是最常见的使用场景。
    """

    def __init__(self, name: str = ""):
        # 保持 None，让 GUI 的黄色草稿 ▷ 一直用草稿色而不是说话人色
        self.last_label = None
        self.names = {1: name} if name else {}
        self.protos = {}
        self.counts = {}

    def assign(self, samples) -> Tuple[int, float, bool]:
        return 1, 0.0, False

    def display_name(self, label: int = 1) -> str:
        return self.names.get(1) or "说话人1"

    def num_speakers(self) -> int:
        return 1

    def save(self, path) -> Path:
        """单人模式没有声纹可存；不写文件，仅保持接口兼容。"""
        return Path(path)

    def load(self, path) -> int:
        return 0


def build_extractor(model_path, num_threads: int = 2):
    import sherpa_onnx
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_path), num_threads=num_threads, provider="cpu")
    if not cfg.validate():
        raise RuntimeError("声纹模型配置无效: %s" % model_path)
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
