# 开发调试记录 / DEVLOG

日期：2026-11-09　｜　工程：`SpeakerCaption`　｜　版本：0.1.0

本文件记录从零搭建到验收的全过程、实测数据与踩过的坑。**保留它是为了让第二台机器
能复现，也为了说明每一个默认参数为什么是现在这个值。**

起因：因为本人在工作中需要在听直播时马上做出一些回应，所以我想截取直播中的文字以便及时反馈。另外以前也常需要对一些直播播客的部分内容向朋友做出及时的推荐，我就想应该做这样一个工具来帮助我处理。我首先做了同类产品的调研，感觉类似产品不能满足我的需求，于是开始做这个工具。首先想到的是利用whisper，但我担心它在中文处理上有些弱，所以加上了Confucius4-R2T2，因为它在中文处理上更好一些。


---

## 1. 目标与边界

- 需求：听录双人播客 / 多人访谈时能分辨不同说话人；同时工程要能开源到 GitHub，
  不能只在这台 Windows 机器上跑。
- 决策：**新建独立工程目录**（`SpeakerCaption`，与原 Confucius4-R2T2
  项目平级），不改动原项目、不把原项目的编译模块带进来。
- 定位：**离线文件听录**。实时系统声音字幕是另一套（WASAPI 回环 + 流式
  Paraformer），两者通过 `rttm/json` 可对接，但互不依赖。
- 许可前提（先查后写）：Confucius4-R2T2 代码 Apache-2.0，但**模型权重是网易模型
  使用许可协议（非 OSI）**，因此开源版默认栈不依赖它，只提供一个可选 HTTP 后端。

## 2. 环境探测

`scripts/check_env.py` 是正式交付的自检脚本，可重复运行。两个环境对照：

| 项 | 原项目 `py312` | 新工程 `.venv`（纯净） |
|---|---|---|
| Python | 3.12.7 | 3.12.7 |
| sherpa-onnx | 2.0.1 | 2.0.1 |
| onnxruntime | 1.21.0 | 1.31.1 |
| faster-whisper | 1.2.1 | 1.2.1 |
| ctranslate2 | （随 faster-whisper） | 4.8.5 |
| torch | 2.8.0+cu128 | **未安装**（不需要） |
| CUDA 设备 | RTX 4090 D | 1 个（ctranslate2 探测） |

结论：去掉 torch 依赖后，纯净 venv 只有约 200 MB 就能跑 GPU 转写。
GPU 探测改成 `ctranslate2.get_cuda_device_count()`，失败再退回 torch，最后退回 CPU。

## 3. 网络与模型下载

GitHub release 连通性实测：

| 资产 | 大小 | 实测下载耗时 |
|---|---|---|
| `sherpa-onnx-pyannote-segmentation-3-0.tar.bz2` | 6,958,444 B | 2.5 s |
| `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` | 28,281,164 B | 97.4 s（约 0.28 MB/s） |
| `1-two-speakers-en.wav` | 512,044 B | 1.5 s |
| `0-four-speakers-zh.wav` | 1,819,586 B | 1.8 s |

分段模型压缩包内含 `LICENSE`（MIT / CNRS）、`README.md`（注明由
`pyannote/segmentation-3.0` 转换）、`model.onnx` 5.72 MB、`model.int8.onnx` 1.47 MB，
以及若干转换脚本。解压时 `tarfile.extractall` 在 Python 3.12+ 有
DeprecationWarning，已改为 `filter="data"`（旧版本 Python 用 try/except 回退）。

下载进度条最初用 `\r` 刷新，在 PowerShell 里被展开成上百行；已改成每 25% 打印一行。

## 4. sherpa-onnx API 适配（踩坑 2：稀疏标签）

`sherpa_onnx.OfflineSpeakerDiarization` 是 C++ 绑定，没有可读的 `inspect.signature`。
按官方 `python-api-examples/offline-speaker-diarization.py` 构造：

```python
cfg = OfflineSpeakerDiarizationConfig(
    segmentation=OfflineSpeakerSegmentationModelConfig(
        pyannote=OfflineSpeakerSegmentationPyannoteModelConfig(
            model=..., window_shift_ratio=0.1)),
    embedding=SpeakerEmbeddingExtractorConfig(model=...),
    clustering=FastClusteringConfig(num_clusters=N, threshold=T),
    min_duration_on=0.3, min_duration_off=0.5)
sd = OfflineSpeakerDiarization(cfg)      # sd.sample_rate == 16000
result = sd.process(audio).sort_by_start_time()   # r.start / r.end / r.speaker
```

输入必须是 **16 kHz、float32 一维、连续内存**的 numpy 数组。

自动聚类时标签**不连续**：实测出现过 `[0, 1, 3, 7, 8]`、`[0, 1, 2, 4]`。
若直接输出，我会看到「说话人7」。因此 `diarize.py` 里 `_normalise_labels()`
按首次出现顺序重映射为 1..N，这是必须的一步，不是美化。

## 5. 【关键发现】int8 分段模型会合并说话人（踩坑 3）

第一次冒烟用 `model.int8.onnx`，结果为：

- 双人英文：4 段 / 2 人 ✓
- 四人中文：10 段 / **3 人** ✗（把两个人并成了一类）

于是做了 10 组参数扫描（`_diag.py`，临时诊断脚本，已清理）：

| segmentation | 聚类 | 段数 | 标签 | 耗时 |
|---|---|---|---|---|
| int8 | k=4 | 10 | `[0,1,2]` ← 少 1 人 | 3.3 s |
| int8 | auto th=0.5 | 10 | `[0,1,3,7,8]` ← 过分割 | 3.4 s |
| int8 | auto th=0.7 | 10 | `[0,1,2,5]` | 3.4 s |
| int8 | auto th=0.9 | 10 | `[0,1,2]` ← 欠分割 | 3.3 s |
| int8 | k=4, min_on=0.1 | 10 | `[0,1,2]` | 3.3 s |
| **fp32** | **k=4** | 10 | **`[0,1,2,3]` ✓** | **3.7 s** |
| fp32 | auto th=0.5 | 10 | `[0,1,2,7]`（4 类） | 3.7 s |
| fp32 | auto th=0.7 | 10 | `[0,1,2,4]`（4 类） | 3.8 s |
| fp32 | auto th=0.9 | 10 | `[0,1,2]` ← 人又被合并 | 3.8 s |
| fp32 | k=4, min_on=0.1 | 10 | `[0,1,2,3]` ✓ | 3.7 s |

结论：

1. **默认改用 fp32 `model.onnx`**（5.72 MB）。速度只慢约 11%，但四人样本从 3 人
   恢复为正确 4 人。int8 保留为 `--seg-int8`，并在 README 明确警告。
2. 自动估计阈值默认 **0.7**：0.5 会把同一个人拆成多类，0.9 会把不同人合并。
3. 已知人数时显式 `--num-speakers N` 最稳。

## 6. ASR 后端与中文质量（踩坑 4：繁简混出）

| 后端 | 实测 | 特点 |
|---|---|---|
| `whisper` large-v3 / cuda / float16 | 双人 4 段 0.99 s；四人 10 段 2.41~3.83 s | 快、跨平台、MIT；中文偶发繁体、无标点 |
| `confucius`（可选，HTTP） | 四人 10 段 **42.37 s**，零失败 | 带自然标点、热词生效（"雷军"命中）；约慢 10~15 倍 |
| `none` | — | 只出 rttm/json，方便对接别的 ASR |

Whisper 中文实测出现 `這是我第四次`、`我沒有停下對非凡的追求`。加了可选
`--simplify`（opencc `t2s`），同一段音频复跑后变为 `这是我第四次`、
`我没有停下对非凡的追求`，确认生效。opencc 缺失时只警告一次，不中断。

Confucius 后端实现要点（沿用原项目已验证的结论）：

- 不走 `gradio_client`（它曾返回空列表并抛 `list index out of range`），直接走
  `POST /upload` → `POST /call/do_text` → `GET .../event_id` 读 SSE。
- `httpx.Client(trust_env=False)` + 给 `NO_PROXY` 追加 `127.0.0.1,localhost,::1`。
  环境变量代理会把 localhost 请求也塞给代理，导致 404/空结果，这是原项目踩过的坑。
- 挂载前缀自动探测 `/gradio_api` 或空；本工程实测探测到 `/gradio_api`。
- 只含 HTTP 客户端，不含厂商源码、编译模块或权重。

## 7. 独立环境验证

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt \
    -i https://mirrors.cloud.tencent.com/pypi/simple
```

30 秒装完，无报错。本机 pip 全局配置指向清华源且返回 403，故显式指定腾讯云镜像
（此点写进 README 的部署说明由使用者自行选择镜像）。装在纯净 venv 里的包共 34 个，
其中 `sherpa-onnx 2.0.1`、`faster-whisper 1.2.1`、`onnxruntime 1.31.1`、
`ctranslate2 4.8.5`、`opencc-python-reimplemented 0.1.7`、`soxr 1.1.0`。

`ctranslate2.get_cuda_device_count()` 在纯净 venv 里返回 1（系统 CUDA DLL 可用），
所以无 torch 也能 GPU 转写。完整 CLI 在该 venv 下重跑四人中文样本，10.95 s 完成，
与依赖原项目 py312 环境的 10.37 s 基本一致 → **工程已自包含**。

## 8. 端到端验收

| 用例 | 命令要点 | 结果 |
|---|---|---|
| 双人英文 / 仅分离 | `--asr none --num-speakers 2` | 4 段 / 2 人，0.57 s，27.9x 实时 |
| 四人中文 / 仅分离 | `--asr none --num-speakers 4` | 10 段 / 4 人，3.76 s，15.1x 实时 |
| 双人英文 / Whisper | `--asr whisper --language en` | 4 句全部正确（Harvard sentences），总 7.40 s |
| 四人中文 / Whisper | `--asr whisper --language zh` | 10 句通顺，总 10.37 s |
| 四人中文 / Whisper + 简繁 + 时间戳 | 纯净 venv | 繁体修正，总 10.95 s |
| 四人中文 / Confucius | `--asr confucius --hot-word 雷军,测试音频` | 10 段零失败、带标点、热词命中，总 46.50 s |
| 自动估计人数 | `--num-speakers -1 --threshold 0.7` | 4 类（标签 `[0,1,2,4]` → 重映射 1..4）|
| 说话人改名 | `--speaker-names "1=主持人,..."` | txt/srt/json 均生效 |
| 冒烟测试 | `.venv/Scripts/python.exe tests/test_smoke.py` | 3 个用例全 PASSED |

验证完 Confucius 链路后已主动停掉 7861 服务，`PORT_7861_FREE`，端口还给使用者。

## 9. 输出格式抽查

`txt`（`--timestamps` 开）、`srt`、`json`、`rttm` 四个文件全部人工核对：

- `txt`：`[00:00.3] 说话人1：...`；不开时间戳则只有 `说话人1：...`（我偏好默认关）。
- `srt`：`00:00:00,318 --> 00:00:06,865` + `[说话人1] ...`，时间码与分段一致。
- `json`：含 `version / source / duration / num_speakers_detected / 模型名 /
  asr_backend / diarize_seconds / speakers[].total_seconds / segments[]`。
- `rttm`：`SPEAKER <stem> 1 <start> <duration> <NA> <NA> speaker_01 <NA> <NA>`，
  标准格式，可导入标注工具。
- 四种文件固定同名，每次重建覆盖，不会一句一个文件。

## 10. 已知限制 / 后续

- 重叠语音不拆多路；抢话处 ASR 可能丢字。
- 说话人身份只在单文件内一致（跨文件同人需做说话人注册）。
- 6 人以上、远场麦克风时自动估计偏差变大，务必显式给人数。
- Confucius 后端依赖我自行部署 7861 服务并接受网易模型许可；开源版默认不用它。
- 未做：Gradio 网页界面、与实时字幕窗口对接、批量目录处理。

## 11. Windows 启动器（踩坑 5：cmd 括号陷阱）

`run.bat` 最初带 `--version` 调用时 exit 1、零输出；PowerShell 直接调用、
`cmd /c`、Python `subprocess` 捕获都是同样现象。用一个最小 `_probe2.bat`
逐行打点后定位：`if "%~1"=="" ( ... )` 括号块里有一行
`echo ... (or see README.md)`，**未转义的 `)` 被 cmd 当作块的结束**，
后续语句解析错乱。去掉括号后复验：

- 无参数 → 显示 usage，exit 1 ✓
- `--version` → `SpeakerCaption 0.1.0`，exit 0 ✓
- 带真实音频 `--asr none --num-speakers 2` → exit 0，4 段 / 2 人，0.98 s ✓

另两条 Windows 经验（沿用原项目结论）：

- `.bat` 内容必须**纯 ASCII + CRLF**；内容里出现中文会在 GBK 控制台碎片化成命令。
- 不在 bat 里调 `chcp 65001`：它会重置控制台句柄，在被 PowerShell / 管道捕获时
  吞掉 stdout。改为设置 `PYTHONUTF8=1` + `PYTHONIOENCODING=utf-8`，
  cmd 窗口里若中文乱码，让我自行执行 `chcp 65001`。

## 12. 过程文件清理清单（已执行）

| 文件/目录 | 处理 |
|---|---|
| `_diag.py`（参数扫描临时脚本） | 删除，结论已写入本文 §5 |
| `out/`（验收产物） | 清空，格式样例写进 `examples/README.md` |
| `models/*.tar.bz2`（下载压缩包） | 删除，解压结果保留，脚本可重下 |
| `__pycache__/`、`.pyc` | 删除 |
| `.venv/pip 缓存`、`.part` | 不存在则忽略 |
| `tests/data/*.wav` | **保留**（本地验收素材，`.gitignore` 排除，不随仓库分发） |
| `models/*.onnx` + 包内 `LICENSE` | **保留**（本地运行依赖，`.gitignore` 排除） |

## 13. 开源前竞品调研（2026-11-09）

我要求确认「别跟已有 GitHub 项目撞车」。完整结果落在 `docs/RELATED_WORK.md`，
这里只记方法、结论与踩到的工具坑。

- **方法**：GitHub Search API `search/repositories` 按 star 降序，多组关键词交叉；
  重点候选逐个拉 `/repos/{owner}/{repo}` 看 star/许可/`pushed_at`，再读 raw README 核实能力。
- **工具坑（值得记）**：`api.github.com` 匿名配额 60 次/小时且按出口 IP 计。
  走系统代理 `18081` 打完第一轮 4 次搜索就 `API rate limit exceeded`；
  环境变量代理 `22641` 直接连不上；**`curl --noproxy '*'` 直连 api.github.com 反而通**，
  且是独立配额。以后查 GitHub API 优先试直连，别默认套代理。
  另：`q=中文` 检索基本无结果，GitHub 中文索引弱，别据此判断「中文赛道没人做」。
- **结论**：大类红海（`speaker diarization` 2875 个仓库），但交叉检索
  `Confucius R2T2 diarization`=0、`sherpa 3dspeaker diarization`=0、
  `diarization rttm srt whisper`=1（2 star）、`SpeakerCaption`=0；
  GitHub 我/组织名与 PyPI `speakercaption` / `speaker-caption` 均 404 → **名字可用**。
- **定位调整**：README 新增「和同类项目的区别」，主打
  「无 HF token / 无 torch / 无 GPU / 35MB / 四格式含 rttm / 许可干净」，
  不宣称 SOTA。明确避开实时流式（WhisperLiveKit 11k、meetily 32k）与 GUI 平台
  （aTrain / MioSub / OpenTranscribe，均为 AGPL/GPL）。
- **唯一技术威胁**：`OpenMOSS/MOSS-Transcribe-Diarize`（端到端 0.9B，中文 SOTA，
  SRT/ASS/JSON + 热词），但要 transformers 5.x + vLLM/SGLang + CUDA，Windows 不可行，
  属于「硬件门槛相反」而非直接撞车。

> 补注（同日稍后）：我随后要求增加**实时系统声音听录**，见 §14。
> §13 中「避开实时流式」指的是**不对标流式 ASR 服务器 / 低延迟服务框架**
> （WhisperLiveKit、meetily 那条赛道），本项目新增的 `live.py` 是
> 「本机抓系统回环声音 → 本地字幕窗口」的单机 CLI，形态与它们不冲突。

## 14. 实时听录（live.py）：系统声音 → 流式出字 → 在线分人

我明确要的不是「处理一个 mp3」，而是**网络会议 / 直播双人播客中，直接听电脑正在播的声音**。
因此把原项目 `live_caption.py` 的 WASAPI 回环链路与 SpeakerCaption 的 CAM++ 声纹合流，
作为本工程第二个入口；原项目一行未动。

### 选型依据（先探针，后实现）

`sherpa-onnx 2.0.1` 提供 `VoiceActivityDetector`（`accept_waveform` / `current_segment` /
`front` / `pop`）、`OnlineRecognizer.from_paraformer`、`SpeakerEmbeddingExtractor`。
离线用的 `OfflineSpeakerDiarization` 必须拿整段音频全局聚类，**不能用于实时**。

写 `_probe_live.py` 用官方双人英文 16s 音频同时验证三者，结果决定性：

| 段 | VAD 时长 | 与段0 | 与段1 | 与段2 | 与段3 |
|---|---|---|---|---|---|
| 0 | 1.88s | 1.000 | **0.773** | 0.323 | 0.376 |
| 1 | 2.14s | 0.773 | 1.000 | 0.303 | 0.318 |
| 2 | 2.26s | 0.323 | 0.303 | 1.000 | **0.724** |
| 3 | 2.58s | 0.376 | 0.318 | 0.724 | 1.000 |

同人 0.72~0.77、跨人 0.30~0.38 → 阈值 **0.60** 余量充足。16s 音频整链路 0.92s（17x 实时）。
**结论：不必等整段、不必预注册，逐句声纹增量中心即可可靠分人。**

### 架构

```
AudioSource(wasapi / sd / file) → 16k 单声道队列
   → VAD 断句（同时 current_segment 用于草稿）
   → 流式 Paraformer 边听边出字（黄色 ▷）
   → 句末：CAM++ 声纹 → OnlineSpeakerTracker 立即贴标签（绝不回头改）
   → 后台 final_worker：whisper / confucius 精修 → 覆盖定稿 → 重写 4 个文件
```

`OnlineSpeakerTracker`：L2 归一化中心 + 余弦匹配 + EMA(0.08) 自适应；
短于 1s 的句子沿用上一句标签；`--num-speakers` 锁人数上限，达到上限后强制归到最像的人；
退出存 `voices.json`，`--enroll` 可跨次保持标签。

### 踩坑（按发现顺序）

1. **采集 EOF 丢失尾部数据**：最初 `read()` 遇到结束哨兵会把它放回队列，导致重复读到哨兵、
   可能提前退出。修法：`_wrap` 的 finally 里先 `eof.set()` 再 put(None)，FIFO 保证哨兵之前
   的数据全部先被消费；`read()` 用 `_eof_seen` 幂等。
2. **静音看门狗失灵**：WASAPI 回环在完全静音时也有微弱底噪，只看 `peak` 会把底噪算成
   「有声」，实测没播声音时显示「有声占比 100%」。改为 **RMS > 0.003 且 peak > 0.005**
   才算有声；被安全软件拦下时数据是严格全 0，必被识别。
3. **Tkinter 必须在主线程**：最初把 `gui_main` 放子线程、采集放主线程，退出必抛
   `Tcl_AsyncDelete: async handler deleted by the wrong thread`。反过来
   （主线程守窗口、流水线跑工作线程）后消失。
4. **状态条被挤掉**：`txt.pack(expand=True, fill="both")` 若先于状态条 pack，
   在高 DPI 缩放下底部 Label 不可见。改为**先 pack `side="bottom"` 的状态条**，
   再 pack 文本区。另：本机 4K 屏 DPI 缩放约 2x，截图裁剪要到 y≈1150 才看得到状态条，
   前两次「状态条没显示」其实是裁剪没覆盖（已用最小 UI 检查截图确认
   `电平 ▮▮▮▮▮▮▮▮▮▯▯▯ -10.2 dB ● 采集中 ...` 绿色正常显示）。
5. **首字延迟算错**：最初用 `VAD segment.start(样本索引) + 启动墙钟` 反推音频时刻，
   循环回放时音频时间轴重启而 VAD 计数继续，算出假的 1.56s。改用
   **「VAD 判定有人开始说话的墙钟时刻 → 首次出字」** 作基准，实测 0.06s / 最大 0.13s。
   文档里必须写清基准，否则和原 hybrid 项目那个「0.02~0.08s（以音频到达为基准）」不可比。
6. **模型下载源**：GitHub release 的 streaming-paraformer 是 1GB 全档 tar.bz2，
   实际只需 HF 上 `csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en` 的
   int8 三件套（约 226MB）。`hf-mirror.com` 直连可用、无需代理；
   `scripts/download_models.py --with-live` 支持精确字节校验，幂等已验证（全部 skip）。
7. **代理**：`live.py` 顶部给 `NO_PROXY/no_proxy` 追加 localhost，`ConfuciusASR` 用
   `httpx.Client(trust_env=False)`，双保险避开代理劫持 7861（沿用既有踩坑记录）。
8. **许可证已核实**（2026-11-09）：流式 Paraformer 模型卡 `license: apache-2.0`
   （源自 ModelScope damo）；Silero VAD MIT；PyAudioWPatch 仓库 `LICENSE.txt` 为
   Apache-2.0（GitHub API 显示 NOASSERTION 只是未自动识别）；sounddevice MIT。
   全部写进 `THIRD_PARTY_NOTICES.md`。

### 实测验收（RTX 4090 D / Windows）

| 场景 | 结果 |
|---|---|
| 双人英文 16s，`--final none` | 4 句，前 2 句 Host / 后 2 句 Guest，与离线聚类结果**完全一致**；4 个输出文件齐全 |
| 四人中文 56.9s，`--final whisper --simplify` | 7 句 / 4 人，文本通顺带标点；首字 0.06s(avg)/0.13s(max)，句末定稿 0.29s(avg)/0.56s(max) |
| 真实 WASAPI 回环 10~14s | idx=47 扬声器(USBAudio2.0)[Loopback] 48kHz/2ch，**开流 0.08s**，优雅退出；无声时 0 句 |
| GUI 窗口 | 历史行全保留、说话人分色（青/橙）、黄色 ▷ 草稿行、底部绿色状态条，全部截图确认 |
| `run_live.bat --version` | `SpeakerCaption live 0.1.0` |

第一次真实采集曾 180s 无输出被超时终止（推测卡巴斯基授权弹窗等待），
第二次开流 0.08s 正常；与我此前「每次手动点允许」的经验一致，故保留状态条红灯与
有声占比提示，不试图绕过安全软件。

### 已知取舍

- 在线分人**不回头改标签**：每人只说一两句时中心不稳，四人中文里有一句被归错，
  错分率高于离线全局聚类。需要最稳标签就会后把录音丢给 `diarize.py` 重跑。
- 重叠说话人（抢话）判给其中一人，ASR 会丢字。
- `--final none` 定稿零延迟但流式 Paraformer 无标点；`--final confucius` 中文标点/热词
  最好但约 3~5s/句，适合不赶时间的会议记录。


## 15. 繁体字 / 英文被译成汉字：根因与选项化改造（2026-11-09）

### 现象
我实测直播听录后反馈：窗口里中文全是繁体（「忘記了那些文化革命的記憶…」），
英文语音被直接写成中文译文，且同一次运行里简体、繁体、英文混杂。`live_out/live.json`
显示 `final_backend: whisper`，所以不是 Confucius 服务翻译的。

### 根因（实测）
`live.py` 的 `--language` 默认值是 `zh`，`WhisperASR` 把它原样传给
`faster_whisper.transcribe(language="zh", task="transcribe")`。Whisper 被强制
「输出中文」后，遇到英文语音会按中文解码，一部分句子给出中文译文、一部分仍留英文；
而 large-v3 的中文输出本身高频繁体 —— 两个现象是同一个开关造成的。

用 `tests/data/1-two-speakers-en.wav`（16s 英文）做对照：

| 参数 | `info.language` | 输出 |
|---|---|---|
| `language="zh"` | `zh` | `a pencil with black lead writes bestthe lamp shone…`（本段仍为英文，但语种标记已被强行改成 zh） |
| `language=None` | `en` | `a pencil with black lead writes best the lamp shone…` |

纯朗读句子未复现整句翻译，但真实连续语流里已出现大段中文译文；结合
`live.txt` 中「同一段英文演讲有的句子是英文、有的句子是中文译文、有的简体有的繁体」
这一特征，可以判定是强制语言标记 + large-v3 繁简不稳共同作用，而不是模型在做正经翻译。

### 修法：把「副作用」改成「显式选项」
新增 `src/speakercaption/text_pipeline.py`，把文本处理拆成两级：

```
ASR 原文 --[prepare: 可选翻译]--> semantic --[render: 简/繁/原样]--> 上屏文本
```

- `prepare()` 是语义步骤（调本地 LLM），结果按原文缓存；
- `render()` 是纯变换（OpenCC `t2s` / `s2t`），可逆、便宜，所以**运行中切换简繁或时间戳
  时可以把已经上屏的历史全部重算**，并同步重写 4 个输出文件；
- `language` 默认改 `auto`（传给 Whisper 为 `None`），`task="transcribe"` 显式写死，
  避免 Whisper 的 translate 任务把一切变英文；
- `--translate` 默认关，开启后只对「拉丁字母占比高」的句子调本机 Ollama
  (`/api/chat`)，`model=auto` 时读 `/api/tags` 优先挑翻译专用模型；
  Ollama 不可用只告警并保留原文，绝不中断听录；
- 中文流式 Paraformer 只认中文：定稿检测到非中文语音后自动隐藏黄色草稿，
  免得看起来又像「英文被翻成中文」。

### 图形界面
`live.py` 不再一启动就开流，改为先出带设置面板的窗口，点「▶ 开始听录」才启动采集：

- 第一行：`译成中文` / `字形（简体·繁体·原样）` / `时间戳` / `音频语言` /
  `精修后端` / `人数`；第二行：`高级设置…` / `恢复默认` + 说明；
- 高级设置含 Whisper 模型与设备、热词、Ollama 地址与模型、输出目录与格式、
  声纹阈值、断句静音、采集时长、注册声纹等；
- 选项存 `settings.json`，命令行参数当次覆盖；GUI 主线程持有窗口，
  `UIController` 在工作线程里启停 `LiveSession`（沿用 §14 的 Tkinter 线程结论）；
- 运行中禁用需要重启的选项，只放开字形和时间戳即时切换。

踩坑：① 一开始把所有控件塞进一行，1040px 窗口里「恢复默认」被挤出，
改成两行后截图确认全部可见；② `tk.Text` 初始提示文字被 150ms 的 `refresh()`
按空快照重写冲掉，改为在「无定稿、无草稿、无错误」时由 `refresh()` 自己插入；
③ `security_soft_running()` 用 `text=True` 读中文 Windows 的 GBK `tasklist`，
subprocess reader 线程抛 `UnicodeDecodeError` 刷屏，改为 `bytes.decode(..., "ignore")`。

### 验收
| 场景 | 结果 |
|---|---|
| 英文 16s，默认参数 | 4 句全部英文原文（`Pencil with black lead writes best.` …），检测到 en 后自动隐藏中文草稿 |
| 四人中文 22s，默认 | 「这是一个测试说话人日志的音频」——简体 |
| 同上 `--traditional --timestamps` | 「這是一個測試說話人日誌的音頻」，每行带 `[00:00.7]` |
| Ollama 翻译链路 | 自动选中 `melashri/hy-mt2-30b-a3b:latest`，`The new generation has also forgotten…` → 「新一代人也已经忘记了那些关于文化大革命的记忆」，纯中文不送翻译 |
| `--translate` 端到端 13s | 3 句全部译成中文（「用黑铅芯的铅笔书写效果最佳。」…），句末定稿平均 1.53s |
| 简繁 / 时间戳即时切换 | 控制器无 GUI 直跑，历史文本与 `live.txt` 同步重算 |
| GUI 截图 | 两行设置面板、提示文字、底部状态条均正常，无裁切 |

翻译链路还补了两处稳定性：① Windows 上复用 Ollama keep-alive 连接第二句报
`ReadError [WinError 10038]`，改为 `max_keepalive_connections=0` + `Connection: close`，
并在失败时重建 client 重试一次；② 18GB 模型首次加载约 18s，会把第一句定稿拖到 18.5s，
改为会话启动时后台线程 `warmup()` 预热，实测降到 1.53s（模型已驻留时）。

已知取舍：翻译是**按句**调用本地大模型，冷启动仍要加载模型、之后每句也有额外延迟；
18GB 模型与 GPU 版 Whisper 同时加载可能吃满 24GB 显存，需要时把高级设置里的推理设备
改 CPU 或换小模型。已写进 README 已知限制。

## 16. 点「停止」停不下来：实例属性遮蔽了同名方法（2026-11-09）

### 现象
真实 WASAPI 回环听录中我点「■ 停止」没反应，控制台连刷两次：

```
Exception in Tkinter callback
  File "...\speakercaption\live_gui.py", line 449, in start_click
    controller.stop()
TypeError: 'Event' object is not callable
```

按第二次是因为第一次抛异常后 `btn.configure(text="正在停止…")` 没执行，
按钮仍显示「■ 停止」，看起来像「点了没反应」。

### 根因
`UIController` 里同时存在：

```python
def __init__(...):
    self.stop = threading.Event()      # 实例属性
def stop(self):                        # 同名方法，被上面彻底遮蔽
    self.stop.set()
```

Python 实例属性优先于类方法，`controller.stop()` 实际是在**调用 Event 对象**。
异常发生在 Tk 回调里，Tk 只打印 traceback 不中断 mainloop，于是采集继续跑。
`on_close()` 里写的是 `self.stop.set()`（把它当 Event 用）反而没报错，
所以关窗口能退出、只有按钮不行——这个不对称也解释了为什么之前的控制器冒烟
（直接 `ctl.stop.set()`）没暴露问题。

### 修法
- 事件对象改名 `self.stop_event`，`stop()` 恢复成方法；`start()` / `on_close()` /
  `run_ui()` 全部同步改引用，`on_close()` 统一调 `self.stop()` 而不是直接 `.set()`。
- 补 `state["stopping"]`：点停止后按钮立刻变「正在停止…」并禁用，
  状态条显示「正在停止…（等待最后几句定稿，已 Ns）」，重复点击直接忽略；
  pipeline 真正结束（`controller.is_running()` 轮询变 False）才恢复控件。
  等待超过 12s 提示可关窗口强制退出。
- 新增 `tests/test_ui_controller.py` 回归测试：断言 `callable(ctl.stop)`、
  `stop()` 能置位事件，并用 `--simulate --loop-sim --final none` 起真实 pipeline
  后调用 `stop()`，验证工作线程在 30s 内结束且无错误。

### 验收
| 项目 | 结果 |
|---|---|
| `tests/test_ui_controller.py` | 两项全过，日志出现 `[结束] 我停止` |
| 真实 Tk 按钮回调（探针脚本临时注入 `tk.Tk` + 包装 `gui_main`，invoke 两个按钮） | `running_before_stop=True` → `running_after_stop=False`，按钮文字恢复 `▶ 开始听录`，`run_ui rc=0` |
| `tests/test_text_pipeline.py` | 12 项仍全过 |
| 模块导入 / `live.py --version` | 无异常 |

教训：**别把 `threading.Event` 属性命名成和公开方法同名**；Tk 回调里的异常
不会弹窗、不会退出，只会打到 stderr，所以「按钮点了没反应」要第一时间看
控制台 traceback，而不是先怀疑采集线程卡死。临时探针脚本验收后已删除。

## 17. 停止后再开始报 -9999：PortAudio 全局状态与二次开流（2026-11-09）

### 现象
修好停止按钮后，我「听录 → 停止 → 再点开始」时控制台连续两轮：

```
[采集] 失败: OSError: [Errno -9999] Unanticipated host error
[结束] 采集失败  运行 0.4s  共 0 句 / 0 位说话人
```

第一轮正常（有声占比 97.4%、开流 0.08s、11 句定稿），只有第二轮起开不了流。

### 根因（两层）
1. **PortAudio 是进程级全局状态**。`_run_wasapi()` 每轮都 `pyaudio.PyAudio()`
   （= `Pa_Initialize()`），旧 `AudioSource` 被丢弃后其 PyAudio 实例被 GC 时又会在
   不确定时刻 `Pa_Terminate()`。两轮初始化互相踩，加上 WASAPI 回环会话释放有滞后，
   第二次 `Pa_OpenStream` 返回 `paUnanticipatedHostError`。
   `_close_stream()` 还被 pipeline 线程和采集线程并发调用，没有串行化。
2. **安全软件在二次开流时也会拦**。做真实设备「开流→停止→再开流」验收时，
   两轮都停在「正在打开音频设备…」8 秒无错误；截屏确认是卡巴斯基弹窗
   「_wasapi_twice_tmp.py 正在试图访问麦克风」在等人工点「立即允许」。
   即 `-9999` 既可能是设备没释放，也可能是授权被拒/超时——两者都要覆盖。

### 修法
- `live_audio.py`：模块级 `_shared_pyaudio()`（带锁单例），全进程只初始化一次，
  绝不 `terminate()`；`_STREAM_LOCK` 让 `_close_stream()` 串行且幂等；
  `stop()` 在关闭流并 join 采集线程后多等 0.25s 让 WASAPI 释放会话。
- 新增 `_open_loopback_stream()`：把 `open()` + `start_stream()` 包进
  退避重试（0 / 0.4 / 0.9 / 1.6s，共 4 次），只对 `-9999` /
  `unanticipatedhosterror` 重试；最终失败时打印三步诊断（安全软件授权 →
  等 1~2 秒重开 → `net stop/start Audiosrv`）。`_is_host_error()` 去空格后
  同时匹配 `Unanticipated host error` 与 `paUnanticipatedHostError`。
- `live_core.py`：`LiveSession` 增加独立 `_final_stop`，`final_worker` 不再拿外部
  stop 当循环条件——GUI 二次启动会 clear 外部事件，旧 daemon 定稿线程本来会被
  「复活」并永久驻留、继续写同一个 display / 占着上一轮的 ASR 模型。
  采集失败时把原因 `display.set_error(...)` 送到窗口红色状态条。
  安全软件提示改为「**每次**点开始都要授权」。
- `live_ui.py` / `live_gui.py`：`UIController.start()` **每轮新建 Event**（不再
  `clear()` 复用），`gui_main()` 去掉 `stop_event` 参数，关窗口与 mainloop 退出
  统一走 `controller.stop()`，避免指向上一轮的旧事件。

### 验收
| 项目 | 结果 |
|---|---|
| `tests/test_live_audio.py`（6 项，fake pyaudiowpatch） | 单例只初始化 1 次、host error 重试 3 次成功、非 host error 立即抛、重试耗尽抛、**完整 start/stop 两轮**均开流成功 |
| `tests/test_ui_controller.py`（3 项） | 含新增「停止→再开始」两轮，事件对象互不相同 |
| GUI 真实按钮回调（临时探针 invoke 两轮开始/停止） | 4 次点击全部命中；`running: True→False→True→False`，按钮恢复「▶ 开始听录」，`rc=0` |
| `tests/test_text_pipeline.py` | 12 项仍全过 |

**未能自动验收的部分**：真实 WASAPI 回环的二次开流测试被卡巴斯基授权弹窗挡住
（无人值守没人点「允许」），所以这一项需要我在本机实测。若重试后仍报 -9999，
按控制台三步提示处理，重点是弹窗勾「为此应用程序记住我的选择」再点「立即允许」。
临时探针脚本与截图均已删除。

## 18. 纠正 §17：-9999 的真因是 WASAPI 线程亲和性（2026-11-09）

### 我反馈推翻了上一节的归因
§17 把「停止→再开始」的 `-9999` 主要归给卡巴斯基二次授权。我实测后明确反馈
**「并没有弹出卡巴斯基的拦截窗口」**，且日志显示第二轮是**立刻**失败：
开流 0.00s、4 次退避重试共 3.3s 就结束，`有声占比 0.0% / 共 0 句`，
不是弹窗等待那种「停在正在打开音频设备…」的挂起。上节的单例复用、0.25s
等待、4 次重试全都没用，说明方向错了。

### 分组诊断（真实设备，WASAPI 回环 idx=47 48000Hz/2ch）
临时探针 `tests/_pa_diag_tmp.py`，三种模式各起**独立进程**避免互相污染：

| 模式 | 做法 | 结果 |
|---|---|---|
| `same-thread` | 同一线程 `PyAudio()` → open → close → open → close → open | **三次全部成功** |
| `cross-same-pa` | 线程 A 建 PyAudio / open / close，线程 B 复用该实例 open | `-9999` |
| `cross-new-pa` | 线程 A 建 PyAudio / open / close，线程 B **新建** PyAudio open | `-9999` |

结论：与「设备释放滞后」「PyAudio 单例」「安全软件授权」全都无关，
**唯一变量是线程**。PortAudio 在 `Pa_Initialize` 所在线程初始化 WASAPI / COM，
设备接口带线程亲和性，之后 `Pa_OpenStream` 必须回到那条线程；
PyAudio 是不是同一个实例都救不了（第二次 `Pa_Initialize` 在同一进程内只是
返回「已初始化」，COM 仍是 A 线程的）。

原实现 `AudioSource.start()` 每轮 `threading.Thread(target=self._wrap(...))`
新建 `capture` 线程，第二轮必然换线程 → 必现 -9999。第一轮永远成功，
恰好解释了「重启程序就好了」。

### 修法：全进程固定一条音频线程
`live_audio.py` 新增 `_AudioHost`（单条 `audio-host` daemon 线程 + 任务队列）：

- `submit(fn)` 异步跑长任务（一次听录的 `_run_wasapi` / `_run_sd`），返回完成事件；
- `call(fn, timeout)` 同步跑短任务并回传结果/异常，用于 `--list-devices` 与兜底关流；
- `AudioSource.start()` 对 `wasapi` / `sd` 后端把整个 runner **提交给 host**，
  不再自建线程；`file`（`--simulate`）回放仍走普通线程；
- `stop()` 只 set 事件并等 host 任务自己收尾，`_run_wasapi` / `_run_sd` 用
  `try/finally` 在**同一 host 线程**里关流；外部线程绝不直接 close
  （超时兜底才跨线程关一次，宁可异常也别把设备锁死）；
- `_shared_pyaudio()` 检测调用线程，不是 host 就 `host.call()` 转交，
  保证 `PyAudio()` 本身也在 host 线程创建；`list_devices()` 同样转交；
- `_close_stream_obj()` 同时兼容 PyAudio `stop_stream()` 与 sounddevice `stop()`
  （原来 sd 分支的 `stop_stream()` 会抛 AttributeError 被静默吞掉）；
- 误导性提示全部改掉：开流失败诊断把「系统是否在播放 / 默认输出设备是否被切换」
  放第一位，安全软件授权降到第三位；静音告警不再断言「多半是安全软件」。

### 验收
| 项目 | 结果 |
|---|---|
| `tests/test_live_audio.py` 7 项（新增 `test_all_portaudio_calls_on_one_thread`） | 全过；断言两轮的 `PyAudio()` / `open()` / `close()` 线程 id 全等于 `audio-host`，且不是主线程 |
| 真实 WASAPI 回环连续 3 轮 `AudioSource.start()/stop()` | 三轮全部 `error=None`，开流 6.44s / **0.02s / 0.01s**，每轮 `read()` 都拿到 1365 样本；第二三轮零重试 |
| `tests/test_ui_controller.py` 3 项 | 全过，GUI 两次启停不受影响 |
| `tests/test_text_pipeline.py` 12 项 | 全过 |

第一轮 6.44s 是卡巴斯基授权弹窗等待（我当时点了「通过」），
**但后两轮不再弹、也不再报 -9999**，正好把「授权」与「线程」两件事分开验证。

### 教训
1. 我说「没弹窗」时要立刻推翻自己的假设，别继续加固错误方向的提示文案。
2. 「立刻失败 + 重试无效」和「挂住等授权」是两种完全不同的故障形态，
   看 `开流耗时` 与重试日志就能区分。
3. Windows 音频栈里 **COM/WASAPI 的线程亲和性**是个隐形契约：
   PortAudio 初始化、开流、关流、设备枚举都要在同一线程；
   「复用单例」不等于「复用线程」。
4. 单元测试用 fake 模块只能验证代码路径，验证不了线程亲和性；
   这次必须加一条记录 `threading.get_ident()` 的断言，才把根因锁死。
临时诊断脚本验收后删除。

## 19. 单人模式：直播里最常见的其实是「只有一个人讲话」（2026-11-09）

### 需求
我提出：需要一种「部分说话人」的方式，专门给**全程只有一个人讲话**的直播内容，
并指出「其实这种情况更多见」。

### 为什么不能只靠 `--num-speakers 1`
在线分人（`OnlineSpeakerTracker`）是**增量匹配**：每句结束就拿这一句的声纹 embedding
去找最近中心，低于阈值就判成新说话人。单人直播里同一个人的短句（1~2 秒，比如
「对」「是的」）采样太少，embedding 不稳定 → 阈值判不过 → 被判成「新说话人2」→
再判成「新说话人3」……标签在窗口里乱跳，反而比不做分人更糟。
另外声纹模型（CAM++）本身也要占内存和 CPU，单人场景纯属浪费。

### 修法
1. `live_speaker.py` 新增 `SingleSpeakerTracker`：
   - 与 `OnlineSpeakerTracker` 接口对齐（`assign` / `display_name` /
     `num_speakers` / `save` / `load`），但 `assign()` 恒返回 `(1, 0.0, False)`；
   - `last_label = None`，让 GUI 的黄色草稿一直走草稿色而不是说话人色；
   - `save()` 不写文件（没有声纹可存），`load()` 返回 0；
   - `build_extractor()` 在单人模式下**完全不调用**，连 CAM++ 模型都不加载。
2. 三个入口归一到同一个开关：
   - GUI「人数」下拉新增 `单人（不分人）`（值 `1`）；
   - CLI `--single-speaker`；
   - CLI `--num-speakers 1`。
   `live_core.py` 里统一判断 `num_speakers == 1` → `args.single_speaker = True`，
   所以三者行为完全一致，不存在「改了参数没生效」。
3. `live_ui.py` 的 `_sanitize` 只认 `UI_KEYS`，不认 `--single-speaker`（它是命令行别名，
   不是界面选项），所以在 `run_ui()` 里把它翻译成 `num_speakers=1` 再当次覆盖
   `settings.json`——否则命令行强制单人、界面里却显示「自动」，两边对不上。
4. 输出：`show_speaker=False` 时 `txt` 每行只有正文（无「说话人1：」），
   `srt` 无 `[说话人1]` 前缀；`json` 的 `speaker` 字段恒为 `1`（下游程序统一处理）；
   `rttm` 仍输出 `SPEAKER_01`（保持标准说话人日志格式可被其它工具导入）。
5. `cli.py`（离线版 `diarize.py`）同样支持：单人模式仍跑 segmentation（它同时负责断句），
   但聚类强制自动，合并成 1 位说话人，`txt`/`srt` 不显示说话人。

### 验收
| 项目 | 结果 |
|---|---|
| `tests/test_single_speaker.py` 12 项 | 全过：三种入口归一、`show_speaker` 开关、txt/srt 无前缀但时间戳照常、json 保留 speaker 字段、tracker 恒返回 label=1、save 不落盘 |
| 全量回归 `unittest discover tests/` | 33 项全过（含 text_pipeline 12、live_audio 8、ui_controller 3、smoke 等） |
| 真实音频端到端（`--simulate` 官方四人中文 56.9s） | 单人模式正常出字、无说话人前缀、声纹模型未加载 |

### 取舍
- 单人模式下**忽略** `--enroll` / `--save-voices`（没有声纹可存），并打印提示而不是静默忽略；
- 一个人中途换人讲话，单人模式不会发现——这是这个模式的前提假设，不是缺陷；
  真需要分人时切回「自动」或指定人数即可。

### 顺带查清的一件事：`live.py` 默认开 GUI
验收时先怀疑程序卡死：进程占着不退出、控制台零输出、CPU 0%。
分步打点（import → parse_args → build_vad → StreamingASR → build_finalizer）
发现各组件都是秒级加载，**Whisper 加载实测 <1s**（HF 缓存 2.9G 完整，
`model.bin` 在位），根本不是模型下载或 CUDA 初始化慢。

真因：`main()` 里 `args.no_gui` 为假时会走 `run_ui()` 拉起 Tkinter 窗口，
**等人点「▶ 开始听录」**。无头环境下就是一个静悄悄挂着的空窗口——
零输出、零 CPU、不退出，看起来像死锁，其实是「在等一个人」。

教训：
1. 「进程卡住 + 零输出」先分清是**卡在 IO** 还是**在等人交互**；
   本次靠分步打点 + 组件单独加载全绿，把范围缩到 `main()` 的分支上；
2. 凡是要写自动化验收，命令必须显式带上 `--no-gui`；
   默认 GUI 的程序在无人值守下就是这个症状；
3. `--simulate` 回放这条链路（不需要真实设备）应该固化成回归用例。

### 验收（修正后的真实数据）
| 项目 | 结果 |
|---|---|
| `tests/test_single_speaker.py` 12 项 | 全过 |
| 全量回归 `unittest discover tests/` | 33 项全过 |
| 无头端到端（`--no-gui --single-speaker --simulate` 官方四人中文 56.9s，`--final whisper`，CUDA float16） | 运行 30.5s，出 3 句 / 1 位说话人；`txt`/`srt` **零说话人前缀**；`json` 保留 `speaker: 1` 与 `speaker_mode: "single"`；**未生成 voices.json** |
| 同上 `--final none` | 运行 15.5s，1 句，退出码 0 |
| 首字上屏延迟 | 平均 0.35s / 最大 0.47s |
| 句末定稿延迟（whisper） | 平均 0.54s / 最大 0.72s |

## 20. 命令行参数被写回 settings.json：GUI 一直回放测试音频（已修）

### 现象
我跑 GUI 后，窗口里出现的是**测试音频**的内容（「这是一个测试说话人日志的音频」
「下面我们来播放一些测试音频」），而不是电脑上真正在播的声音；
状态栏写着「文件回放 0-four-speakers-zh.wav」。

### 根因
`settings.json` 被写成了：
```json
"simulate": "tests/data/0-four-speakers-zh.wav",
"duration": 30.0,
"outdir": "live_out/_single_test",
"formats": "txt,srt,json"
```
这几个值**全部来自命令行**，不是我在界面上选的。

链路是：
1. `run_ui()` 把「落盘值 + 命令行覆盖」合并成一份 `settings` 传给 `UIController`；
2. `gui_main()` 用这份合并值初始化各个下拉框/输入框；
3. 控件 `trace_add("write", on_change)` 在**初始化时就会触发**，
   `on_change` 调 `controller.save(cur)`；
4. `controller.save()` 直接把 `self.settings`（含命令行覆盖）写回 `settings.json`。

于是「临时用一次 `--simulate` 验收」变成了**永久设置**，之后每次开 GUI 都在回放
那个测试文件，根本不抓真实系统声音。这违反了 README 的承诺
「命令行参数只对当次启动覆盖」。

### 修法
把「本次生效的值」和「真正落盘的值」彻底分开：
- `run_ui()` 分别算出 `persisted`（settings.json 原值）与 `effective`（原值 + 命令行覆盖）；
- `UIController` 新增 `_cli_overrides` / `_persisted` / `_effective_at_start`，
  统一走 `_apply(opts)`：
  - 生效值 = 界面当前值（命令行覆盖照样生效，界面也照样显示它）；
  - 落盘值 = 原落盘值，**只有我在界面上真的改过的那一项**才更新；
    没动过的命令行项一律保持 settings.json 里的原值；
- `UIController.__init__` 在调用方没传 `persisted` 时兜底：
  命令行覆盖过的键回落到默认值，绝不把命令行值当落盘值。

顺带修的两处：
1. **「人数」下拉框截断**：`width=6` 装不下 7 个全角字的「单人（不分人）」，
   界面上只显示「（不分人」。改成 `width=14`，窗口从 1040 加宽到 1200。
2. **`tests/test_ui_controller.py` 从没被执行过**：它用的是裸函数 + `assert`，
   `unittest discover` 只收集 `unittest.TestCase` 子类，所以停止按钮那 3 项
   回归一直是「假绿」。改成 `unittest.TestCase` 后才真正进套件。

### 验收
| 项目 | 结果 |
|---|---|
| `tests/test_ui_controller.py` 改为 unittest 后 | 3 项停止/重启回归**首次真正执行**，全过 |
| 新增 `TestCliOverrideNotPersisted` 3 项 | 命令行覆盖不落盘、我真改了要落盘、改别的项时仍不落盘，全过 |
| 全量回归 | **39 项全过**（此前 33 项） |
| 真实 GUI 启动（带 `--simulate … --duration 30 --outdir …`）后 `diff settings.json` | **零差异**，配置不再被污染 |
| 真实 WASAPI 回环（清空配置后） | `WASAPI 回环 idx=47 扬声器 (USBAudio2.0) [Loopback] 48000Hz/2ch`，开流 0.12s，有声占比 99.3%，正常出字 |
| 界面图 | `docs/ui-options.png` 重新抓取，「单人（不分人）」完整显示、无溢出 |

### 教训
1. **「合并配置」和「要保存的配置」必须是两个对象**。合并是为了本次生效，
   保存是为了下次生效；混在一起必然出「临时设置变永久设置」的坑。
2. GUI 初始化时的控件 `trace` 回调也会触发保存逻辑——「保存」不能假设
   「一定是我操作引起的」，得先比对本次启动时的生效值再决定要不要落盘。
3. **测试文件形式对不对，会决定它有没有被跑过**。裸函数 + `assert` 的
   「测试脚本」混在 `unittest discover` 里等于没写。改完要把数量对上
   （33 → 39），不能只看「全过」。
4. 验收时**先看配置文件本身**：`settings.json` 里出现自己没在界面上选过的值，
   就是被某个路径偷偷写进去的。

## 21. 输出目录用相对路径 → 换个目录启动就写错地方（已修）

### 现象
我问「听录出来的文字 TXT 存放在哪里了」。查下来 `settings.json` 里的
`outdir` 是相对路径 `"live_out"`，而代码里是 `Path(args.outdir)` —— 这是相对
**进程当前工作目录**解析的，不是相对工程目录。

### 为什么是隐患
- `run_live.bat` 里有 `cd /d "%~dp0"`，从 bat 启动没事；
- 但从 IDE 里点运行、或别的脚本 `import` 调用、或者在别的盘符下敲绝对路径启动，
  `live_out` 就会落到那个目录底下——文件确实生成了，只是**我找不到**。
- 离线版 `cli.py` 的 `write_outputs(segments, args.outdir, ...)` 同样问题。

### 修法
`live_core._resolve_outdir()`：相对路径一律锚定到工程根目录 `ROOT`；
绝对路径原样保留；空值回落到 `ROOT/live_out`。`cli.py` 复用同一个函数。

### 验收
| 项目 | 结果 |
|---|---|
| 新增 `tests/test_paths.py` 4 项 | 相对路径锚定工程根、绝对路径保留、空值回落、**换 cwd 结果不变** |
| `tests/test_smoke.py` 也改成 `unittest.TestCase` | 此前同样是裸脚本、`unittest discover` 没收它，**模型是否还能用一直没有回归保护**；现在 2 项真正进套件 |
| 全量回归 | **45 项全过**（此前 39） |
| 真实从 `/tmp` 启动 `live.py --outdir live_out/_cwdtest` | 产物落在 `D:\AI_TTS\SpeakerCaption\live_out\_cwdtest\live.txt`，`/tmp` 下无误写 |

### 教训
5. **相对路径的锚点要写死在代码里**，别指望调用方先 `cd` 到工程目录——
   那是调用方的隐含义务，很容易被破坏。
6. 仓库里凡是「脚本式的测试」，都要确认它真的被测试运行器收集到了；
   这次的 `test_ui_controller.py`、`test_smoke.py` 都是**假绿**。

---

## 22. 三个新需求：媒体文件听录 / Confucius 免启服务 / 界面重做

我提了三件事：(a) 既然能听系统声音，能不能把「听读多媒体文件」也做进来；
(b) Whisper 模型到底在哪；(c) Confucius 是不是必须先在别处起服务。
最后还要求引入 UI 专家做界面美化。

### 22.1 媒体文件听录（GUI 内置）

**为什么不新开一个程序**：GUI 里已经有成熟的「上屏 + 着色 + 写盘」逻辑，
再开一个窗口等于把 `LiveDisplay` 的消费方复制一遍。

**做法**：把离线版的整段管线抽成 `file_mode.FileTranscriber`（解码 → 说话人分段
→ 逐段 ASR → 简繁/翻译 → 写盘），GUI 顶部加**模式切换**（实时听录 / 听录文件），
文件模式下多一条「媒体文件」输入行 + 「浏览…」按钮。

- 文件模式走**离线整段管线**（全局聚类），比实时的在线增量分人更准，
  代价是必须等整个文件处理完；
- 产物按**原文件名**存（`会议录音.mp4` → `live_out/会议录音.txt`）；
- 视频靠 ffmpeg 抽音轨（本机 ffmpeg 7.0.2 在 PATH）；
- 结果复用 `display.add_final()` 上屏，与实时模式同一套着色/滚动逻辑。

### 22.2 Confucius 免启服务（`confucius-local`）

**先做了调查**（这一步决定了架构）：
- `cy_app` 是个 Cython 编译模块（`cy_app.cp312-win_amd64.pyd`），
  里面导出了 `do_text(audio, video, input_chage, text_type, hot_word, model_type)`；
- 直接 `import cy_app` + 消费它的 generator，实测能拿到 `(文本, 输出文件路径)`，
  耗时 9.66s / 6.1s 音频 —— **说明可以完全绕开 Gradio HTTP 协议**；
- 但 `cy_app` 依赖 torch，而 SpeakerCaption 的 venv **刻意不装 torch**
  （2.5 GB+，还和 faster-whisper 的 ctranslate2 栈互相干扰）；
- Confucius 自带 `py312/python.exe`，里面是 torch 2.8.0+cu128 且 CUDA 可用。

**最终方案**：**子进程隔离**。
SpeakerCaption 用 **Confucius 自带的解释器**起一个 worker 脚本
（`confucius_worker.py`，几十 KB），worker 在自己进程里 `import cy_app` 并调
`do_text()`，结果通过 stdin/stdout 的 JSON 行传回。

> 这样同时满足：我不用手动起 7861 服务；不占端口；模型只加载一次可复用；
> **SpeakerCaption 侧零新增重量级依赖**。

### 22.3 顺手修的既有 bug

`build_finalizer`（实时）传的是 `url=`，而 `build_asr` 读的是 `kw.get("confucius_url")`
—— **实时的 `--confucius-url` 自定义地址一直被静默忽略**，永远连默认 7861。
现在两种 kwarg 名都认。

### 22.4 界面重做（UI 专家视角）

专家市场里**没有匹配的 UI 专家**（`search_plugins` 两次都返回空），
于是按通用 UI/UX 方法论自己做了，并写进代码注释当规范：

- **设计令牌化**：颜色/字体全部提为顶部常量（`CONTROL` / `CONTROL_HI` / `ACCENT` /
  `DANGER` / `WARN` / `FONT_*`），改一处全局生效，避免"临时加颜色"；
- **三级灰阶分层**（背景 < 面板 < 控件），靠明度差表达层次，不靠描边；
- **强调色收敛到三个**：绿=主操作、红=停止、琥珀=提示，其余全中性灰；
- **两级字号**：正文 10pt，字幕区 15pt（它才是主角）；
- 模式切换做成**分段按钮**（选中态用 `CONTROL` 底色区分），不用下拉；
- 文件模式的主按钮文案随之变成「开始转写」，状态栏也跟着变。

### 22.5 踩到的坑（都是 Tk 的）

1. **`tk.Frame(pady=(8,0))` 报 `bad screen distance "8 0"`** ——
   `Frame` 的 padx/pady 只接受**整数**，元组是 `pack`/`grid` 的语法。
2. **`pack` 顺序 = 上下顺序** —— `file_hint` 在创建时顺手 `pack()` 了，
   后来 `apply_mode()` 再 `pack(file_bar)`，说明文字就跑到了文件选择框**上面**。
   同理 `mode_area` 必须用 `pack(before=top)` 插到指定位置，直接 pack 会掉到最底下。
3. **`RLock` vs `Lock`**：`ConfuciusLocalASR._request()` 持有锁时又调
   `_ensure_worker()`，后者要拿同一把锁 —— 普通 `Lock` 不可重入，直接**自死锁**。
   表现为「进程卡住、CPU 0%、无输出」，`faulthandler` 一抓就看到了。
4. **跨进程读 stdout 必须单独开线程**：`readline()` 在管道上会阻塞，
   没法实现"等 N 秒就超时"；改成后台线程读 + `queue.Queue`。
5. **worker 的 stderr 必须抽干**：否则管道写满会让子进程卡死。

### 22.6 验收

| 项目 | 结果 |
|---|---|
| `tests/test_file_mode.py`（新增 12 项） | 单人压平标签、多人保留、ASR 参数透传、产物落盘、缺文件报错、目录探测、解释器探测、HTTP 后端两种 kwarg |
| 全量回归 | **57 项全过**（此前 45） |
| 文件模式端到端（`FileTranscriber`，官方四人中文 56.9s，whisper CUDA float16） | 8 段 / 1 说话人 / **11.0s**，txt/srt/json 正确 |
| 真实 GUI 文件模式（起窗口 → `controller.start_file`） | 8 段上屏、状态条显示阶段进度、产物按原文件名落盘 |
| `confucius-local` 端到端（`督工_去噪.wav` 6.1s） | 2 段，正确出「丙都宫油价略有下调。」等中文标点，**全程没起 7861** |
| `confucius-local` 初始化 / 首次转写 / 二次转写 | 8.8s / 4.8s / 3.0s（模型只加载一次） |
| GUI 冒烟（实时 / 文件两种模式） | 无异常，界面图已重抓（`docs/ui-options.png` / `docs/ui-file-mode.png`） |

### 22.7 关于「Whisper 模型在哪」

faster-whisper 从 HuggingFace 自动下载，落在
`%USERPROFILE%\.cache\huggingface\hub\models--Systran--faster-whisper-large-v3\`
（**2.9 GB**），**不在工程目录**，所以在 `models/` 里看不到它是正常的。
新增 `scripts/whisper_models.py list|path|link`：
`list` 列出本机缓存的模型和大小，`path` 打印完整路径，
`link` 在工程 `models/whisper/` 建**符号链接/目录联接**（不复制数据、不占额外空间）。

### 22.8 教训
7. **先做技术调查再定架构**：「必须手动起服务」这个前提，是靠
   `import cy_app` 实测 + 看导出符号证伪的。如果一开始就按原设计加个
   "请先启动服务"的提示，就白白让我多干一件事。
8. **重依赖要隔离，别硬塞进主环境**：torch 放在 Confucius 自带的解释器里，
   SpeakerCaption 这边只多几十 KB 的 worker 脚本。
9. **Tk 的 pack 顺序就是布局顺序**——任何"先创建后 pack"的控件都要想清楚
   它最终排在哪，容易出现"说明文字跑到输入框上面"这种反直觉布局。
10. **自死锁要用 faulthandler 抓**：`dump_traceback_later` 一行就定位到
    `_request → _ensure_worker` 的锁重入，比读代码快十倍。

## 23. GitHub 分发：Whisper 权重怎么给到别人

### 23.1 问题

我问：「Whisper 不在项目目录中，本机没问题，但分享到 GitHub 后其他人怎么用？
是假设所有人电脑里都已经部署好 Whisper 了吗？」

结论：**不假设**。要区分两样东西：

| | 是什么 | 怎么来 |
|---|---|---|
| faster-whisper **代码** | pip 包（MIT） | `pip install -r requirements.txt`，几秒 |
| Whisper **权重** | HuggingFace 上的 .bin（MIT） | 首次使用下载，或手动下好 |

所以新我只需要一条命令。但这暴露了原工程的三个**实际缺陷**：

1. 权重只在 HF 缓存里 → 不在工程目录 → 换机器要重下 2.88 GB，
   也无法「把 models/ 打包发给同事」；
2. 没有下载脚本 → 新我不知道该下什么、下多大、国内网络走哪条路；
3. **实时模式的 `--whisper-model` / `--whisper-device` 一直被静默忽略**（见 23.4）。

### 23.2 方案：来源优先级

新增 `src/speakercaption/model_paths.py`，统一解析模型来源：

```
显式路径 > download_root > 工程内 models/whisper/<档位> > HF 缓存 > 在线下载
```

关键点：**工程内存在模型时完全不发网络请求**（天然离线可用）。
我填了显式路径但目录不存在 → **报错，不静默降级**换成别的模型。

档位与体积（实测 HF API）：

| 档位 | 体积 | 建议 |
|---|---|---|
| tiny | 72 MB | 勉强能跑 |
| base | 145 MB | 草稿级 |
| small | 461 MB | **CPU 我起点** |
| medium | 1.42 GB | 平衡点 |
| large-v3 | 2.88 GB | **有显卡时的默认** |
| large-v3-turbo | 1.62 GB | 大显存追速度 |

### 23.3 实现

- `scripts/download_models.py` 加 `--with-whisper [TIER]` / `--whisper-tier A B` /
  `--endpoint`。用 `huggingface_hub.snapshot_download(local_dir=...)` 下到
  `models/whisper/<档位>/`——**真实文件**（不是缓存符号链接），所以整个
  `models/` 目录可以打包分享，对方解压即用、零联网。
  `--endpoint https://hf-mirror.com` 走国内镜像；`allow_patterns` 只取
  config/tokenizer/model.bin/vocabulary 四个文件，不下冗余。
- `asr.py` 的 `WhisperASR.__init__` 改用 `resolve_whisper()` 解析，
  并按来源打印不同日志（本地目录 / HF 缓存 / 即将下载）。
  下载失败时给出**三条可执行的解决办法**，而不是把
  `LocalEntryNotFoundError` 甩到脸上。
- 实时模式新增 `--download-root`，接进 `build_finalizer`；
  GUI 高级设置新增「Whisper 权重目录」字段，`download_root` 进 `UI_KEYS`。
- `scripts/check_env.py` 增加模型资产体检：逐项列出分人/VAD/流式模型在不在，
  Whisper 在工程内还是在 HF 缓存。
- README 新增「模型从哪来？（GitHub 我必读）」：三条路径 + 档位表 +
  镜像设置 + 手动下载方法。
- THIRD_PARTY_NOTICES.md 补 Whisper 权重的两种落地位置与 MIT 归属。

### 23.4 顺手修的既有 bug（重要）

`build_asr` 读 `kw.get("whisper_model")`，而实时的 `build_finalizer` 传的是
`model=` / `device=` / `compute_type=` —— **键名对不上**。
后果：**实时的 `--whisper-model` 和 `--whisper-device` 一直是被忽略的**，
永远退回默认的 `large-v3` + `auto` 设备。

实测证据：`--whisper-model tiny --whisper-device cpu` 跑出来日志仍是
`large-v3 / cuda`。修复后同一命令正确使用 `tiny / cpu`。

修法：`build_asr` 里用 `pick(short, long, default)` 同时认两种键名
（与之前修 Confucius `url=` vs `confucius_url=` 同一个套路）。

### 23.5 验收

- 测试 57 → **79 项全过**（新增 `tests/test_model_paths.py` 22 项：
  来源优先级、显式路径不静默降级、工程内优先于 HF 缓存、
  HF_HOME 生效、build_asr 两种键名兼容）。
- `--with-whisper tiny` 实测下载成功（72 MB）；`--with-whisper small`
  84.6s / 461 MB，走 hf-mirror + 本机代理。
- **断网验证**：`HF_HUB_OFFLINE=1` + `--whisper-model small` 端到端跑通，
  日志显示「whisper 模型（本地目录）」，正常出 2 句文字 → **工程内模型
  完全不依赖网络**。
- 离线且模型缺失时，报错信息给出三条可执行方案（手动下载 / 镜像 / 换小档位）。

### 23.6 教训

11. **「自动下载」不等于「可分发」**。原始形态对单机我够用，但一旦要
    分享，权重就得有明确的、可控的落盘位置。
12. **静默忽略参数比报错更可怕**。键名对不上时程序照跑，只是用了默认值，
    我完全看不出来自己传的 `--whisper-model` 没生效。
13. **分发开源项目要同时想清楚三件事**：代码许可（Apache-2.0，已有）、
    权重许可（MIT / Apache-2.0 / 网易非 OSI，各不相同）、
    以及**权重怎么落到对方机器上**。前两件做好了，第三件最容易漏。






---

## 24. 三个定位与修复（2026-11-09）

三件事：① 选「单人（部分人）」听录文件，窗口里还是满屏【说话人1】，
而且明显还在跑声纹；② 「Whisper 为什么用 tiny？是不是为了省空间和算力？
GitHub 被分享者能不能自己选？」；③ 一男一女两个人的节目，Confucius 后端
全部标成【说话人1】，没分开。

### 24.1 单人模式仍显示【说话人1】：两个独立的原因

**原因 A：文件听录根本没设「不显示说话人」这个开关。**

`LiveSession.__init__` 里有 `self.display.set_show_speaker(not self.single)`，
但**文件听录不走 `LiveSession`**——`UIController._run_file()` 直接构造
`FileTranscriber`，于是 `LiveDisplay.show_speaker` 保持默认的 `True`，
窗口就照常渲染 `[说话人1]`。修法：在 `_run_file()` 里显式
`display.set_show_speaker(not single)`。

**原因 B：单人模式仍在跑完整的声纹分离。**

原 `file_mode.run()` 里是 `Diarizer(num_speakers=-1 if self.single else ...)`，
也就是说单人模式**照样加载 pyannote 分割模型 + CAM++ 声纹模型做全局聚类**，
跑完再把所有段的 speaker 强行改成 1。慢、耗内存，而且从行为上看
「我明明选了单人，程序怎么还在分人」。

修法：单人模式**只做断句**（silero VAD），完全不碰声纹模型：

```python
if self.single:
    vad = build_vad(...)                 # 只要 silero_vad.onnx
    segments = vad_segments(audio, sr, vad=vad, merge_gap=self.merge_gap)
    for s in segments: s.speaker = 1
else:
    dia = Diarizer(...)                   # 原来的全局聚类
```

实测（我那份 165 秒的 `01.wav`）：走 VAD 只断句 **0.6 秒**跑完，
原来跑完整分离要好几秒；窗口、`txt`、`srt` 都不再出现说话人前缀，
状态栏写「单人模式：只断句，不做声纹」。

> `json` 里仍保留 `speaker: 1` —— 那是给下游程序的结构化字段，
> 本来就该和给人看的标签分开（见 `outputs.write_outputs` 的注释）。

### 24.2 附带发现：合并后的单段太长，ASR 会吃亏

VAD 在这份音频上切出 14 段，每段 3.9~14.8 秒（`max_speech_duration=12`
在 sherpa-onnx 里是滞后生效的，会超出一点）。再按 `merge_gap=0.3` 合并，
直接变成 **43 秒 / 45 秒**的巨段——Whisper 的窗是 30 秒、Paraformer 同量级，
超长单段转写质量反而下降。

修法：`merge_adjacent` 增加 `max_len` 上限（文件听录默认 20 秒），
到顶就另起一段。合并后段数 5 → 8，单段长度回到 20 秒以内。

### 24.3 双人节目全判成【说话人1】：根因是「整段算一个声纹」

先复现。写 `scripts/_diag_speaker.py`，完全照抄实时链路
（同样的 silero VAD 参数 → CAM++ 声纹 → `OnlineSpeakerTracker.assign`）：

| 音频 | 段长 | 旧实现结果 |
|---|---|---|
| 官方双人英文 16s | 1.9~2.6s | ✅ 1,1,2,2（正确） |
| 官方四人中文 56.9s | 2.5~13.1s | ⚠️ 长段段被判错（0.185 的跨人相似度也归成同一人） |

再看我那次真实抓取（`live_out/live.rttm`）：**13 段，每段 7~19 秒**，
而且 `voices.json` 里只有 1 个说话人、count=13 —— 说明声纹**算了**，
但每一段的相似度都过了 0.60 阈值，于是永远判成同一个人。

**根因**：广播/直播音频里 VAD 切出的段特别长（电台几乎不停顿，
0.45 秒的静音门槛触发不了，只有换人处才有零点几秒的缝）。
用**整段**（十几秒、可能含好几个人）算一个 192 维声纹，等于把几个人
**平均**成一个向量：这个向量既不像 A 也不像 B，后面每个人跟它比都是
「离最近的中心最近」→ 整场只剩【说话人1】。

用四人中文音频切 13 秒段复现得非常干净：段 1 与段 4 的余弦只有 **0.185**
（明显是不同的人），却都被判成同一个说话人。

**修法（`live_speaker.OnlineSpeakerTracker` v2）**：

1. **切窗取声纹**：每段再切成 3 秒窗（步进 1.5 秒），每窗单独算一个声纹。
   3 秒正好是 CAM++ / 3D-Speaker 的训练窗长，小窗比大段稳得多；
   一段里若混了两个人，至少有一半的窗是纯的。
2. **原型库**：每个说话人保留最多 4 个原型（同一个人的不同音色/信道/情绪），
   匹配时取「与所有说话人所有原型里最像的那个」，不再只跟一个平均向量比。
3. **投票**：一段的标签 = 投票最多的说话人；只有过半的窗都认不出来，
   才算出现了新说话人。

性能：13 秒音频切 7 个窗共 0.143 秒（**90 倍实时**），完全无压力。

**修后验证**（同一套代码、同样的阈值 0.60）：

| 音频 | 旧 | 新 |
|---|---|---|
| 官方双人英文 | 1,1,2,2 | 1,1,2,2（不变） |
| 官方四人中文 | 1,2,1,2,2,2,2（塌缩） | 1,2,1,3,3,4,1 ✅ 与真值一致 |
| 四人中文切 13 秒长段 | 1,2,2,2,2（塌缩） | 1,2,3,2,1 ✅ |
| 我那份 13 段长音频 | 全 1 | 能分出换人 |

「四人中文」那一行的真值是按整段余弦矩阵人工分组的
（{1,3,7} / {4,5} / {6} / {2}），新实现与真值**完全一致**。

新增 `tests/test_speaker_tracker.py`（9 项）用假声纹提取器把这件事钉死：
假提取器对长段做「多人方向平均」，和真 CAM++ 行为一致；
对照旧算法同一输入得 `[1,1,2]`（错），新算法得 `[1,2,2]`（对）。

### 24.4 Whisper 为什么是 tiny：它根本不是默认值

`parse_args` 默认 `--whisper-model large-v3`，`settings.json` 里也是
`large-v3`，`download_models.py --with-whisper` 不带参数时默认也是
`large-v3`。`models/whisper/` 里那两个 `tiny` / `small` 目录，
是**开发调试时下的测试权重**，跟默认档位没关系。

不过「tiny 是最小的一档」这个印象本身是对的，而且对 GitHub 被分享者是
个真问题：权重在 HuggingFace 上，`large-v3` 有 2.88 GB，
**GitHub 单文件 100 MB 上限，权重进不了仓库**，分享者只能告诉对方
「下哪一档」。那就得把取舍摆明白，不能让人猜。

因此做了两件事：

1. **界面里给选择权**：主界面第二行加「**Whisper 档位**」下拉
   （每档标了体积：large-v3 2.9GB / large-v3-turbo 1.6GB / medium 1.4GB /
   small 461MB / base 145MB / tiny 72MB）和「**下载权重…**」按钮，
   后台跑 `scripts/download_models.py --with-whisper <档位>`，
   弹窗报告结果；档位目录可以整个拷给同事，**离线可用**。
2. **README 写清取舍表**，并明确说明「`tiny` 不是默认档位，
   只是最小的一档」，以及三种下权重的路径（界面按钮 / 命令行 / 自动下载）。

### 24.5 验收

- 测试 79 → **97 项全过**（新增 `tests/test_speaker_tracker.py` 9 项，
  改写 `tests/test_file_mode.py` 的单人模式用例：
  用「一被实例化就报错」的替身证明单人模式**没有**碰声纹模型）。
- 单人文件听录端到端（真实 Whisper small + CUDA）：`show_speaker=False`，
  `txt`/`srt` 无说话人前缀，状态栏「单人模式：只断句，不做声纹」。
- 实时链路端到端（`--simulate` 官方四人中文 + `--num-speakers 4`）：
  分出 **4 位说话人**，分组 {1,3,7}/{2}/{4,5}/{6} 与真值一致；
  旧实现同一输入只有 1 位。
- 界面截图核对：第二行的档位下拉与下载按钮排版正常，无截断。
- 诊断脚本 `scripts/check_speakers.py` 正式化：照抄实时链路打印每段
  标签、分数与段间余弦矩阵，并给出「同一说话人 >= 0.7 / 不同人 < 0.5」
  的判读标准；`--chunk 13` 可复现长段场景。分不出人时先跑它。

---

## 25. 两个体验问题：文字稿被覆盖 + 准备阶段无反馈（2026-11-18）

我连续听几场直播时发现：① 每次点开始，上一场的文字稿就没了；
② 点开始后程序要准备十几秒，界面一片平静，分不清是卡住还是在加载。

### 25.1 先确认改的是哪套界面

这一步差点踩坑：`live_ui.py` 第 14~21 行是 try 导入 `live_gui_qt.gui_main`、
失败才回退 `live_gui.gui_main`。本机 **PySide6 6.12.0 已安装，实际 `_UI_BACKEND = "qt"`**，
跑的是 11-11 新增的「曜石 × 火焰」界面 `live_gui_qt.py`（44 KB），
而不是之前 UI_DESIGN.md 1.0 描述的 Tkinter 版。目录里的 `live_out/live.txt`
（11-18 11:24）也证实我用的是这套。教训：**改 UI 前先查实际加载的后端**，
不能沿用上一轮的印象。Tk 版作为回退路径也做了兼容，但只给文字不加动画。

### 25.2 文字稿被覆盖

根因有两处，都在 `live_core.py`：

- `--stem` 默认恒为 `live`，`write_all()` 每次写同名文件；
- `_run_pipeline()` 启动时还有一段 `for fmt: p.unlink()` —— **主动删旧文件**。

修法：新增 `session_stem.py`，命名 `<前缀>_<YYYYmmdd_HHMMSS>`；
同一秒连开两次自动加 `_2` / `_3`（检查**所有** formats，只判 .txt 会漏撞 .srt）；
`safe_base()` 洗掉非法字符、剥掉媒体扩展名、限长 60。
`LiveSession.__init__` 里定好本次 `session_stem`，`args.stem` 同步改写
（args 与 self.args 是同一个 Namespace，后面所有写出点自动跟上）。
`unlink` 那段改成**只在 `--stem-suffix none` 时执行**，默认模式一个历史文件都不删。

`--stem-suffix none` 是显式保留旧行为的开关（批处理脚本要固定文件名时用），
默认 `timestamp`。文件模式由 `UIController.start_file()` 用同一套规则生成 stem，
前缀是原文件名，传给 `FileTranscriber(stem=...)`。

界面右下角新增 `本次: <stem>.txt` 标签（Tk 版追加在状态条末尾），
悬停显示输出目录；`*.json` 的 meta 增加 `session_id` / `session_started_at`，
方便程序追溯哪份稿子是哪场。

实测：两场 `--simulate --duration 8` 连跑，得到
`live_20261118_234930.{txt,srt,json}` 与 `live_20261118_234941.{txt,srt,json}`，
两组都在，旧稿内容完整。

### 25.3 准备阶段「界面很平静」

根因：点开始后 `_running=True`，但 `CaptureStats` 要等 `src.start()` 之后才有值，
在这之前 `stats` 是空 dict，状态条走默认分支显示「设置好选项后点『开始听录』」，
波形是静止待机线。加载 VAD / CAM++ / Whisper large-v3 那十几秒毫无反馈。
文件模式更严重：`task.run()` 全部跑完才 `add_final`，整个转写期间字幕区是空的。

修法分三层：

1. **状态容器**：`LiveDisplay` 增加 `preparing`（`stage/detail/step/total/at`）与
   `session`。两者**刻意不放进 `stats`** —— `set_stats()` 是整体替换，
   采集一开始就冲掉了。`at` 只在第一轮记录，阶段切换不重置等待计时。
2. **引擎上报**：`LiveSession._stage()` 分 5 步（VAD → 声纹 → Paraformer →
   句末精修 → 开音频设备），`src.start()` 后清除；文件模式由 `progress()` 同步推进
   probe / decode / diarize / asr / write。
3. **视图层**：Qt 版新增 `EmberPulse`（70ms 自绘脉冲火点）、`PreparingState`
   （QStackedWidget 第三页，火焰渐变不定态进度条 + 阶段文字 + 步骤圆点 +
   已等待秒数 + 本次文件名）、状态条小进度条与 `session_label`；
   准备期波形给轻微呼吸能量，引擎 pill 显示「正在加载模型…」。

**最关键的边界**：`clear_preparing()` 必须覆盖所有退出路径 —— 正常开始、
`done`、`finally`、控制器 `except`（构造失败 / 运行异常）。否则界面会
**永远卡在加载动画上，比原来的「平静」更糟**。`UIController.start()` /
`start_file()` 的构造段都包了 try/except，失败时清状态 + 写 error + 不设 `_running`，
Qt 的 `_start_click()` 也会检查 `controller.is_running()`，不假装在运行。

设计取舍：进度条用**不定态**而不是伪造百分比 —— 模型加载没有可信的可测进度；
动画是低幅呼吸，颜色之外同时给文字、步数、秒数，不靠颜色单独编码。

### 25.4 顺带修掉一个 flaky 测试

全量测试偶尔在 `test_gui_options.test_tiers_carry_size_hint` 报
`'2.9GB' not found in 'large-v3-turbo 1.6GB'`。根因是测试用
`lab.startswith(tier)` 匹配标签，`large-v3` 会误命中 `large-v3-turbo`，
且 `labels` 是**无序 set**，`hit[0]` 取到谁取决于 hash 随机种子 ——
之前几次通过纯属运气。改成按值精确匹配 `by_value = {val: lab ...}`，
两种 `PYTHONHASHSEED` 下各跑一遍全量均稳定通过。

### 25.5 验收

- 新增 `tests/test_session_stem.py`（11 项）+ `tests/test_preparing.py`（7 项）：
  覆盖命名格式、同秒冲突、全 formats 避让、真实 `write_all()` 两场并存、
  `none` 模式保留旧行为、preparing 生命周期、session 不被 set_stats 冲掉、
  构造失败/运行异常清理、文件模式 stem 传递、pipeline 5 步上报。
- 全量 **115 项 × 2 种哈希种子全过**；真实 Qt GUI 启动冒烟无 traceback。
- 离屏截图字体为 0 会把中文渲染成方块，改用真实 Windows 平台 `win.grab()`
  复核（782 个字体，含 Microsoft YaHei），排版无重叠无截断 →
  `docs/ui-preparing.png`。

---

## 26. 波形不再是假动画：接上真实 PCM（2026-11-19）

### 26.1 我反馈

> 「在听读的时候中间设计了一个波形动画，但它的功能性比较差，我在想是否可以把
> 这个波形与实际听读的音频相关联？让这个波动能实际体现音波的变化。」

确实如此。旧 `WaveWidget.paintEvent()` 里每根柱子是：

```python
noise = sin(i*0.35 + phase*2.1)*0.5 + sin(i*0.13 - phase*1.3)*0.35 + ...
amp = (0.18 + abs(noise)*0.82) * self._energy * sin(i/n*pi)
```

`self._energy` 只是 `stats.peak` 换算出来的**一个标量**。也就是说音量变了
只是整体变高变矮，柱子怎么起伏完全由 phase 决定，跟声音一点关系都没有。
25 章加的准备期呼吸动画同样只表达「程序活着」这一种语义。

### 26.2 三条设计判断

1. **不能走 `LiveDisplay`**。`snapshot()` 每 150ms 才轮询一次，而波形要 40ms 刷新；
   更要命的是 `set_stats()` 是整体替换字典，把高频波形塞进去会被下一次
   `set_stats()` 冲掉 —— 25 章的 `preparing` / `session` 已经踩过这个坑。
   所以 `UIController` 单独持有 `wave_meter`，视图层直接读。
2. **必须在 `AudioSource._put()` 接一次就够**。它同时是 WASAPI 回环、
   sounddevice、文件回放三条后端的共同入口；在那里挂旁路不会漏掉任何一条链路。
3. **形状要保真，不能偷懒画 RMS 包络或假曲线**。每柱存原始采样的 `min/max`
   两个值，画的时候上下各画一半，看到的就是波形本身。唯一处理是固定的
   `tanh(3x)` 软压缩（把 -30dBFS 的语音抬到看得见），**不是 AGC**，
   悄悄话和喊叫的相对强弱不会被拉平；硬 `clip` 会把大信号削成等高实心块，弃用。

### 26.3 改动清单

| 文件 | 改动 |
|---|---|
| `src/speakercaption/wave_meter.py` | **新增**：`wave_envelope()`、`WaveMeter`（512 柱环形缓冲 / `WaveFrame` / 4 种模式）、`level_db()` |
| `live_audio.py` | `AudioSource(..., wave_meter=None)`；`_put()` 推真实 PCM，异常时置空旁路但**不中断采集** |
| `live_core.py` | `LiveSession(..., wave_meter=None)`；`src.start()` 前 `begin_live()`（顺带清上一场历史）、`finally` 里 `finish()` |
| `file_mode.py` | `FileTranscriber(..., wave_meter=None)`；解码后 `begin_file(audio, sr)`，每转完一段按 `s.end/duration` 推播放头，`done` 置 100% |
| `live_ui.py` | `UIController.wave_meter = WaveMeter()`；`start()` / `start_file()` 先 `begin_preparing()`，传给会话 / 转写器，构造失败与线程 `finally` 都 `finish()` |
| `live_gui_qt.py` | `WaveWidget` 重写：40ms 拉帧画双极柱、文件总览 + 点亮 + 播放头、右上角 dB / 百分比；无 meter 时保留旧 `set_energy()` 兼容路径；`LevelBars` 改用真实 `level_db()` |
| `README.md` / `docs/UI_DESIGN.md` | 特性条目、新增「波形跟着声音走」章节 / §5.7，文档升 1.2 |

**文件模式为什么是「整段总览 + 进度」而不是模仿实时滚动**：文件转写不等比例于
播放时长（GPU 快时能几十倍速跑完），硬要做滚动动画只会造出一段和音频对不上的
假运动。整段总览是这个文件真实的声音分布，播放头按段落**结束时间**推进，
长句和空白在画面上的位置与音频一致，是可以核对的信息。

**生命周期**：

```
点开始        → begin_preparing()   呼吸待机（还没音频可画）
模型加载完    → begin_live()        清历史，进来的每块 PCM 开始滚动
文件解码完    → begin_file(audio)   整段总览，播放头 0%
停止/出错/done→ finish()           实时回待机；文件保留总览
```

### 26.4 几个容易踩的点

- **`push()` 在 PortAudio 回调线程**：只做一次 reshape + min/max（4096 样本量级，
  几十微秒），锁内只写环；不做日志、不碰 Qt、不分配大对象。
- **峰值必须取压缩前的原始采样**。一开始想从 `env` 反推 `peak`（除以 gain），
  但 `tanh` 已经饱和，反推不回真实峰值；改成 `np.max(np.abs(a))`。
- **文件包络降采样要保住尖峰**：`FILE_BINS=512 → 110` 用 `edges` 分段后对
  min 取最小 / max 取最大，不能用 `linspace` 抽点，否则会漏掉短促的爆发音。
- **离屏截图无字体**：`QFontDatabase.families()` 为空时画文字全是豆腐块，
  加了 `_has_fonts` 守卫；整窗截图仍用真实 Windows 平台 `win.grab()` 复核。
- **`--no-gui` 零开销**：CLI 路径压根不创建 meter，`_put()` 只多一个 `if None`。

### 26.5 验收

- 新增 `tests/test_wave_meter.py` **23 项**：包络保真 / 静音严格为 0 /
  幅度单调不削顶 / 环形回绕 / 并发读写 / 四种模式 / 进度钳制 /
  `_put` 接线与容错 / 控制器传参 / 文件转写进度 / Qt 控件取帧 /
  走完整条 `LiveSession --simulate` 的端到端「平线 + 起伏」断言。
- 全量 **138 项 × `PYTHONHASHSEED=0/1/7` 三种种子全过**（约 40~50s/轮）。
- 真实 Windows 平台 `QApplication` 生成整窗截图 `docs/ui-waveform.png`：
  1200×620，中文、时间戳、说话人配色、状态条、两段语音波形与中间停顿都正常；
  组件截图 `ui-waveform-live.png` / `ui-waveform-file.png` /
  `ui-waveform-standby.png`。
- 端到端实测：测试 wav 第 1 秒近静音、第 2 秒有语音（`--duration 2.5`），
  波形帧里同时存在 `<0.01` 的平线柱和 `>0.3` 的语音柱；`有声占比 42.3%`
  与画面一致，证实数据确实来自采集而非合成。
