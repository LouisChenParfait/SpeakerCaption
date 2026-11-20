# 输出格式样例

下面四段内容取自本工程对官方 2/4 说话人测试音频的**真实运行结果**（见
[`docs/DEVLOG.md`](../docs/DEVLOG.md) §8/§9）。把自己的音频放进任意目录，
运行时用 `--outdir` 指定输出位置即可；`examples/` 不存放音频文件。

## `*.txt`（默认纯文字，一句一段）

```
说话人1：这是一个测试说话人日志的音频，下面我们来播放一些测试音频。
说话人2：非常累的时候就会站在阳台上吹吹风。
说话人3：和你们一起走过很多很多的路，坚持到底。
```

`--timestamps` 开启后：

```
[00:00.3] 说话人1：这是一个测试说话人日志的音频下面我们来播放一些测试音频
[00:07.0] 说话人2：非常累的时候就会站在阳台上吹吹风
```

`--speaker-names "1=主持人,2=嘉宾"` 后：

```
主持人：这是一个测试说话人日志的音频...
嘉宾：非常累的时候就会站在阳台上吹吹风...
```

## `*.srt`

```srt
1
00:00:00,318 --> 00:00:06,865
[说话人1] 这是一个测试说话人日志的音频下面我们来播放一些测试音频

2
00:00:07,017 --> 00:00:10,747
[说话人2] 非常累的时候就会站在阳台上吹吹风

3
00:00:11,455 --> 00:00:13,632
[说话人2] 看远处的灯火明明暗暗
```

## `*.json`（节选）

```json
{
  "version": "0.1.0",
  "duration_seconds": 56.861,
  "num_speakers_detected": 4,
  "asr_backend": "whisper",
  "diarize_seconds": 3.602,
  "speakers": [
    {"id": 1, "name": "说话人1", "total_seconds": 11.323}
  ],
  "segments": [
    {"start": 0.318, "end": 6.865, "duration": 6.548,
     "speaker": 1, "speaker_name": "说话人1", "text": "这是一个测试..."}
  ]
}
```

## `*.rttm`

```text
SPEAKER interview 1 0.318 6.548 <NA> <NA> speaker_01 <NA> <NA>
SPEAKER interview 1 7.017 3.729 <NA> <NA> speaker_02 <NA> <NA>
```

标准 RTTM 可直接导入标注/评测工具；不带文字，适合 `--asr none` 时对接其它 ASR。

## 复现

```bash
python scripts/download_models.py --with-test-audio
python diarize.py tests/data/1-two-speakers-en.wav --num-speakers 2
python diarize.py tests/data/0-four-speakers-zh.wav --num-speakers 4 --timestamps
```

字形与翻译（`live.py` 的图形界面里就是这几个勾选框）：

```bash
# 默认：自动判语种、英文保持英文、中文统一简体、不加时间戳
python diarize.py podcast.wav --num-speakers 2

# 要繁体
python diarize.py podcast.wav --num-speakers 2 --script traditional

# 要原样（不做任何简繁转换）
python diarize.py podcast.wav --num-speakers 2 --script as-is

# 要译文（需本机 Ollama，仅对外文占比高的句子生效）
python diarize.py podcast.wav --num-speakers 2 --translate
```

示意（译文随所选本地模型而变）：

```text
说话人1：A pencil with black lead writes best.          # 默认
说话人1：黑铅铅笔写得最好。                              # --translate
```

