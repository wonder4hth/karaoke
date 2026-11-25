"""卡拉OK流水线：下载 → 分离人声 → 歌词 → 逐字对齐 → 拼音/翻译 → 渲染"""
import functools, hashlib, json, os, re, shutil, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK, OUT, MODELS, BIN = (ROOT / d for d in ("work", "output", "models", "bin"))
for _d in (WORK, OUT, MODELS, BIN):
    _d.mkdir(exist_ok=True)

import imageio_ffmpeg
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
_link = BIN / "ffmpeg"
if not _link.exists():
    _link.unlink(missing_ok=True)
    _link.symlink_to(FFMPEG)
_probe = BIN / "ffprobe"   # Gradio 播放器需要
if not _probe.exists():
    from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise
    _probe.unlink(missing_ok=True)
    _probe.symlink_to(get_or_fetch_platform_executables_else_raise()[1])
# audio-separator 从 PATH 找 ffmpeg；yt-dlp 从 PATH 找 deno（在 .venv/bin）
os.environ["PATH"] = f"{BIN}:{ROOT / '.venv/bin'}:{os.environ['PATH']}"
os.environ.setdefault("TORCH_HOME", str(MODELS / "torch"))

HAN = re.compile(r"[㐀-鿿]")
TOKEN = re.compile(r"[㐀-鿿]|[A-Za-zÀ-ÿ]+(?:['’][A-Za-zÀ-ÿ]+)*|\d")


def ff(*args):
    p = subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *map(str, args)],
                       capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError("ffmpeg 失败：" + p.stderr[-1500:])


def free_gpu():
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()


def simplify(s):
    return _opencc().convert(s)


@functools.lru_cache
def _opencc():
    from opencc import OpenCC
    return OpenCC("t2s")


# ---------- 1. 下载 ----------
def fetch(source, want_video=False):
    """source 为链接或本地文件。返回 (工作目录, 信息)"""
    source = str(source).strip()
    if os.path.isfile(source):
        with open(source, "rb") as f:
            key = "file_" + hashlib.file_digest(f, "sha1").hexdigest()[:12]
        d = WORK / key
        d.mkdir(exist_ok=True)
        src, info = Path(source), dict(title=Path(source).stem, track="", artist="")
    else:
        if not re.match(r"https?://", source):
            raise ValueError("请输入 http(s) 链接或上传文件")
        import yt_dlp
        base = dict(quiet=True, noprogress=True, noplaylist=True, no_warnings=True)
        with yt_dlp.YoutubeDL(base) as y:
            m = y.extract_info(source, download=False)
        if m.get("entries"):
            m = list(m["entries"])[0]
        key = re.sub(r"\W", "_", f"{m['extractor_key']}_{m['id']}")
        d = WORK / key
        d.mkdir(exist_ok=True)

        def dl(name, fmt):
            got = [p for p in d.glob(name + ".*") if p.suffix != ".part"]
            if not got:
                with yt_dlp.YoutubeDL({**base, "format": fmt, "outtmpl": str(d / f"{name}.%(ext)s")}) as y:
                    y.download([m["webpage_url"]])
                got = [p for p in d.glob(name + ".*") if p.suffix != ".part"]
            return got[0]
        src = dl("source", "bestaudio/best")
        if want_video:
            dl("video", "bv*[height<=1080]/b[height<=1080]/b")
        if m.get("thumbnail") and not (d / "cover.jpg").exists():
            try:
                import requests
                r = requests.get(m["thumbnail"], timeout=20)
                r.raise_for_status()
                (d / "cover.raw").write_bytes(r.content)
                ff("-i", d / "cover.raw", "-frames:v", 1, d / "cover.jpg")
            except Exception as e:
                print("封面下载失败，改用纯色背景：", e)
        info = dict(title=m.get("title") or "", track=m.get("track") or "",
                    artist=m.get("artist") or "", channel=m.get("channel") or "")
    if os.path.isfile(source) and want_video and not (d / "video.mkv").exists():
        try:
            ff("-i", src, "-map", "0:V:0", "-c", "copy", d / "video.mkv")
        except RuntimeError:
            (d / "video.mkv").unlink(missing_ok=True)
    wav = d / "audio.wav"
    if not wav.exists():
        ff("-i", src, "-vn", "-ac", 2, "-ar", 44100, d / "audio_tmp.wav")
        (d / "audio_tmp.wav").rename(wav)
    import soundfile as sf
    info["duration"] = sf.info(str(wav)).duration
    info["key"] = key
    return d, info


# ---------- 2. 去人声 ----------
def separate(d):
    inst, voc = d / "instrumental.wav", d / "vocals.wav"
    if not (inst.exists() and voc.exists()):
        from audio_separator.separator import Separator
        s = Separator(output_dir=str(d), model_file_dir=str(MODELS), output_format="WAV", log_level=30)
        s.load_model("model_bs_roformer_ep_317_sdr_12.9755.ckpt")
        s.separate(str(d / "audio.wav"), {"Vocals": "vocals", "Instrumental": "instrumental"})
        del s
        free_gpu()
        if not (inst.exists() and voc.exists()):
            raise RuntimeError("人声分离没有产出文件")
    return inst, voc


# ---------- 2b. 链接本身是伴奏时：找原唱来对轴 ----------
INSTR = re.compile(r"伴奏|instrumental|off[ -]?vocal|karaoke|カラオケ|minus one|无人声|無人聲", re.I)
NOTORIG = re.compile(r"翻唱|cover|live|现场|現場|remix|piano|钢琴|鋼琴|slowed|sped|nightcore|1 ?hour|教学|tutorial", re.I)


def is_instrumental(info):
    return bool(INSTR.search(f"{info['title']} {info.get('track', '')}"))


def offset_between(ref_inst, tgt_inst):
    """两版伴奏波形互相关（GCC-PHAT）→ (平移秒数, 是否可信)。目标时间 = 参考时间 + 平移"""
    import numpy as np, soundfile as sf
    def load(p):
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = x.mean(1)
        n = len(x) // 4 * 4
        return x[:n].reshape(-1, 4).mean(1), sr / 4
    (a, sr), (b, _) = load(ref_inst), load(tgt_inst)

    def lag(i, j):
        u, v = a[i:j], b[i:j]
        n = 1 << (len(u) + len(v)).bit_length()
        X = np.fft.rfft(v, n) * np.conj(np.fft.rfft(u, n))
        c = np.fft.irfft(X / (np.abs(X) + 1e-9), n)
        m = min(int(30 * sr), n // 2 - 1)
        c = np.concatenate([c[-m:], c[:m + 1]])
        k = int(np.abs(c).argmax())
        return (k - m) / sr, float(np.abs(c[k]) / (c.std() + 1e-12))
    L = min(len(a), len(b))
    full, half1, half2 = lag(0, L), lag(0, L // 2), lag(L // 2, L)
    print("平移估计（秒, 峰值强度）：", full, half1, half2)
    ok = full[1] > 12 and all(abs(h[0] - full[0]) < 0.06 for h in (half1, half2))
    return full[0], ok


def use_reference(d, url):
    """下载参考原唱并分离 → (人声路径, 平移)；不是同一版编曲则报错"""
    rd, rinfo = fetch(url)
    if rd == d:
        raise RuntimeError("参考原唱与歌曲是同一个链接")
    rinst, rvoc = separate(rd)
    shift, ok = offset_between(rinst, d / "instrumental.wav")
    if not ok:
        raise RuntimeError(f"“{rinfo['title']}”与该伴奏不是同一版编曲，无法对轴")
    return rvoc, shift, rinfo["title"]


def find_reference(d, info, track, artist):
    import yt_dlp
    q = f"{track} {artist}".strip()
    with yt_dlp.YoutubeDL(dict(quiet=True, no_warnings=True, extract_flat=True)) as y:
        es = y.extract_info(f"ytsearch10:{q}", download=False)["entries"]
    es = [e for e in es if not INSTR.search(e["title"]) and not NOTORIG.search(e["title"])
          and abs((e.get("duration") or 0) - info["duration"]) <= 20]
    es.sort(key=lambda e: (e.get("channel") != info.get("channel"), abs(e["duration"] - info["duration"])))
    for e in es[:3]:
        try:
            return use_reference(d, f"https://www.youtube.com/watch?v={e['id']}")
        except Exception as ex:
            print("参考原唱不可用：", e["title"], ex)
    raise RuntimeError("这是伴奏，没有主唱可以对轴，也没有自动找到同版原唱。请在“参考原唱链接”里填一个同编曲的原唱")


# ---------- Claude（通过本机 claude CLI，使用你的登录） ----------
def ask_claude(system, prompt, model="sonnet"):
    p = subprocess.run(["claude", "-p", "--model", model, "--tools", "", "--setting-sources", "",
                        "--strict-mcp-config", "--no-session-persistence",
                        "--system-prompt", system, "--output-format", "text"],
                       input=prompt, capture_output=True, text=True, timeout=300, cwd=WORK)
    if p.returncode:
        raise RuntimeError("claude 调用失败：" + (p.stderr or p.stdout)[-400:])
    t = p.stdout
    return json.loads(t[min(i for i in (t.find("["), t.find("{")) if i >= 0):max(t.rfind("]"), t.rfind("}")) + 1])


def guess_song(info):
    if info.get("track") and info.get("artist"):
        return re.sub(r"\s*[(（\[【].*?[)）\]】]", "", info["track"]).strip() or info["track"], info["artist"].split(",")[0].strip()
    try:
        r = ask_claude('从视频标题中提取歌名和歌手。只输出 JSON：{"track":"","artist":""}，不确定的留空。',
                       json.dumps({k: info.get(k, "") for k in ("title", "channel")}, ensure_ascii=False), "haiku")
        return str(r.get("track", "")), str(r.get("artist", ""))
    except Exception as e:
        print("歌名识别失败：", e)
        m = re.search(r"《(.+?)》", info["title"])
        return (m[1] if m else info["title"]), ""


# ---------- 3. 歌词 ----------
LRC = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
CREDIT = re.compile(r"^(?:作?[词曲詞]|作词|作詞|作曲|编曲|編曲|制作人?|製作人?|监制|監製|混音|录音|錄音|母带|吉他|和声|和聲|"
                    r"演唱|歌手|原唱|出品|发行|[A-Za-z ]*(?:lyrics?|composer|producer|arranger)[A-Za-z ]*)[^:：]{0,12}[:：]", re.I)


def parse_lyrics(text):
    """LRC 或纯文本 → [(时间或None, 文本)]"""
    res = []
    for raw in text.splitlines():
        ts = LRC.findall(raw)
        t = simplify(LRC.sub("", raw)).strip()
        if not t or re.match(r"^\[\w+:.*\]$", t) or CREDIT.match(t) or not TOKEN.search(t):
            continue
        for m, s in ts or [(None, None)]:
            res.append((int(m) * 60 + float(s) if m else None, t))
    if res and all(t is not None for t, _ in res):
        res.sort(key=lambda x: x[0])
    return res


def search_lyrics(track, artist, duration):
    import requests
    track = re.sub(r"[(（\[【].*?[)）\]】]", "", track).strip() or track  # 去掉“(重制版)”之类
    hits, seen = [], set()
    for q in ({"track_name": track, "artist_name": artist}, {"q": f"{track} {artist}".strip()}):
        try:
            r = requests.get("https://lrclib.net/api/search", params={k: v for k, v in q.items() if v},
                             timeout=20, headers={"User-Agent": "karaoke-maker"})
            r.raise_for_status()
        except Exception as e:
            print("LRCLIB：", e)
            continue
        for h in r.json():
            if h["id"] not in seen and (h.get("syncedLyrics") or h.get("plainLyrics")):
                seen.add(h["id"])
                hits.append(h)
    want = simplify(track).lower()

    def score(h):
        name = simplify(h["trackName"] or "").lower()
        if not want or (want not in name and name not in want):
            return 1e9
        body = h.get("syncedLyrics") or h["plainLyrics"]
        dd = abs((h.get("duration") or 0) - duration)
        sc = dd + (0 if h.get("syncedLyrics") else 8)
        if artist and simplify(artist).lower() not in simplify(h["artistName"] or "").lower():
            sc += 25
        if HAN.search(track) and not HAN.search(body):  # 全拼音的歌词不要
            sc += 500
        return sc
    hits.sort(key=score)
    if hits and score(hits[0]) < 120:
        h = hits[0]
        return h.get("syncedLyrics") or h["plainLyrics"], f"LRCLIB：{h['trackName']} - {h['artistName']}"
    return None


HALLU = re.compile(r"字幕|Amara|李宗盛|谢谢观看|感谢观看|请不吝点赞|订阅|www\.|For more information|Thanks for watching", re.I)


def transcribe(voc, title=""):
    """title 含汉字时按中文识别，否则自动判断语言"""
    import torch  # 先加载 torch 自带的 CUDA 库
    from faster_whisper import WhisperModel
    m = WhisperModel("large-v3", device="cuda", compute_type="float16", download_root=str(MODELS / "whisper"))
    # 自己解码：faster-whisper 与新版 PyAV 不兼容
    import soundfile as sf, torchaudio
    x, sr = sf.read(str(voc), dtype="float32", always_2d=True)
    x = torchaudio.functional.resample(torch.from_numpy(x.mean(1)), sr, 16000).numpy()
    segs, _ = m.transcribe(x, vad_filter=False, language="zh" if HAN.search(title) else None,
                           language_detection_segments=4, condition_on_previous_text=False,
                           word_timestamps=True)
    lines, cur = [], []

    def flush():
        t = "".join(w.word for w in cur).strip()
        if t and TOKEN.search(t):
            lines.append((cur[0].start, t))
        cur.clear()
    for s in segs:
        if HALLU.search(s.text):
            continue
        a, b = int(s.start * 16000), int(s.end * 16000)
        if abs(x[a:b]).max(initial=0) < 0.02:   # 静音处的文字都是幻觉
            continue
        for w in s.words:
            if cur and (w.start - cur[-1].end > 0.8 or len("".join(x.word for x in cur)) > 18):
                flush()
            cur.append(w)
        flush()
    del m
    free_gpu()
    return "\n".join(f"[{int(t // 60):02d}:{t % 60:05.2f}]{x}" for t, x in lines)


# ---------- 4. 逐字对齐 ----------
DIGITS = dict(zip("0123456789", "ling yi er san si wu liu qi ba jiu".split()))


def line_pinyin(text):
    from pypinyin import lazy_pinyin, Style
    return lazy_pinyin(text, style=Style.TONE, errors=lambda x: list(x))


def tokenize(text):
    """→ [{t 文本, r 罗马音, py 拼音, sp 前面是否有间隔}]"""
    import unicodedata
    py, toks, prev = line_pinyin(text), [], 0
    for m in TOKEN.finditer(text):
        t = m[0]
        han = bool(HAN.match(t))
        p = py[m.start()] if han and len(py) == len(text) else ("".join(line_pinyin(t)) if han else "")
        r = DIGITS.get(t) or (p if han else t)
        r = "".join(c for c in unicodedata.normalize("NFD", r.lower().replace("’", "'")) if c in "abcdefghijklmnopqrstuvwxyz'")
        toks.append(dict(t=t, r=r or "a", py=p, sp=bool(text[prev:m.start()].strip(" ") or text[prev:m.start()])))
        prev = m.end()
    if toks:
        toks[0]["sp"] = False
    return toks


class Aligner:
    """MMS 强制对齐。整首一次对齐，行间插 * 吸收歌词以外的声音"""

    def __init__(self, voc):
        import numpy as np, soundfile as sf, torch, torchaudio
        bundle = torchaudio.pipelines.MMS_FA
        x, sr = sf.read(str(voc), dtype="float32")
        x = torch.from_numpy(x.mean(1) if x.ndim > 1 else x)
        x = torchaudio.functional.resample(x, sr, 16000)
        model = bundle.get_model(with_star=True).cuda().eval()
        win, pad, outs = 16000 * 24, 16000 * 2, []
        with torch.inference_mode():
            for s in range(0, len(x), win):
                a, b = max(0, s - pad), min(len(x), s + win + pad)
                seg = x[a:b]
                if len(seg) < 800:
                    break
                seg = seg / (seg.abs().max() + 1e-5) * 0.7 if seg.abs().max() > 1e-3 else seg
                e = model(seg[None].cuda())[0][0].cpu()
                n0 = round((s - a) / 320)
                outs.append(e[n0:n0 + (min(len(x), s + win) - s) // 320])
        del model
        free_gpu()
        self.em = torch.cat(outs).contiguous()
        self.dict = bundle.get_dict(star="*")
        self.sec = len(x) / 16000 / len(self.em)
        hop = 320
        rms = x[:len(x) // hop * hop].reshape(-1, hop).pow(2).mean(1).sqrt().numpy()
        self.rms = np.convolve(rms, np.ones(5) / 5, "same")
        self.thr = max(float(np.percentile(self.rms, 95)) * 0.06, 1e-3)

    def align(self, lines, t0=0.0, t1=None):
        """lines: 每行 token 列表，就地写入 s/e"""
        import torch
        from torchaudio.functional import forced_align, merge_tokens
        a, b = max(0, int(t0 / self.sec)), min(len(self.em), int(t1 / self.sec) if t1 else len(self.em))
        ids, owner = [], []
        for ln in lines:
            ids.append(self.dict["*"]); owner.append(None)
            for tk in ln:
                for c in tk["r"]:
                    ids.append(self.dict[c]); owner.append(tk)
        ids.append(self.dict["*"]); owner.append(None)
        if b - a < len(ids) * 2:
            return False
        ali, sc = forced_align(self.em[None, a:b], torch.tensor([ids], dtype=torch.int32), blank=0)
        spans = merge_tokens(ali[0], sc[0].exp())
        if len(spans) != len(ids):
            return False
        for ln in lines:
            for tk in ln:
                tk.pop("s", None); tk["sc"] = []
        for sp, tk in zip(spans, owner):
            if tk is None:
                continue
            tk.setdefault("s", (a + sp.start) * self.sec)
            tk["e"] = (a + sp.end) * self.sec
            tk["sc"].append(float(sp.score))
        return True

    def voiced_until(self, t, limit):
        i, quiet = int(t / 0.02), 0
        while i < len(self.rms) and i * 0.02 < limit:
            quiet = quiet + 1 if self.rms[i] < self.thr else 0
            if quiet >= 6:
                return max(t, (i - 5) * 0.02)
            i += 1
        return min(i * 0.02, limit)


def fix_pinyin(lines, title):
    """多音字由 Claude 结合上下文选读音；只接受字典里该字真实存在的读音"""
    from pypinyin import pinyin, Style
    ask, idx = [], {}
    for l in lines:
        if l["text"] in idx:
            continue
        opts = {}
        for i, t in enumerate(l["toks"]):
            if t["py"]:
                c = list(dict.fromkeys([t["py"], *pinyin(t["t"], style=Style.TONE, heteronym=True)[0]]))
                if len(c) > 1:
                    opts[str(i)] = {"字": t["t"], "默认": t["py"], "可选": c}
        idx[l["text"]] = opts
        if opts:
            ask.append({"行": l["text"], "多音字": opts})
    if not ask:
        return
    cache = WORK / ("py_" + hashlib.sha1("\n".join([title, *idx]).encode()).hexdigest()[:16] + ".json")
    if cache.exists():
        fix = json.loads(cache.read_text())
    else:
        fix = ask_claude("你是普通话正音专家。逐行根据歌词语境判断多音字读音（注意分词，如“当晚风”是“当+晚风”读 dāng，"
                         "“发线”指头发读 fà；助词 的/了/着/子 一般读轻声；不要做“一/不”变调以外的改动）。"
                         "只列出默认读音错误、需要更正的字，读音必须来自“可选”。"
                         '只输出 JSON 数组：[{"行":"原文","序号":"3","读音":"dāng"}]，没有就输出 []。',
                         json.dumps(ask, ensure_ascii=False))
        cache.write_text(json.dumps(fix, ensure_ascii=False))
    for f in fix:
        opt = idx.get(f.get("行"), {}).get(str(f.get("序号")))
        if opt and f.get("读音") in opt["可选"]:
            for l in lines:
                if l["text"] == f["行"]:
                    l["toks"][int(f["序号"])]["py"] = f["读音"]


def suspicious(l):
    """这一行的对轴是否可疑（置信度极低，或整行被挤在一瞬间）"""
    return l["conf"] < 0.1 or (l["end"] - l["start"]) / len(l["toks"]) < 0.09


def align_lyrics(voc, lyrics, shift=0.0):
    """shift: voc 来自参考原唱时，平移到目标伴奏的时间"""
    import statistics as st
    al = Aligner(voc)
    toks = [tokenize(t) for _, t in lyrics]
    if not al.align(toks):
        raise RuntimeError("歌词比歌曲还长，无法对齐，请检查歌词是否匹配")
    lrc = [t for t, _ in lyrics]
    if all(t is not None for t in lrc) and len(lrc) > 3:
        # 歌词库时间轴可能整体偏移：取中位数偏移，偏离太远的行在小窗口里重新对齐
        off = st.median(tk[0]["s"] - t for tk, t in zip(toks, lrc))
        dev = [abs(tk[0]["s"] - t - off) for tk, t in zip(toks, lrc)]
        if sum(x < 1.5 for x in dev) > len(dev) * 0.5:
            end = len(al.em) * al.sec
            for i, (tk, t) in enumerate(zip(toks, lrc)):
                if dev[i] > 2.0:
                    nxt = lrc[i + 1] + off if i + 1 < len(lrc) else end
                    al.align([tk], max(0, t + off - 0.6), min(end, nxt + 0.4, t + off + 15))
    res = []
    for i, (tk, (_, text)) in enumerate(zip(toks, lyrics)):
        later = [x[0]["s"] for x in toks[i + 1:i + 2]]
        hard = later[0] - 0.02 if later and later[0] > tk[-1]["e"] else tk[-1]["e"] + 4
        for x, y in zip(tk, tk[1:]):
            x["e"] = max(x["s"] + 0.04, min(y["s"], al.voiced_until(x["e"], y["s"])) if y["s"] - x["e"] > 0.5 else y["s"])
        tk[-1]["e"] = max(tk[-1]["e"], min(al.voiced_until(tk[-1]["e"], hard), tk[-1]["e"] + 4))
        sc = [c for x in tk for c in x.pop("sc")]
        for x in tk:
            x["s"], x["e"] = max(0, x["s"] + shift), max(0.02, x["e"] + shift)
        res.append(dict(text=text, start=tk[0]["s"], end=tk[-1]["e"], conf=sum(sc) / len(sc), toks=tk, tr=""))
    return res


# ---------- 5. 翻译 ----------
def translate(lines, title):
    uniq = list(dict.fromkeys(l["text"] for l in lines))
    cache = WORK / ("tr_" + hashlib.sha1("\n".join([title, *uniq]).encode()).hexdigest()[:16] + ".json")
    if cache.exists():
        m = json.loads(cache.read_text())
    else:
        sys_p = ("你是歌词译者。输入是 JSON 数组，每项标明目标语言，必须严格遵守。译成英文时自然、简洁、有诗意，忠于原意、不误解词义（如“发线”是发丝轮廓而非发际线；人称未写明时不要臆加）（单行，尽量 12 词以内）；"
                 "译成中文时不要保留英文原句。结合整首歌的上下文。只输出与输入等长、顺序一致的 JSON 字符串数组。")
        out = ask_claude(sys_p, f"歌名：{title}\n" + json.dumps([{"译成": "英文" if HAN.search(u) else "简体中文", "行": u} for u in uniq], ensure_ascii=False))
        if not isinstance(out, list) or len(out) != len(uniq):
            raise RuntimeError("翻译行数与歌词不一致，请重试")
        m = dict(zip(uniq, map(str, out)))
        cache.write_text(json.dumps(m, ensure_ascii=False))
    for l in lines:
        l["tr"] = m[l["text"]]


# ---------- 6. 字幕 ----------
W, H = 1920, 1080
WIN_FONTS = Path("/mnt/c/Windows/Fonts")


@functools.lru_cache
def font():
    """→ (文件, 字体名, 目录)"""
    fd = MODELS / "fonts"
    fd.mkdir(exist_ok=True)
    for f in ("msyhbd.ttc", "simhei.ttf"):
        if not (fd / f).exists() and (WIN_FONTS / f).exists():
            shutil.copyfile(WIN_FONTS / f, fd / f)
    for f in [*fd.glob("*.tt?"), *fd.glob("*.otf"), Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")]:
        if f.exists():
            from PIL import ImageFont
            return f, ImageFont.truetype(str(f), 100).getname()[0], f.parent
    raise RuntimeError("没有中文字体，请放一个到 models/fonts/")


@functools.lru_cache
def _pil():
    from PIL import ImageFont
    return ImageFont.truetype(str(font()[0]), 1000)


def width(text, size):
    f = _pil()
    asc, desc = f.getmetrics()   # libass 的字号 = ascent+descent
    return f.getlength(text) * size / (asc + desc)


def ts(t):
    c = max(0, round(t * 100))
    return f"{c // 360000}:{c // 6000 % 60:02d}:{c // 100 % 60:02d}.{c % 100:02d}"


def esc(s):
    return s.replace("\\", "＼").replace("{", "（").replace("}", "）").replace("\n", " ")


def make_ass(lines, title, artist, path):
    name = font()[1]
    ev = []

    def add(a, b, style, x, y, body, layer=1):
        if b > a:
            ev.append(f"Dialogue: {layer},{ts(a)},{ts(b)},{style},,0,0,0,,{{\\an5\\pos({x:.0f},{y:.0f}){body}")

    def fit(text, size, maxw):
        w = width(text, size)
        return int(size * maxw / w) if w > maxw else size

    n = len(lines)
    lines = sorted(lines, key=lambda l: l["start"])
    # 分段：间奏超过 6 秒另起一段
    para = [0] * n
    for i in range(1, n):
        para[i] = para[i - 1] + (lines[i]["start"] - lines[i - 1]["end"] > 6)
    slot = []
    for i in range(n):
        slot.append(0 if i == 0 or para[i] != para[i - 1] else 1 - slot[-1])
    show, hide = [], []
    for i, l in enumerate(lines):
        if i == 0 or para[i] != para[i - 1]:
            a = max(0, l["start"] - 4.0)
            if i:
                a = max(a, lines[i - 1]["end"] + 0.4)
        elif i == 1 or para[i - 2] != para[i]:
            a = show[i - 1]
        else:
            a = lines[i - 2]["end"] + 0.25
        show.append(min(a, l["start"]))
        hide.append(l["end"] + (0.2 if i + 2 < n and para[i + 2] == para[i] else 1.2))
    for i in range(n):   # 同一位置、以及跨段落绝不重叠
        for j in range(i + 1, min(n, i + 3)):
            if slot[j] == slot[i] or para[j] != para[i]:
                hide[i] = max(lines[i]["start"], min(hide[i], show[j]))

    first = lines[0]["start"] if n else 8
    if first > 3.5:
        e = min(first - 2, 6) if first > 6 else first - 1.5
        show = [max(s, e) for s in show]
        add(0.3, e, "Title", W / 2, 450, "\\fad(400,400)}" + esc(title), 2)
        if artist:
            add(0.3, e, "Tr", W / 2, 590, "\\fad(400,400)}" + esc(artist), 2)

    for i, l in enumerate(lines):
        a, b, toks = show[i], hide[i], l["toks"]
        base = 320 if slot[i] == 0 else 730
        S, P = 112, 46
        for _ in range(2):
            gap = S * 0.12
            ws = [max(width(t["t"], S), width(t["py"], P) if t["py"] else 0) + gap for t in toks]
            sps = [S * 0.4 if t["sp"] else 0 for t in toks]
            total = sum(ws) + sum(sps)
            if total <= W - 160:
                break
            k = (W - 160) / total
            S, P = S * k, P * k
        x = (W - total) / 2
        for t, w, sp in zip(toks, ws, sps):
            x += sp
            d0, d1 = max(0, round((t["s"] - a) * 100)), max(1, round((t["e"] - t["s"]) * 100))
            k = f"\\k{d0}}}{{\\kf{d1}}}"
            add(a, b, "Han", x + w / 2, base, f"\\fs{S:.0f}\\fad(150,150){k}{esc(t['t'])}")
            if t["py"]:
                add(a, b, "Py", x + w / 2, base - S * 0.5 - P * 0.6, f"\\fs{P:.0f}\\fad(150,150){k}{esc(t['py'])}")
            x += w
        tr = l.get("tr", "")
        if tr:
            size = fit(tr, 56, W - 160)
            if HAN.search(tr):   # 英文歌词：译文是中文，也配拼音
                tt = tokenize(tr)
                tw = [max(width(t["t"], 60), width(t["py"], 34)) + 8 for t in tt]
                k = min(1, (W - 160) / max(1, sum(tw)))
                x = (W - sum(tw) * k) / 2
                for t, w in zip(tt, tw):
                    add(a, b, "Py", x + w * k / 2, base + 92, f"\\fs{34 * k:.0f}\\1c&HFFFFFF&\\2c&HFFFFFF&\\fad(150,150)}}" + t["py"])
                    add(a, b, "Tr", x + w * k / 2, base + 142, f"\\fs{60 * k:.0f}\\fad(150,150)}}" + esc(t["t"]))
                    x += w * k
            else:
                add(a, b, "Tr", W / 2, base + 108, f"\\fs{size}\\fad(150,150)}}" + esc(tr))
        if (i == 0 or para[i] != para[i - 1]) and l["start"] - a >= 2.9:
            for q in range(3):   # 倒数圆点
                add(l["start"] - 3 + 0.0, l["start"] - q, "Dot", W / 2 - 60 + q * 60, 150, "}●")

    def style(nm, size, c1, c2, outline):
        return f"Style: {nm},{name},{size},{c1},{c2},&H00301810,&H80000000,0,0,0,0,100,100,0,0,1,{outline},2,5,0,0,0,1"
    BLUE, WHITE, YEL = "&H00FFB43C", "&H00FFFFFF", "&H0078E6FF"
    path.write_text("\n".join([
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 2",
        "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,"
        "StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        style("Han", 112, BLUE, WHITE, 5), style("Py", 46, BLUE, WHITE, 3), style("Tr", 56, YEL, YEL, 3),
        style("Title", 150, WHITE, WHITE, 6), style("Dot", 60, BLUE, BLUE, 3), "",
        "[Events]", "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text", *ev, ""]), encoding="utf-8")


# ---------- 7. 渲染 ----------
def render(d, lines, title, artist, duration, background="封面模糊", guide=0.0, original=None, shift=0.0):
    """guide: 保留原唱的音量比例，0 = 纯伴奏
    original: 给出人声路径则生成原唱版（忽略 guide）；人声来自参考原唱时按 shift 平移后混入伴奏"""
    make_ass(lines, title, artist, d / "lyrics.ass")
    shutil.copyfile(font()[0], d / font()[0].name)
    vids = [p for p in d.glob("video.*") if p.suffix != ".part"]
    fill = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
    if background == "原视频" and vids:
        inp, vf = ["-stream_loop", -1, "-i", vids[0]], f"{fill},fps=30,eq=brightness=-0.28:saturation=0.8"
    elif background == "封面模糊" and (d / "cover.jpg").exists():
        ff("-i", d / "cover.jpg", "-vf", f"{fill},gblur=sigma=40,eq=brightness=-0.2:saturation=1.1", "-frames:v", 1, d / "bg.png")
        inp, vf = ["-loop", 1, "-framerate", 30, "-i", d / "bg.png"], "null"
    else:
        inp, vf = ["-f", "lavfi", "-i", f"gradients=s={W}x{H}:c0=0x141432:c1=0x3c1e50:speed=0.01:r=30"], "null"
    # 滤镜参数用相对路径，避免路径转义问题
    vf += ",ass=lyrics.ass:fontsdir=.,format=yuv420p"
    MIX = "[1:a][v]amix=inputs=2:normalize=0,alimiter=limit=0.97:level=0[a]"
    auds, af = ["instrumental.wav"], "[1:a]anull[a]"
    if original and Path(original).parent.resolve() == Path(d).resolve():
        auds = ["audio.wav"]   # 歌曲本身带原唱：直接用未分离的原声
    elif original:
        move = f"adelay={round(shift * 1000)}:all=1" if shift >= 0 else f"atrim=start={-shift:.4f},asetpts=PTS-STARTPTS"
        auds, af = ["instrumental.wav", Path(original).resolve()], f"[2:a]{move}[v];{MIX}"
    elif guide > 0:
        auds, af = ["instrumental.wav", "vocals.wav"], f"[2:a]volume={guide}[v];{MIX}"
    safe = re.sub(r'[\\/:*?"<>|\s]+', "_", f"{title}_{artist}".strip("_"))[:60] or "karaoke"
    out = OUT / f"{safe}_{'原唱' if original else ''}卡拉OK.mp4"
    for codec in (["h264_nvenc", "-preset", "p5", "-cq", "21", "-b:v", "0"], ["libx264", "-preset", "veryfast", "-crf", "20"]):
        p = subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *map(str, inp),
                            *(x for f in auds for x in ("-i", str(f))),
                            "-filter_complex", f"[0:v]{vf}[o];{af}", "-map", "[o]", "-map", "[a]", "-c:v", *codec,
                            "-c:a", "aac", "-b:a", "256k", "-t", str(duration), "-movflags", "+faststart", str(out)],
                           cwd=d, capture_output=True, text=True)
        if p.returncode == 0:
            return out
        print(codec[0], "失败：", p.stderr[-300:])
    raise RuntimeError("视频编码失败：" + p.stderr[-800:])
