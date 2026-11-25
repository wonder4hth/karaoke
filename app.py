"""网页界面：uv run app.py"""
import pipeline as K
import gradio as gr

BGS = ["封面模糊", "原视频", "渐变色"]
COLS = ["开始(秒)", "歌词", "拼音", "译文", "置信度"]


def read_text(f):
    raw = K.Path(f).read_bytes()
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    raise gr.Error("歌词文件编码无法识别，请存成 UTF-8")


def step1(url, file, bg, own, ownfile, ref, p=gr.Progress()):
    if not (url or "").strip() and not file:
        raise gr.Error("请粘贴链接或上传文件")
    own = (own or "").strip() or (read_text(ownfile).strip() if ownfile else "")
    if own and not K.parse_lyrics(own):
        raise gr.Error("自备歌词里没有可用的歌词行")
    note = ""
    try:
        p(0.1, "下载中…")
        d, info = K.fetch(file or url, bg == "原视频")
        p(0.3, "分离人声…")
        inst, voc = K.separate(d)
        p(0.5, "识别歌曲…")
        track, artist = K.guess_song(info)
        shift = 0.0
        if (ref or "").strip() or K.is_instrumental(info):
            p(0.55, "用原唱版对轴…")
            voc, shift, name = (K.use_reference(d, ref) if (ref or "").strip()
                                else K.find_reference(d, info, track, artist))
            note = f"对轴参考原唱：**{name}**（平移 {shift:+.2f} 秒）。"
        p(0.8, "查找歌词…")
        hit = (own, "自备歌词") if own else K.search_lyrics(track, artist, info["duration"]) if track else None
        if hit:
            text, src = hit
        else:
            p(0.8, "歌词库没有，AI 听写中…")
            text, src = K.transcribe(voc, track or info["title"]), "Whisper 听写（请务必校对）"
    except Exception as e:
        raise gr.Error(str(e))
    st = dict(dir=str(d), voc=str(voc), shift=shift, auto=bool(own), duration=info["duration"])
    return st, track or info["title"], artist, text, note + f"歌词来源：**{src}**。" + ("正在自动对轴…完成后请看下方表格。" if own else "请核对歌词，再点第 2 步。"), str(inst), [], None


def relyrics(st, track, artist, mode):
    if not st:
        raise gr.Error("请先完成第 1 步")
    if mode == "asr":
        return K.transcribe(K.Path(st["voc"]), track), "歌词来源：**Whisper 听写（请务必校对）**"
    hit = K.search_lyrics(track, artist, st["duration"])
    if not hit:
        raise gr.Error("歌词库没有找到，可改歌名/歌手，或用 AI 听写")
    return hit[0], f"歌词来源：**{hit[1]}**"


def step2(st, track, text, p=gr.Progress()):
    if not st:
        raise gr.Error("请先完成第 1 步")
    ly = K.parse_lyrics(text)
    if not ly:
        raise gr.Error("歌词为空")
    try:
        p(0.2, "逐字对齐…")
        lines = K.align_lyrics(K.Path(st["voc"]), ly, st["shift"])
    except Exception as e:
        raise gr.Error(str(e))
    bad = [str(i) for i, l in enumerate(lines, 1) if K.suspicious(l)]
    if bad:
        gr.Warning(f"第 {'、'.join(bad)} 行对轴可疑，请重点检查。若整片可疑，通常是歌词不匹配，或该音频没有主唱（可填参考原唱链接）")
    p(0.6, "翻译…")
    try:
        K.translate(lines, track)
    except Exception as e:
        gr.Warning(f"翻译失败，可手动填写：{e}")
    try:
        p(0.85, "校对多音字…")
        K.fix_pinyin(lines, track)
    except Exception as e:
        gr.Warning(f"多音字校对失败，沿用默认拼音：{e}")
    rows = [[round(l["start"], 2), l["text"], " ".join(t["py"] for t in l["toks"] if t["py"]), l["tr"], round(l["conf"], 2)] for l in lines]
    return {**st, "lines": lines}, rows


def auto_step2(st, track, text, p=gr.Progress()):
    return step2(st, track, text, p) if st["auto"] else (st, [])


def step3(st, track, artist, rows, bg, guide, original=False, p=gr.Progress()):
    if not st or "lines" not in st:
        raise gr.Error("请先完成第 2 步")
    if len(rows) != len(st["lines"]):
        raise gr.Error("请不要增删表格行；要增删请改歌词后重跑第 2 步")
    lines = []
    for i, (l, r) in enumerate(zip(st["lines"], rows), 1):
        try:
            sh = float(r[0]) - round(l["start"], 2)
        except (TypeError, ValueError):
            raise gr.Error(f"第 {i} 行开始时间无效")
        py = str(r[2] or "").split()
        han = [t for t in l["toks"] if t["py"]]
        if len(py) != len(han):
            raise gr.Error(f"第 {i} 行拼音应有 {len(han)} 个音节，现在是 {len(py)} 个")
        it = iter(py)
        toks = [{**t, "s": t["s"] + sh, "e": t["e"] + sh, "py": next(it) if t["py"] else ""} for t in l["toks"]]
        lines.append({**l, "start": l["start"] + sh, "end": l["end"] + sh, "toks": toks, "tr": str(r[3] or "").strip()})
    p(0.2, "渲染原唱版视频…" if original else "渲染视频…")
    try:
        return str(K.render(K.Path(st["dir"]), lines, track, artist, st["duration"], bg, guide,
                            original=st["voc"] if original else None, shift=st["shift"]))
    except Exception as e:
        raise gr.Error(str(e))


def step3_original(st, track, artist, rows, bg, p=gr.Progress()):
    return step3(st, track, artist, rows, bg, 0, True, p)


with gr.Blocks(title="卡拉OK视频生成器") as demo:
    gr.Markdown("# 🎤 卡拉OK视频生成器\n粘贴歌曲链接 → 纯伴奏 + 中文 / 拼音 / 英文逐字字幕视频")
    st = gr.State()
    with gr.Row():
        with gr.Column():
            url = gr.Textbox(label="歌曲链接（YouTube、B站等）")
            file = gr.File(label="或上传音频/视频文件", type="filepath", height=110)
            bg = gr.Radio(BGS, value=BGS[0], label="背景（选“原视频”需在第 1 步之前）")
            own = gr.Textbox(label="自备歌词（可选，一行一句，空行会忽略）——填了就直接用它并自动对轴", lines=8, max_lines=14)
            ownfile = gr.File(label="或上传歌词文件 .txt / .lrc", type="filepath", file_types=[".txt", ".lrc"], height=90)
            ref = gr.Textbox(label="参考原唱链接（可选）——歌曲是伴奏版时用它对轴；留空会自动找")
            b1 = gr.Button("1️⃣ 开始（下载、去人声、歌词）", variant="primary")
            inst = gr.Audio(label="伴奏试听", interactive=False)
        with gr.Column():
            with gr.Row():
                track, artist = gr.Textbox(label="歌名"), gr.Textbox(label="歌手")
            msg = gr.Markdown()
            text = gr.Textbox(label="歌词（可直接修改或粘贴，一行一句；时间戳可有可无）", lines=14, max_lines=14)
            with gr.Row():
                bs, ba = gr.Button("重新搜索歌词"), gr.Button("AI 听写歌词")
            b2 = gr.Button("2️⃣ 逐字对齐 + 拼音 + 翻译", variant="primary")
    table = gr.Dataframe(headers=COLS, datatype=["number", "str", "str", "str", "number"], type="array", interactive=True, wrap=True,
                         label="可修改：开始时间（整行平移）、拼音、译文。改歌词请在上面改后重跑第 2 步；置信度低的行请重点检查")
    guide = gr.Slider(0, 1, 0, step=0.05, label="人声音量（0 = 完全无人声；伴奏版调到 1 可保留原有和声）")
    with gr.Row():
        b3 = gr.Button("3️⃣ 生成伴奏版视频", variant="primary")
        b4 = gr.Button("🎧 生成原唱版视频（跟唱练习）", variant="primary")
    video = gr.Video(label="成品（保存在 output/）")
    b1.click(step1, [url, file, bg, own, ownfile, ref], [st, track, artist, text, msg, inst, table, video]).success(
        auto_step2, [st, track, text], [st, table])
    bs.click(lambda s, t, a: relyrics(s, t, a, "db"), [st, track, artist], [text, msg])
    ba.click(lambda s, t, a: relyrics(s, t, a, "asr"), [st, track, artist], [text, msg])
    b2.click(step2, [st, track, text], [st, table])
    # 两个按钮共用字幕等中间文件，必须排队
    b3.click(step3, [st, track, artist, table, bg, guide], video, concurrency_id="render", concurrency_limit=1)
    b4.click(step3_original, [st, track, artist, table, bg], video, concurrency_id="render", concurrency_limit=1)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", show_error=True,
                                                   allowed_paths=[str(K.WORK), str(K.OUT)])
