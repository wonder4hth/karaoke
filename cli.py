"""命令行：uv run cli.py <链接或文件> [--lyrics 歌词.txt] [--bg 原视频] [--guide 0.1] [--original]"""
import argparse, json, time
import pipeline as K

a = argparse.ArgumentParser()
a.add_argument("source"); a.add_argument("--lyrics"); a.add_argument("--track"); a.add_argument("--artist", default="")
a.add_argument("--ref"); a.add_argument("--bg", default="封面模糊"); a.add_argument("--guide", type=float, default=0); a.add_argument("--asr", action="store_true")
a.add_argument("--original", action="store_true", help="生成原唱版")
a = a.parse_args()
t0 = time.time()
def log(*x): print(f"[{time.time() - t0:6.1f}s]", *x, flush=True)

d, info = K.fetch(a.source, a.bg == "原视频"); log("已下载", info["title"])
_, voc = K.separate(d); log("已分离人声")
track, artist = (a.track, a.artist) if a.track else K.guess_song(info); log("歌曲：", track, "/", artist)
shift, hit = 0.0, None
if a.ref or K.is_instrumental(info):
    voc, shift, name = K.use_reference(d, a.ref) if a.ref else K.find_reference(d, info, track, artist)
    log("对轴参考原唱：", name, f"平移 {shift:+.3f}s")
if a.lyrics:
    hit = (open(a.lyrics, encoding="utf-8").read(), a.lyrics)
elif not a.asr:
    hit = K.search_lyrics(track, artist, info["duration"])
text, src = hit or (K.transcribe(voc, track), "Whisper"); log("歌词来源：", src)
lines = K.align_lyrics(voc, K.parse_lyrics(text), shift); log("已对齐")
K.translate(lines, track); K.fix_pinyin(lines, track)
for l in lines: log(f"{l['start']:6.2f}-{l['end']:6.2f} 置信{l['conf']:.2f}{' ⚠' if K.suspicious(l) else ''} {l['text']} | {l['tr']}")
(d / "lines.json").write_text(json.dumps(lines, ensure_ascii=False))
log("完成：", K.render(d, lines, track, artist, info["duration"], a.bg, a.guide,
                     original=voc if a.original else None, shift=shift))
