# 卡拉OK视频生成器
    uv run app.py        # 打开 http://127.0.0.1:7860
    uv run cli.py <链接或文件> [--lyrics 歌词.txt] [--asr] [--bg 原视频|封面模糊|渐变色] [--guide 0.1] [--original]
- 成品在 `output/`，中间文件在 `work/`（可随时删除），模型在 `models/`。
- 翻译和歌名识别会调用本机 `claude` 命令（歌词/标题会发给 Claude）；歌词搜索使用 lrclib.net。
- 链接下载失败时先运行 `uv lock --upgrade-package yt-dlp && uv sync`。
- 对齐模型 MMS_FA 为 CC-BY-NC（非商用）许可。请只处理你有权使用的歌曲。
- 自备歌词：界面左侧粘贴或上传 .txt/.lrc，点“开始”即自动对轴（命令行 `--lyrics`）。
- “生成原唱版视频”（命令行 `--original`）：保留原唱 + 同样的逐字字幕，供跟唱练习；另存为 `_原唱卡拉OK.mp4`，不覆盖伴奏版。
- 链接是伴奏版（标题含“伴奏”等）时，会自动找同版原唱来对轴；也可手动指定参考原唱（命令行 `--ref`）。
