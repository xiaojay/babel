# 这个 patch 需要手动应用到 babel.py

# 1. 在 argparse 部分添加:
parser.add_argument(
    "--auto-publish",
    action="store_true",
    help="转录完成后自动发布到网站",
)
parser.add_argument(
    "--publish-title",
    default=None,
    help="发布时使用的标题（默认从文件名提取）",
)
parser.add_argument(
    "--publish-slug",
    default=None,
    help="发布时使用的 URL slug（默认从标题生成）",
)

# 2. 在 "完成！" 之后、finally 之前添加:
if args.auto_publish:
    print()
    print("[Step 6] 自动发布到网站...")
    publish_cmd = [
        sys.executable, 
        str(Path(__file__).parent / "publish.py"),
        str(output_path),
    ]
    if args.publish_title:
        publish_cmd.extend(["--title", args.publish_title])
    if args.publish_slug:
        publish_cmd.extend(["--slug", args.publish_slug])
    if en_audio_path:
        publish_cmd.extend(["--en-audio", str(en_audio_path)])
    
    import subprocess
    result = subprocess.run(publish_cmd)
    if result.returncode != 0:
        print("⚠️ 自动发布失败，请手动运行 publish.py")
