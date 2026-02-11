"""Local HTTP server for previewing the built site."""

import functools
import http.server
from pathlib import Path


def serve_site(args):
    """Start a local HTTP server serving site/build/."""
    build_dir = Path(args.site_dir) / "build"
    if not build_dir.exists():
        print(f"错误: 构建目录不存在: {build_dir}")
        print("请先运行 build 命令。")
        return

    port = args.port
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(build_dir),
    )
    with http.server.HTTPServer(("", port), handler) as httpd:
        print(f"预览服务器已启动: http://localhost:{port}")
        print("按 Ctrl+C 停止。")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止。")
