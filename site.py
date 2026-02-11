#!/usr/bin/env python3
"""Babel 播客静态网站生成器

Commands: init / add / build / serve
"""

import argparse

from site_tools.config import init_site
from site_tools.episodes import add_episode
from site_tools.build import build_site
from site_tools.serve import serve_site


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Babel 播客静态网站生成器",
    )
    parser.add_argument(
        "--site-dir", default="site",
        help="站点目录（默认 site/）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = subparsers.add_parser("init", help="初始化站点目录")
    p_init.add_argument("--title", default=None, help="站点标题")
    p_init.add_argument("--base-url", default=None, help="站点基础 URL")
    p_init.add_argument("--description", default=None, help="站点描述")
    p_init.add_argument("--author", default=None, help="作者名称")

    # add
    p_add = subparsers.add_parser("add", help="添加一集")
    p_add.add_argument("--title", required=True, help="剧集标题")
    p_add.add_argument("--zh-audio", required=True, help="中文音频 MP3 路径")
    p_add.add_argument("--en-audio", default=None, help="英文原版音频 MP3 路径")
    p_add.add_argument("--summary", default=None, help="简短摘要 .txt 文件路径")
    p_add.add_argument("--detailed-summary", default=None, help="详细摘要 .md 文件路径")
    p_add.add_argument("--slug", default=None, help="自定义 URL slug（默认从标题生成）")
    p_add.add_argument("--pub-date", default=None, help="发布日期 YYYY-MM-DD（默认今天）")

    # build
    subparsers.add_parser("build", help="生成静态站点")

    # serve
    p_serve = subparsers.add_parser("serve", help="本地预览服务器")
    p_serve.add_argument("--port", type=int, default=8000, help="端口号（默认 8000）")

    args = parser.parse_args()

    if args.command == "init":
        init_site(args)
    elif args.command == "add":
        add_episode(args)
    elif args.command == "build":
        build_site(args)
    elif args.command == "serve":
        serve_site(args)


if __name__ == "__main__":
    main()
