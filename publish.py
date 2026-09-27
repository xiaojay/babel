#!/usr/bin/env python3
"""Babel Publish - 自动上传播客到网站

用法: python publish.py <zh_audio_path> [--title TITLE] [--slug SLUG]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# 配置
R2_BUCKET = "babel-podcast"
CDN_BASE = "https://cdn.jaylab.io"
PAGES_PROJECT = "babel-podcast"


def slugify(title: str) -> str:
    """Generate URL-safe slug from title."""
    # 移除非ASCII字符，转小写，替换空格和特殊字符
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    if not slug or len(slug) < 3:
        # 如果 slug 太短，用文件名
        return None
    return slug[:80]  # 限制长度


def extract_title_from_path(zh_audio_path: Path) -> str:
    """从文件路径提取标题."""
    name = zh_audio_path.stem  # 不含扩展名
    if name.endswith("_zh"):
        name = name[:-3]
    return name


def generate_chinese_title(original_title: str) -> str:
    """尝试生成中文标题（简单映射）."""
    # 这里可以后续接入 LLM 翻译，目前直接返回原标题
    return original_title


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """执行命令并打印."""
    print(f"  $ { ' '.join(cmd[:3])}...")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def upload_to_r2(local_path: Path, r2_key: str) -> bool:
    """上传文件到 R2."""
    cmd = [
        "wrangler", "r2", "object", "put",
        f"{R2_BUCKET}/{r2_key}",
        f"--file={local_path}",
        "--remote",
    ]
    result = run_cmd(cmd, check=False)
    if result.returncode != 0:
        print(f"  ❌ R2 上传失败: {result.stderr}")
        return False
    print(f"  ✅ R2: {r2_key}")
    return True


def add_episode(site_dir: Path, title: str, slug: str, zh_audio: Path, en_audio: Path = None):
    """调用 site.py add 添加剧集."""
    cmd = [
        sys.executable, "site.py", "add",
        "--title", title,
        "--slug", slug,
        "--zh-audio", str(zh_audio),
    ]
    if en_audio and en_audio.exists():
        cmd.extend(["--en-audio", str(en_audio)])

    # 查找 summary 文件（babel.py 写在输出 MP3 同目录）
    summary_file = zh_audio.with_suffix(".summary.txt")
    detailed_file = zh_audio.with_suffix(".summary.detailed.md")

    if summary_file.exists():
        cmd.extend(["--summary", str(summary_file)])
    if detailed_file.exists():
        cmd.extend(["--detailed-summary", str(detailed_file)])
    
    result = run_cmd(cmd, check=False)
    if result.returncode != 0:
        print(f"  ❌ 添加剧集失败: {result.stderr}")
        return False
    print(f"  ✅ 已添加: {title}")
    return True


def build_and_deploy(site_dir: Path) -> bool:
    """构建并部署网站."""
    # Build
    result = run_cmd([sys.executable, "site.py", "build"], check=False)
    if result.returncode != 0:
        print(f"  ❌ 构建失败: {result.stderr}")
        return False
    print("  ✅ 构建完成")
    
    # Remove audio symlink
    audio_link = site_dir / "build" / "audio"
    if audio_link.exists() or audio_link.is_symlink():
        audio_link.unlink()
    
    # Deploy
    result = run_cmd([
        "wrangler", "pages", "deploy",
        str(site_dir / "build"),
        f"--project-name={PAGES_PROJECT}",
        "--commit-dirty=true",
    ], check=False)
    if result.returncode != 0:
        print(f"  ❌ 部署失败: {result.stderr}")
        return False
    print("  ✅ 部署完成")
    return True


def main():
    parser = argparse.ArgumentParser(description="自动发布播客到网站")
    parser.add_argument("zh_audio", help="中文音频文件路径")
    parser.add_argument("--title", help="剧集标题（默认从文件名提取）")
    parser.add_argument("--slug", help="URL slug（默认从标题生成）")
    parser.add_argument("--en-audio", help="英文原版音频路径")
    parser.add_argument("--skip-upload", action="store_true", help="跳过 R2 上传")
    parser.add_argument("--skip-deploy", action="store_true", help="跳过部署")
    
    args = parser.parse_args()
    
    zh_audio = Path(args.zh_audio).resolve()
    if not zh_audio.exists():
        print(f"❌ 文件不存在: {zh_audio}")
        sys.exit(1)
    
    # 确定英文音频路径
    en_audio = None
    if args.en_audio:
        en_audio = Path(args.en_audio).resolve()
    else:
        # 尝试自动查找
        possible_en = zh_audio.parent / zh_audio.name.replace("_zh.mp3", ".mp3")
        if possible_en.exists() and possible_en != zh_audio:
            en_audio = possible_en
    
    # 确定标题和 slug
    original_title = args.title or extract_title_from_path(zh_audio)
    title = generate_chinese_title(original_title)
    slug = args.slug or slugify(original_title)
    
    if not slug:
        print("❌ 无法生成 slug，请用 --slug 指定")
        sys.exit(1)
    
    print(f"\n📻 发布播客")
    print(f"   标题: {title}")
    print(f"   Slug: {slug}")
    print(f"   中文: {zh_audio.name}")
    if en_audio:
        print(f"   英文: {en_audio.name}")
    print()
    
    # 切换到 babel 目录
    babel_dir = Path(__file__).parent.resolve()
    os.chdir(babel_dir)
    site_dir = babel_dir / "site"
    
    # Step 1: 上传到 R2
    if not args.skip_upload:
        print("[1/3] 上传到 R2...")
        if not upload_to_r2(zh_audio, f"audio/{slug}/zh.mp3"):
            sys.exit(1)
        if en_audio:
            if not upload_to_r2(en_audio, f"audio/{slug}/en.mp3"):
                sys.exit(1)
    else:
        print("[1/3] 跳过 R2 上传")
    
    # Step 2: 添加剧集
    print("[2/3] 添加剧集...")
    if not add_episode(site_dir, title, slug, zh_audio, en_audio):
        sys.exit(1)
    
    # Step 3: 构建并部署
    if not args.skip_deploy:
        print("[3/3] 构建并部署...")
        if not build_and_deploy(site_dir):
            sys.exit(1)
    else:
        print("[3/3] 跳过部署")
    
    print(f"\n🎉 发布完成!")
    print(f"   🌐 https://podcast.jaylab.io/episodes/{slug}")


if __name__ == "__main__":
    main()
