"""Tests for the tools package exports."""

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess:
    """Run code in a fresh interpreter so import state is not shared."""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    )


def test_importing_one_step_does_not_load_the_others():
    result = _run(
        "import sys\n"
        "import tools.youtube_download\n"
        "loaded = [m for m in ('torch', 'whisperx', 'tools.transcribe',"
        " 'tools.synthesize') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )

    assert result.returncode == 0, result.stderr


def test_package_exports_functions_not_submodules():
    # tools.transcribe is both a submodule and an exported function name.
    result = _run(
        "import sys, types\n"
        "sys.modules['whisperx'] = types.ModuleType('whisperx')\n"
        "from tools import transcribe, is_youtube_url\n"
        "import tools\n"
        "assert transcribe.__module__ == 'tools.transcribe', transcribe\n"
        "assert tools.transcribe is transcribe\n"
        "assert is_youtube_url('https://youtu.be/abc')\n"
    )

    assert result.returncode == 0, result.stderr


def test_unknown_attribute_raises():
    result = _run(
        "import tools\n"
        "try:\n"
        "    tools.does_not_exist\n"
        "except AttributeError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('expected AttributeError')\n"
    )

    assert result.returncode == 0, result.stderr
