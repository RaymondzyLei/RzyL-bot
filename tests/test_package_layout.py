"""打包与依赖方向的验收测试（第 0 步的产出）。

两件事：两个顶层模块都能 import；以及 core 的"不依赖 NoneBot"这条约束在源码层面
真的成立——后者是 issue #1 里写死的硬约束，靠人自觉守不住，所以让测试来守。
"""

import ast
import importlib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"


def test_both_top_level_modules_import() -> None:
    assert importlib.import_module("rzyl_bot") is not None
    assert importlib.import_module("rzyl_core") is not None


def test_core_does_not_import_nonebot() -> None:
    """core 里出现 nonebot 就是设计错误：它必须能脱离机器人单测与离线运行。"""
    offenders: list[str] = []

    for source in sorted((SRC / "rzyl_core").rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "nonebot" or name.startswith("nonebot.") for name in names):
                offenders.append(f"{source}:{node.lineno}")

    assert not offenders, f"rzyl_core 不得依赖 NoneBot，但发现：{offenders}"


@pytest.mark.parametrize("module", ["rzyl_bot", "rzyl_core"])
def test_module_lives_under_src(module: str) -> None:
    assert (SRC / module / "__init__.py").is_file()
