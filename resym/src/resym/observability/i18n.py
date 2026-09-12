"""Small, task-neutral Chinese localization table for the run viewer."""

from __future__ import annotations

import html


ZH_PHRASES = {
    "reSym run viewer": "reSym 运行查看器",
    "Live execution": "实时执行",
    "System library": "系统符号库",
    "Capability catalog": "能力目录",
    "Grounding factories": "接地工厂",
    "Latest run": "最近运行",
    "Stage A · world": "阶段 A · 世界",
    "Stage B · repair process": "阶段 B · 修复过程",
    "Stage C · task solve": "阶段 C · 任务求解",
    "Stage D · visualization": "阶段 D · 可视化",
    "execution trace — the event stream the live page renders": (
        "执行轨迹 — 实时页面展示的事件流"
    ),
    "Grounded plan": "接地后的计划",
    "Current action": "当前动作",
    "Current transformation": "当前转换",
    "Recent events": "最近事件",
    "Library version stores": "符号库版本记录",
    "How the pieces connect": "各部分如何连接",
    "design time — stored in the symbol library": "设计期 — 存储在符号库中",
    "run time — created per dispatched action": "运行期 — 每次动作派发时创建",
}


ZH_TERMS = {
    "runs": "运行记录",
    "world": "世界",
    "repair process": "修复过程",
    "task solve": "任务求解",
    "visualization": "可视化",
    "goal": "目标",
    "results": "结果",
    "predicates": "谓词",
    "operators": "算子",
    "capabilities": "能力",
    "checks": "检查",
}


def _variants(text: str) -> tuple[str, ...]:
    """Raw and escaped forms a phrase may have in rendered HTML."""
    return tuple(
        dict.fromkeys(
            (text, html.escape(text, quote=False), html.escape(text, quote=True))
        )
    )


def to_chinese(page: str) -> str:
    """Translate known viewer chrome while leaving identifiers and data untouched."""
    for english, chinese in sorted(
        ZH_PHRASES.items(), key=lambda item: -len(item[0])
    ):
        for variant in _variants(english):
            page = page.replace(variant, chinese)
    for english, chinese in ZH_TERMS.items():
        for variant in _variants(english):
            page = page.replace(f">{variant}<", f">{chinese}<")
    return page
