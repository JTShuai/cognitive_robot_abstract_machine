"""
Chinese localization for the run viewer.

The viewer renders English HTML; :func:`to_chinese` translates a finished
page by phrase substitution.  Three tiers keep it safe and maintainable:

- ``ZH_PATTERNS``: regexes for sentences with interpolated numbers.
- ``ZH_PHRASES``: long, distinctive strings replaced verbatim (each key is
  matched raw and in both html-escaped forms, so body text and data-tip
  attributes both translate).
- ``ZH_TERMS``: short labels replaced only as full element text (``>x<``),
  so they can never corrupt attributes, URLs, or displayed data.

Anything not in the tables simply stays English — graceful degradation,
never a broken page.  Technical identifiers (backend names, fault template
ids, file names, predicate names) are data and are deliberately not
translated.
"""

from __future__ import annotations

import html
import re

ZH_PATTERNS: list[tuple[str, str]] = [
    (r"repair case (\d+)", r"修复案例 \1"),
    (
        r"browse the (\d+) repair cases \((\d+) conversations\)",
        r"浏览 \1 个修复案例（\2 个对话）",
    ),
    (r"browse the (\d+) conversations", r"浏览 \1 个对话"),
    (
        r"The (\d+) calls group into (\d+) conversations, and the "
        r"conversations into (\d+) repair cases — ",
        r"这 \1 次调用归为 \2 个对话，对话再归为 \3 个修复案例 — ",
    ),
    (
        r"(\d+) independent attempts on this one failure",
        r"\1 次相互独立的修复尝试，都针对这同一个故障",
    ),
    (
        r" — the agentic-rag loop first, then (\d+) one-shot proposals? "
        r"\(closed-book, rag-one-shot, fixed-pipeline\)",
        r" — agentic-rag 循环先上，随后是 \1 个一次性提案"
        r"（closed-book、rag-one-shot、fixed-pipeline）",
    ),
    (
        r" · showing (\d+) of them \(role filter: ([^)<]+)\)",
        r" · 因角色筛选只显示其中 \1 个（\2）",
    ),
    (r"showing (\d+) of (\d+) cases", r"命中 \1 个，共 \2 个案例"),
    (r"showing (\d+) of (\d+) conversations", r"命中 \1 个，共 \2 个对话"),
    (r"all faults \((\d+) cases\)", r"全部故障（\1 个案例）"),
    (r">all \((\d+)\)<", r">全部（\1）<"),
    (
        r"(\d+) episodes = (\d+) cases &times; (\d+) backends\. ",
        r"\1 条 episode = \2 个案例 × \3 个 backend。",
    ),
    (
        r" ?In (\d+) cases the injected fault never made planning fail, so "
        r"there was nothing to repair; the remaining (\d+) cases are "
        r"exactly the ones whose LLM conversations Stage B records\.",
        r"其中 \1 个案例注入的故障并未导致规划失败，因此无需修复；"
        r"剩下的 \2 个案例正是阶段 B 记录了 LLM 对话的那些案例。",
    ),
    (
        r"browse the (\d+) cases \((\d+) episodes\)",
        r"浏览 \1 个案例（\2 条 episode）",
    ),
    (r">correct repair \((\d+)\)<", r">正确修复（\1）<"),
    (r">false admission \((\d+)\)<", r">误准入（\1）<"),
    (r">fault never tripped \((\d+)\)<", r">故障未触发（\1）<"),
    (
        r">proposed, rejected at review \((\d+)\)<",
        r">已提案，审核未通过（\1）<",
    ),
    (r">no fix found \((\d+)\)<", r">未找到修复（\1）<"),
    (r">budget exhausted \((\d+)\)<", r">预算耗尽（\1）<"),
    (
        r">rightly declared unsupported \((\d+)\)<",
        r">正确判定为不支持（\1）<",
    ),
    (
        r">wrongly declared unsupported \((\d+)\)<",
        r">误判为不支持（\1）<",
    ),
    (r">all conversations \((\d+)\)<", r">全部对话（\1）<"),
    (r"(\d+) turns, (\d+) calls", r"\1 轮 · \2 次调用"),
    (r"(\d+) turns", r"\1 轮"),
    (r"1 call, (\d+) retr(?:ies|y)", r"1 次调用 + \1 次重试"),
    (r"· 1 call ", r"· 1 次调用 "),
    (r">turn (\d+) · ", r">第 \1 轮 · "),
    (r"· (\d+) attempts", r"· \1 次尝试"),
    (r"· (\d+) parse errors?", r"· \1 次解析失败"),
    (r"prompt \((\d+) chars\)", r"提示词（\1 字符）"),
    (r"response \((\d+) chars\)", r"回复（\1 字符）"),
    (r"stopped after ([\w_]+)", r"止步于 \1"),
    (r"held-out failures \((\d+)\)", r"held-out 复测失败（\1 个）"),
    (r" by fault template</", r" 按故障模板分</"),
    (r"full library at (v\d+)", r"\1 的完整库"),
    (r"reviewing plan \((\d+)s\)", r"计划预览中（\1 秒）"),
    (
        r"This task completed at ([^\s<]+)",
        r"该任务已于 \1 完成",
    ),
    (
        r"(\d+) more version stores not rendered \(limit (\d+)\)",
        r"另有 \1 个版本库未渲染（上限 \2）",
    ),
    (r"· ((?:\d+|&lt;1))% of its calls", r"· 占其调用的 \1%"),
    (
        r"· (\d+) sections, (\d+) changed since the previous turn",
        r"· \1 个小节，较上一轮 \2 个有变化",
    ),
    (r"· (\d+) sections", r"· \1 个小节"),
    (r"· (\d+) chars", r"· \1 字符"),
    (
        r"earlier attempts \((\d+)\) — replies failed to parse and were " r"retried",
        r"之前的 \1 次尝试 — 回复解析失败后被重试",
    ),
    (r"final attempt \((\d+)\)", r"最终尝试（第 \1 次）"),
    (r">attempt (\d+)<", r">第 \1 次尝试<"),
]

ZH_PHRASES: dict[str, str] = {
    # ---- header / navigation --------------------------------------------
    "&#9635; runs": "&#9635; 运行记录",
    # ---- index -----------------------------------------------------------
    "reSym run viewer": "reSym 运行查看器",
    "reSym repairs a robot's symbolic planning model when a missing or "
    "wrong symbol makes a task fail. This viewer shows everything the "
    "pipeline records: the symbol libraries it plans with, each run's "
    "plan and execution trace, and the repair experiments with their "
    "admission decisions.": (
        "当符号缺失或错误导致机器人任务失败时，reSym 会修复其符号规划模型。"
        "本查看器展示流水线记录的一切：规划所用的符号库、每次运行的计划与"
        "执行轨迹，以及修复实验及其准入决定。"
    ),
    "⚡ Live execution": "⚡ 实时执行",
    "Follow the most recent task as it runs: grounding counts, the plan "
    "step by step, and every Boolean check.": (
        "跟随最近一次任务的实时执行：接地统计、逐步执行的计划，以及每一次" "布尔检查。"
    ),
    "🧩 System library": "🧩 系统符号库",
    "The shipped symbol libraries and capability contracts, with an "
    "interactive graph of how predicates, operators and capabilities "
    "connect.": (
        "随系统发布的符号库与能力契约，附交互图展示谓词、算子与能力如何" "关联。"
    ),
    "🕐 Latest run": "🕐 最新运行",
    "— its goal, artifacts, and (for experiments) the admission "
    "results.": "— 它的目标、产物，以及（实验运行的）准入结果。",
    "Every recorded run, newest first — demos and experiments alike. "
    "Open one for its artifacts, execution trace, and the libraries it "
    "used.": (
        "全部运行记录，最新在前 — 演示与实验都在。点开可查看其产物、执行"
        "轨迹和所用的符号库。"
    ),
    ">started ": ">开始于 ",
    "see run.json": "详见 run.json",
    # ---- run page chrome -------------------------------------------------
    "Stage A · world": "阶段 A · 世界",
    "Stage B · repair process": "阶段 B · 修复过程",
    "Stage C · task solve": "阶段 C · 任务求解",
    "Stage D · experiment results": "阶段 D · 实验结果",
    "Stage D · visualization": "阶段 D · 可视化",
    "Load the scene, extract the typed planning objects.": (
        "加载场景，抽取带类型的规划对象。"
    ),
    "An LLM proposes a library patch — only when a symbol gap made "
    "planning fail.": "LLM 提出符号库补丁 — 仅当符号缺口导致规划失败时。",
    "Ground predicates → plan → execute → verify, once per task.": (
        "谓词接地 → 规划 → 执行 → 验证，每个任务一轮。"
    ),
    "Episode records and admission decisions.": "episode 记录与准入决定。",
    "RViz recording of the execution.": "执行过程的 RViz 录像。",
    "not present: no model gap in this run, so no repair happened": (
        "未出现：本次运行没有模型缺口，因此没有发生修复"
    ),
    "The reSym pipeline for this run — grayed stages did not occur. "
    "Click a stage to jump to its artifacts.": (
        "本次运行的 reSym 流水线 — 灰色阶段未发生。点击阶段可跳到它的产物。"
    ),
    "The typed objects this run planned over, extracted from the "
    "annotated world model.": (
        "本次运行规划所用的带类型对象，抽取自带标注的世界模型。"
    ),
    "The repair process, recorded raw: the LLM call log keeps every "
    "prompt, reply, and retry. This is the how — what became of each "
    "proposal (admitted or not, correct or not) is settled in Stage D.": (
        "修复过程的原始记录：LLM 调用日志保存了每一次提问、回复与重试。"
        "这里回答的是“怎么修的”——每个提案的结局（是否被接纳、是否修对）"
        "在阶段 D 中结算。"
    ),
    "One folder per task (named scene_task, e.g. apartment_open): the "
    "goal, the generated PDDL, the plan, and the execution trace.": (
        "每个任务一个文件夹（命名为 场景_任务，如 apartment_open）：目标、"
        "生成的 PDDL、计划与执行轨迹。"
    ),
    "The results ledger: one settled verdict per episode — which "
    "backend, which fault case, whether its repair was admitted and "
    "correct, and what it cost. The summary reports above are computed "
    "from these rows; the conversations behind them live in Stage B.": (
        "结果账本：每个 episode 一条已结算的裁决——哪个 backend、哪个故障"
        "案例、修复是否被接纳且正确、花了多少开销。上方的汇总报告都由这些"
        "记录算出；它们背后的对话在阶段 B 中。"
    ),
    "RViz recordings of the execution.": "执行过程的 RViz 录像。",
    "This is an experiment run": "这是一次实验运行",
    "E1 breaks the symbol library in known ways and asks: how often does "
    "each repair method produce a correct fix that passes review? E2 "
    "keeps the fixes themselves fixed (one right, the others wrong on "
    "purpose) and asks: how reliably does each review plan accept the "
    "right fix and reject the wrong ones? The reports and the raw "
    "decision records are below.": (
        "E1 以已知方式弄坏符号库，然后问：每种修复方法多大比例能产出通过"
        "审核的正确修复？E2 把候选修复本身固定住（一个正确、其余故意做"
        "错），然后问：每种审核方案能多可靠地接受正确修复、拒绝错误修复？"
        "报告与原始决定记录见下方。"
    ),
    "Top-level artifacts of this run: reports, library snapshots, "
    "recordings.": "本次运行的顶层产物：报告、库快照、录像。",
    "heavy per-episode working trees (probe scenes, version stores, "
    "suite scratch) are kept on disk but not rendered:": (
        "逐 episode 的重型工作目录（探针场景、版本库、测试暂存）保留在磁盘"
        "上但不渲染："
    ),
    # ---- gate verdict ----------------------------------------------------
    "Gate P2 decision — the pre-registered claim check": (
        "Gate P2 判定 — 预注册论断的检验"
    ),
    "the run's verdict on the pre-registered claim": ("本次运行对预注册论断的判定"),
    "show the raw gate text": "查看 gate 原文",
    # ---- results / reports -----------------------------------------------
    "E1 outcome rates per repair backend": "各修复 backend 的 E1 结果率",
    "Pre-registered E1 outcome rates per repair backend. Bars show the "
    "rate; the dark ticks span the 95% Wilson interval — overlapping "
    "intervals mean the difference is not settled by this data.": (
        "预注册的 E1 各 backend 结果率。条形为比率；深色刻度为 95% Wilson "
        "置信区间 — 区间重叠说明该数据尚不能定论差异。"
    ),
    "pre-registered E1 report": "预注册 E1 报告",
    "pre-registered E2 report": "预注册 E2 报告",
    "The pre-registered plain-text record — this exact text is what the "
    "paper cites. The tables above show the same numbers with context, "
    "so you rarely need to read it raw.": (
        "预注册的纯文本记录 — 论文引用的就是这段原文。上方表格以更有上下文"
        "的方式展示同样的数字，通常无需读原文。"
    ),
    "show the text report": "查看文本报告",
    "false admission (lower is better)": "误准入（越低越好）",
    # ---- transcript ------------------------------------------------------
    "the LLM call log — every prompt and reply, kept as the audit trail": (
        "LLM 调用日志 — 每条提示词与回复，留作审计凭据"
    ),
    "The call log of every LLM use in this run — what was asked, what "
    "came back, what it cost — kept so every repair can be audited "
    "later. Each row is one LLM role; their calls sum to the total, and "
    "a failed parse is one of those calls whose reply was unusable and "
    "got retried. Too long to show here.": (
        "本次运行每次 LLM 调用的记录 — 问了什么、答了什么、花了多少 — "
        "留存以便日后审计每个修复。表中每行是一个 LLM 角色；各行调用数加"
        "起来即总数，解析失败是其中回复不可用而被重试的调用，不是额外调"
        "用。内容太长，此处不展开。"
    ),
    "replies that failed to parse": "解析失败的回复",
    "one exchange = one prompt sent to the model plus its reply; the "
    "roles' calls sum to this total": (
        "一次 exchange = 发给模型的一条提示词加上它的回复；各角色调用数之和" "即此总数"
    ),
    "of that row's calls, how many replies were not in the required "
    "machine-readable format — a subset of the calls, each retried, not "
    "extra calls": (
        "该行调用中有多少回复不符合规定的机器可读格式 — 是调用的子集，每个"
        "都被重试，不是额外调用"
    ),
    "total text sent to the model (in) and generated by it (out) — the "
    "run's LLM cost": (
        "发给模型（in）与模型生成（out）的文字总量 — 本次运行的 LLM 开销"
    ),
    "one conversation = the calls one role made for one repair case: the "
    "looping agent's turns until it submits or gives up, or a one-shot "
    "call plus its format retries": (
        "一个对话 = 一个角色为一个修复案例发起的全部调用：循环 agent 的多轮"
        "（直到提交或放弃），或一次性调用加上其格式重试"
    ),
    "one case = one broken-library failure; inside it every repair "
    "method under test attacks that same failure independently": (
        "一个案例 = 一次符号库故障；案例内每种受测修复方法各自独立地攻克这" "同一个故障"
    ),
    "the step-by-step repair agent (used by the agentic-rag method): "
    "each call is one turn of its loop — look at the failure, pick a "
    "tool, refine the fix": (
        "多步修复 agent（agentic-rag 方法使用）：每次调用是其循环中的一轮 — "
        "查看故障、选择工具、打磨修复"
    ),
    "the one-shot proposer (used by closed-book, rag-one-shot and "
    "fixed-pipeline): one prompt in, one proposed fix out": (
        "一次性提案者（closed-book、rag-one-shot、fixed-pipeline 使用）："
        "一条提示词进，一个修复提案出"
    ),
    "an LLM role in the repair pipeline": "修复流水线中的一个 LLM 角色",
    "this conversation IS the agentic-rag method: the multi-step "
    "tool-loop repair backend": (
        "这个对话就是 agentic-rag 方法本身：多步工具循环修复 backend"
    ),
    "one of the three one-shot methods made this proposal — the call log "
    "does not record which one": (
        "此提案出自三个一次性方法之一 — 调用日志未记录具体是哪个"
    ),
    "recorded by the experiment harness at call time": ("由实验框架在调用时记录"),
    "the boxes below, one per planted failure, in the order they ran. A "
    "case is one fault template (what was broken) tried in one "
    "randomized scene (where) — the same fault recurs across several "
    "scenes so a method's success rate means something. Inside a box "
    "the LLM-using repair methods each attack that same failure "
    "separately: they compete, they never collaborate, and no state "
    "carries over between cases. The backends that never call the LLM "
    "(typed-enumeration, no-repair, oracle-reference) leave no "
    "conversations here. Open a turn for the full prompt and reply.": (
        "即下方方框，每框一个植入的故障，按运行顺序排列。一个案例 = 一种"
        "故障模板（坏了什么）在一个随机场景（在哪里）中的测试 — 同一故障"
        "在多个场景重复出现，方法的成功率才有意义。框内各个调用 LLM 的修复"
        "方法各自独立攻克同一故障：它们是竞争关系，从不协作，案例之间也不"
        "延续任何状态。不调用 LLM 的 backend（typed-enumeration、"
        "no-repair、oracle-reference）不会在此留下对话。展开某一轮可见完整"
        "提示词与回复。"
    ),
    "; none of them sees the others' work": "；它们彼此看不到对方的工作",
    " — each labelled with its backend": " — 每个都标注了各自的 backend",
    "submitted a fix for review": "已提交修复待审",
    "declared the gap unsupported": "已声明缺口不受支持",
    "proposed a fix": "提出了修复",
    "no usable reply": "没有可用回复",
    "ended on parse failures": "以解析失败告终",
    "unparsed reply": "回复无法解析",
    "the call · ": "该次调用 · ",
    "search the knowledge corpus for relevant symbols": ("在知识库中检索相关符号"),
    "put forward a candidate fix for the library": "提出一个候选库修复",
    "ask the reviewer's static check about the current candidate": (
        "就当前候选请求审核者的静态检查"
    ),
    "compare the current candidate with the previous one": ("比较当前候选与上一个候选"),
    "try the current fix on one test scene": "在一个测试场景上试跑当前修复",
    "look at the current predicates and operators": "查看当前谓词与算子",
    "look at what the robot platform can do": "查看机器人平台能做什么",
    "read the full record of one retrieved fragment": ("阅读某条检索片段的完整记录"),
    "give up: the platform lacks a needed capability": ("放弃：平台缺少所需能力"),
    "finish and hand the candidate to the reviewer": ("结束并把候选交给审核者"),
    "the one-shot fix proposal itself": "一次性修复提案本身",
    "the randomized scene this case ran in — the same fault is planted "
    "in several different scenes, so one bad method can't get lucky "
    "once and pass": (
        "本案例运行所在的随机场景 — 同一故障被植入多个不同场景，坏方法无法"
        "靠一次好运过关"
    ),
    "difficulty group D1 — missing model elements: the library lacks "
    "something it needs (an operator or a predicate was removed)": (
        "难度组 D1 — 缺少模型元素：符号库缺了所需的东西（某个算子或谓词被" "移除）"
    ),
    "difficulty group D2 — wrong model content: something in the library "
    "is incorrect (an inverted precondition, a wrong effect, a wrong "
    "parameter type)": (
        "难度组 D2 — 模型内容错误：库中有内容不正确（前提条件写反、效果"
        "错误、参数类型错误）"
    ),
    "difficulty group D3 — combined defects and platform mismatches: "
    "several things broken at once, or the platform cannot support the "
    "task at all": (
        "难度组 D3 — 复合缺陷与平台不匹配：多处同时损坏，或平台根本无法" "支持该任务"
    ),
    "clear filters": "清除筛选",
    "nothing matches the current filters — ": "没有内容匹配当前筛选 — ",
    "clear them": "点此清除",
    "&lsaquo; prev": "&lsaquo; 上一页",
    "next &rsaquo;": "下一页 &rsaquo;",
    "open it on its own page": "在独立页面打开",
    "too large to show inline": "太大，不在此内联展示",
    # ---- fault templates -------------------------------------------------
    "the library never learned how to close a drawer": ("符号库从未学过如何关抽屉"),
    "the library never learned how to open a drawer": ("符号库从未学过如何开抽屉"),
    "the joint-state concept 'opened' (and everything using it) is "
    "gone": "关节状态概念 ‘opened’（及所有用到它的内容）不见了",
    "the joint-state concept 'closed' (and everything using it) is "
    "gone": "关节状态概念 ‘closed’（及所有用到它的内容）不见了",
    "open-drawer demands the drawer be already opened": (
        "open-drawer 要求抽屉已经是打开的"
    ),
    "open-drawer claims to achieve 'closed' instead of 'opened'": (
        "open-drawer 声称达成 ‘closed’ 而不是 ‘opened’"
    ),
    "open-drawer forgets that opening un-closes the drawer": (
        "open-drawer 忘了开抽屉会取消 'closed' 状态"
    ),
    "open-drawer requests the CLOSED target state": (
        "open-drawer 请求了 CLOSED 目标状态"
    ),
    "open-drawer no longer requires reachability": ("open-drawer 不再要求可达性"),
    "open-drawer types its handle parameter as a drawer": (
        "open-drawer 把把手参数的类型标成了抽屉"
    ),
    "the articulation contract no longer covers the claimed effect": (
        "关节能力契约不再覆盖所声称的效果"
    ),
    "models drawer opening the mobile-robot way on a fixed arm": (
        "在固定机械臂上按移动机器人的方式建模开抽屉"
    ),
    "'opened' references an evaluator no embodiment registers": (
        "‘opened’ 引用了任何平台都未注册的求值器"
    ),
    "no close operator, and open-drawer also forgets to un-close": (
        "没有关抽屉算子，且 open-drawer 也忘了取消 ‘closed’"
    ),
    "open-drawer requests CLOSED and the close operator is missing": (
        "open-drawer 请求 CLOSED 且关抽屉算子缺失"
    ),
    # ---- E2 --------------------------------------------------------------
    "How this experiment works": "这个实验怎么运作",
    "— open for a 30-second primer": "— 展开看 30 秒入门",
    "we break the correct library on purpose, in one known way per "
    "row": "我们故意弄坏正确的符号库，每行一种已知坏法",
    "we then offer candidate fixes: ": "然后我们提供候选修复：",
    " is the right one, the others are wrong on purpose — to see if the "
    "reviewer catches them": (" 是正确的那个，其余故意做错 — 看审核者能否识破"),
    "an independent, deterministic reviewer (the curator) tests each fix "
    "before letting it into the library; each column is one test plan, "
    "from a single quick test to the full suite": (
        "一个独立、确定性的审核者（curator）在放行前测试每个修复；每一列是"
        "一种测试方案，从单次快测到完整套件"
    ),
    "letting a bad fix in is a ": "放进一个坏修复叫",
    "; rejecting the right fix is a ": "；拒掉正确修复叫",
    "accepted fixes are re-tested later on scenes the reviewer never "
    "saw (": "被接受的修复之后会在审核者从未见过的场景上复测（",
    "Every decision, one by one. Each row: one injected fault, probed "
    "with one candidate fix. Each column: one test plan, weakest to "
    "strictest. Green = the plan decided correctly; red = it let a bad "
    "fix in or rejected the right one. Hover any dotted name for what "
    "it means.": (
        "每个决定逐一列出。每行：一个注入的故障，配一个候选修复。每列："
        "一种测试方案，从最弱到最严。绿色 = 方案判断正确；红色 = 放进了坏"
        "修复或拒掉了正确修复。悬停任何虚线词条可见释义。"
    ),
    "One row per test plan, weakest first. Stricter plans run more tests "
    "and let fewer bad fixes through. Hover any dotted term for its "
    "meaning; the numbers match the report below.": (
        "每行一种测试方案，最弱在前。越严的方案跑越多测试、放过越少坏修复。"
        "悬停任何虚线词条可见释义；数字与下方报告一致。"
    ),
    "the known-correct repair for the injected fault": ("针对注入故障的已知正确修复"),
    "deliberately faulty probe: reachability precondition removed, so it "
    "claims success on out-of-reach targets": (
        "故意做错的探针：移除了可达性前提，因此在够不到的目标上也声称成功"
    ),
    "deliberately faulty probe: the operator requests the opposite "
    "target state": "故意做错的探针：算子请求了相反的目标状态",
    "tests the fix only on the one scene where the failure happened": (
        "只在故障发生的那一个场景上测试修复"
    ),
    "tests the fix in several scenes, but only checks that it works": (
        "在多个场景测试修复，但只检查它能否成功"
    ),
    "also runs counterexamples that a bad fix would wrongly pass": (
        "还会运行坏修复会错误通过的反例"
    ),
    "adds edge-case scenes on top of that": "在此之上再加边界场景",
    "runs every test: working scenes, counterexamples, edge cases, and "
    "regression": "运行全部测试：正常场景、反例、边界场景与回归",
    "the test plan the curator runs before accepting or rejecting a "
    "fix": "curator 在接受或拒绝修复前运行的测试方案",
    "bad fixes that were accepted, out of all fixes accepted — lower is "
    "better": "被接受的修复中坏修复的占比 — 越低越好",
    "good fixes that were wrongly rejected, out of all good fixes "
    "offered — lower is better": ("被错误拒绝的好修复占全部好修复的比例 — 越低越好"),
    "accepted fixes that later passed tests on scenes the curator never "
    "saw — higher is better": (
        "被接受且随后在 curator 未见过的场景上通过测试的比例 — 越高越好"
    ),
    "average number of tests the plan ran per decision — its cost": (
        "该方案平均每个决定运行的测试数 — 即成本"
    ),
    "admission decisions — one per candidate patch and policy": (
        "准入决定 — 每个候选补丁 × 每种方案一条"
    ),
    "&#10003; succeeded": "&#10003; 成功",
    "&#10007; failed": "&#10007; 失败",
    "run.json records no end time — the run is either still going or "
    "was interrupted": ("run.json 没有记录结束时间 — 运行仍在进行中，或者被中断了"),
    "executed & verified": "已执行并通过验证",
    "execution violated": "执行违规",
    " — plan: ": " — 计划: ",
    "where the time went": "时间花在哪了",
    "from the run start (or the previous stage's last event) to this "
    "stage's last logged event": (
        "从运行开始（或上一阶段最后一条事件）到本阶段最后一条记录的事件"
    ),
    "read this case's LLM conversations": "阅读这个案例的 LLM 对话",
    "how did every backend fare on this case? see the settled verdicts": (
        "每个 backend 在这个案例上表现如何？查看已结算的裁决"
    ),
    ">experiment E1<": ">实验 E1<",
    ">experiment E2<": ">实验 E2<",
    ">experiment E1+E2<": ">实验 E1+E2<",
    "A case is one fault template injected into one scene; every "
    "backend attacks the same case, so its rows compare like for like.": (
        "一个案例 = 一种故障模板注入到一个场景；每个 backend 都攻克同一个"
        "案例，因此每行之间是同等条件的对比。"
    ),
    "one episode = one backend's full attempt at one case; "
    "cases × backends = episodes": (
        "一条 episode = 一个 backend 对一个案例的一次完整尝试；"
        "案例数 × backend 数 = episode 数"
    ),
    "one case = one fault template injected into one scene; "
    "every backend attacks the same case": (
        "一个案例 = 一种故障模板注入到一个场景；每个 backend 都攻克" "同一个案例"
    ),
    "experiment episodes — one record per repair attempt": (
        "实验 episode — 每次修复尝试一条记录"
    ),
    "Outcome counts per repair backend; open a backend for its "
    "per-fault-template breakdown. Raw records stay in "
    "episodes.jsonl.": (
        "各修复 backend 的结果计数；展开某个 backend 可见其按故障模板的"
        "细分。原始记录保存在 episodes.jsonl。"
    ),
    # ---- provenance / misc files ----------------------------------------
    "reproducibility record — code version and command line": (
        "可复现性记录 — 代码版本与命令行"
    ),
    "What produced this run: the exact code version and command line, "
    "for bit-level reproducibility.": (
        "本次运行由什么产生：确切的代码版本与命令行，用于比特级复现。"
    ),
    "full provenance JSON": "完整 provenance JSON",
    "stage log — timestamped events this stage emitted": (
        "阶段日志 — 该阶段发出的带时间戳事件"
    ),
    "object universe — the typed objects extracted from the world": (
        "对象全集 — 从世界中抽取的带类型对象"
    ),
    "task goal — the literals that must end up TRUE": (
        "任务目标 — 最终必须为 TRUE 的文字（literal）"
    ),
    "PDDL domain — the operators, projected for the planner": (
        "PDDL domain — 投影给规划器的算子"
    ),
    "PDDL problem — this scene's objects and grounded truth": (
        "PDDL problem — 本场景的对象与接地真值"
    ),
    "task result — plan, execution verdict, cost counters": (
        "任务结果 — 计划、执行裁定与成本计数"
    ),
    "execution trace — the event stream the live page renders": (
        "执行轨迹 — 实时页面渲染所用的事件流"
    ),
    "uncommitted changes at run time (empty when the tree was clean)": (
        "运行时未提交的改动（工作树干净则为空）"
    ),
    "video recording metadata": "录像元数据",
    "video encoder log": "视频编码器日志",
    "video recording": "视频录像",
    "clean tree": "工作树干净",
    "dirty tree": "工作树有改动",
    "show raw JSON": "查看原始 JSON",
    "show raw PDDL": "查看原始 PDDL",
    "show JSON": "查看 JSON",
    # ---- PDDL / universe / result ---------------------------------------
    "What the planner was given: the predicate vocabulary and the "
    "projected actions. The cram-type-… predicates are static type "
    "atoms standing in for the CRAM class hierarchy.": (
        "交给规划器的内容：谓词词汇表与投影后的动作。cram-type-… 谓词是"
        "代表 CRAM 类型层级的静态类型原子。"
    ),
    "This scene, written down for the planner: the object names, every "
    "grounded atom that held (grouped by predicate), and the goal.": (
        "写给规划器的本场景：对象名、成立的每个接地原子（按谓词分组），" "以及目标。"
    ),
    "Grouped by type — open a type for its instances (hover an instance "
    "for the world body it denotes).": (
        "按类型分组 — 展开类型可见实例（悬停实例可见其指代的世界物体）。"
    ),
    "How the task went: the plan the planner found, whether execution "
    "and checks succeeded, and what it cost (hover a counter for its "
    "meaning).": (
        "任务进展如何：规划器找到的计划、执行与检查是否成功、花费多少"
        "（悬停计数可见含义）。"
    ),
    "Init atoms by predicate": "初始原子（按谓词）",
    # ---- live page -------------------------------------------------------
    "Live symbolic execution": "实时符号执行",
    "Execution replay: this task's recorded event stream, reduced to "
    "the same panels as the live execution page — shown in its final "
    "state.": (
        "执行回放：该任务录制的事件流，浓缩为与实时页相同的面板 — 展示其" "最终状态。"
    ),
    "Execution replay": "执行回放",
    "Grounded plan": "接地后的计划",
    "✓ done · ▶ active · ○ pending. Open a step record for its full "
    "platform chain and checks.": (
        "✓ 完成 · ▶ 进行中 · ○ 待执行。展开步骤记录可见完整平台链路与检查。"
    ),
    "Selected action": "当前动作",
    "How the active plan step becomes platform motion, layer by "
    "layer.": "当前计划步骤如何逐层变成平台动作。",
    "Grounded Operator": "接地算子",
    "Truth recomputed around the active action: what each precondition, "
    "effect and goal literal was expected to be, and what the world "
    "actually says.": (
        "围绕当前动作重算的真值：每个前提、效果与目标文字的预期值，以及"
        "世界的实际值。"
    ),
    "Recent events": "最近事件",
    "The raw tail of the trace this whole view is rendered from "
    "(trace.jsonl) — the last 12 entries, for debugging.": (
        "本视图渲染所依据的轨迹原始尾部（trace.jsonl）— 最后 12 条，供" "调试。"
    ),
    "· Model evaluation": "· 模型求值",
    "· Plan ready": "· 计划就绪",
    "· Executing": "· 执行中",
    "· Verifying": "· 验证中",
    " — shown here as its final state for review. Start a new task and "
    "this page follows it automatically.": (
        " — 此处显示其最终状态供查看。启动新任务后本页会自动跟随。"
    ),
    "waiting for planner": "等待规划器",
    "waiting for events": "等待事件",
    "no checks yet": "尚无检查",
    "checking preconditions": "检查前提条件",
    "Coraplex executing": "Coraplex 执行中",
    "verifying effects": "验证效果",
    "execution starting": "开始执行",
    "waiting for a run": "等待运行",
    "Waiting for a run…": "等待运行…",
    " · expected ": " · 预期 ",
    " · observed ": " · 实际 ",
    ">goal: ": ">目标: ",
    "predicate evaluations": "谓词求值次数",
    "grounding seconds": "接地耗时（秒）",
    "planning seconds": "规划耗时（秒）",
    "replanning rounds": "重规划轮数",
    "init atoms": "初始原子数",
    "A non-terminal trace with no event for this long is flagged as "
    "stalled.": "非终止轨迹这么久没有事件，会被标记为疑似停滞。",
    "step record": "步骤记录",
    # ---- library pages ---------------------------------------------------
    "The shipped symbol libraries and capability contracts — the system "
    "layer, independent of any run. One tab per library; the graph "
    "shows how predicates, operators and capabilities connect.": (
        "随系统发布的符号库与能力契约 — 系统层，独立于任何运行。每个库一个"
        "标签页；图中展示谓词、算子与能力如何关联。"
    ),
    "Hover a symbol to trace its connections; click it to pin the trace "
    "and jump to its definition below.": (
        "悬停符号可追踪其关联；点击可固定追踪并跳到下方定义。"
    ),
    "— each stores how to decide, never truth values": (
        "— 每个只存如何判定，从不存真值"
    ),
    "— symbolic actions and their capability bindings": ("— 符号动作及其能力绑定"),
    "Typed parameters, preconditions, effects — and the binding that "
    "maps parameters and constants onto one capability's roles.": (
        "带类型参数、前提与效果 — 以及把参数和常量映射到某个能力角色上的" "绑定。"
    ),
    "— frozen platform interfaces; the repair loop may never change "
    "them": "— 冻结的平台接口；修复循环永远不得修改",
    "What the platform can physically do: semantic roles, a success "
    "condition, and the effects that can be verified afterwards.": (
        "平台在物理上能做什么：语义角色、成功条件，以及事后可验证的效果。"
    ),
    "+ added · − removed · ~ changed (each changed field shown as old → "
    "new).": "+ 新增 · − 移除 · ~ 修改（每个修改字段以 旧 → 新 展示）。",
    "no differences": "无差异",
    "the faulted library the repair started from": ("修复起点：被弄坏的符号库"),
    "the library after the curator admitted the patch": (
        "curator 准入补丁之后的符号库"
    ),
    "This run recorded no library snapshots or version stores.": (
        "本次运行未记录库快照或版本库。"
    ),
    "No library directory configured; pass": "未配置符号库目录；请传入",
    " or keep a ": " 或保留一个 ",
    "directory beside the runs root.": "目录在 runs 根目录旁。",
    "known-good": "已知正确",
    "changed since last turn": "较上一轮有变化",
    "same as last turn": "与上一轮相同",
    "new this turn": "本轮新增",
    "full raw prompt": "完整原始提示词",
    ">role &amp; instructions": ">角色与指令",
    "prompt — identical to attempt 1": "提示词 — 与第 1 次尝试相同",
    ">Failure certificate": ">失败证书（Failure certificate）",
    ">Current library": ">当前符号库（Current library）",
    ">Menus (the only building blocks you may reference)": (">可用构件菜单（Menus）"),
    ">Episode so far": ">本回合历史（Episode so far）",
    ">Tools <": ">可用工具（Tools） <",
    ">Rules <": ">行为规则（Rules） <",
    "✓ admitted": "✓ 准入",
    "✓ rejected": "✓ 拒绝",
    "✗ admitted": "✗ 误准入",
    "✗ rejected": "✗ 误拒绝",
    # ---- capability chain anatomy and contract cards ---------------------
    "— one worked example, opening a drawer": "— 一个贯穿实例：开抽屉",
    "design time — stored in the symbol library": "设计时 — 保存在符号库里",
    "run time — created per dispatched action": ("运行时 — 每派发一步动作生成一次"),
    "dispatching one plan step fills the roles with grounded objects ↓": (
        "派发一步计划动作时，用接地后的具体对象填入各角色 ↓"
    ),
    "a named yes/no question": "一个有名字的是/否问题",
    "planned action: preconditions, effects": "规划动作：前提与效果",
    "maps parameters to roles": "把算子参数映射到能力角色",
    "reviewed platform interface": "经审核的平台语义接口",
    "one message to the platform": "发给平台的一条执行消息",
    "translates to a native Coraplex action": "翻译成 Coraplex 原生动作",
    "by the predicate's truth procedure": "由该谓词的真值程序完成",
    "— re-checked after execution by each "
    "predicate's truth procedure": "— 执行后由各谓词的真值程序复查",
    "none in the shipped libraries": "随附符号库中没有算子绑定它",
    "view in the capability catalog →": "在能力目录中查看 →",
    "— base motion toward goals": "— 底盘移动到目标位置",
    "— drawers, doors and other jointed parts": "— 抽屉、柜门等带关节的部件",
    "— pointing the robot's sensors at something": ("— 把机器人的传感器对准目标"),
    "— detecting objects in the scene": "— 在场景中探测物体",
    "— reaching, grasping, picking up, placing, transporting": (
        "— 伸手、抓取、拿起、放置、搬运"
    ),
    "— gripper, arm, torso and carry postures": ("— 夹爪、手臂、躯干与携带姿态"),
    "— mixing, pouring, cutting": "— 搅拌、倾倒、切割",
}

ZH_TERMS: dict[str, str] = {
    "How the pieces connect": "各部件如何衔接",
    "used by operators": "被哪些算子使用",
    "success when": "成功判据",
    "verifiable effects": "可验证效果",
    "operator": "算子（operator）",
    "binding": "绑定（binding）",
    "adapter": "适配器（adapter）",
    "verification": "验证（verification）",
    "name": "名称",
    "stable id": "全局编号",
    "what it does": "做什么（签名）",
    "roles": "角色（roles）",
    "implemented by": "由哪些原生动作实现",
    "Moving around": "移动",
    "Opening and closing": "开与关",
    "Looking": "看向目标",
    "Perceiving": "感知物体",
    "Handling objects": "抓放操作",
    "Body posture": "身体姿态",
    "Working with material": "材料加工",
    "runs": "运行记录",
    "Runs": "运行记录",
    "Results": "结果",
    "libraries": "符号库",
    "artifacts": "产物",
    "run overview": "运行总览",
    "live execution": "实时执行",
    "system library": "系统符号库",
    "label": "标签",
    "started": "开始",
    "ended": "结束",
    "duration": "时长",
    "scene": "场景",
    "summary": "摘要",
    "commit": "提交",
    "invocation": "调用命令",
    "backend": "backend（修复方法）",
    "episodes": "episode 数",
    "admitted": "被准入",
    "calls": "调用数",
    "LLM role": "LLM 角色",
    "exchanges": "调用总数",
    "all exchanges": "全部调用",
    "conversations": "对话数",
    "repair cases": "修复案例数",
    "parse errors": "解析失败",
    "tokens": "token 用量",
    "difficulty": "难度组",
    "fault": "故障",
    "role": "角色",
    "fix": "修复",
    "review": "审核",
    "verdict": "裁定",
    "cases": "案例数",
    "cost": "开销",
    "fault never tripped": "故障未触发",
    "unfinished": "未完成",
    "task outcome": "任务结果",
    "proposed, rejected at review": "已提案，审核未通过",
    "no fix found": "未找到修复",
    "budget exhausted": "预算耗尽",
    "rightly declared unsupported": "正确判定为不支持",
    "wrongly declared unsupported": "误判为不支持",
    "re-check": "复测",
    "decisions": "决定数",
    "correct": "正确",
    "false/missed": "误准/漏准",
    "curation policy": "审核方案",
    "false admissions": "误准入",
    "missed admissions": "漏准入",
    "held-out clean": "held-out 通过率",
    "mean tests": "平均测试数",
    "fault template": "故障模板",
    "candidate patch": "候选补丁",
    "correct repair": "正确修复",
    "false admission": "误准入",
    "unsupported detection": "识别不支持",
    "outcomes": "结果",
    "finished": "已结束",
    "stalled?": "疑似停滞？",
    "no summary": "无摘要",
    "size": "大小",
    "text file": "文本文件",
    "event log": "事件日志",
    "JSON document": "JSON 文档",
    "prompt": "提示词",
    "response": "回复",
    "Checks": "检查",
    "predicate": "谓词",
    "signature": "签名",
    "truth procedure": "真值程序",
    "kind": "类别",
    "source": "来源",
    "Predicates": "谓词",
    "Operators": "算子",
    "Capability contracts": "能力契约",
    "capability": "能力",
    "platform": "平台",
    "coraplex": "coraplex",
    "waiting": "等待中",
    "goal": "目标",
    "failure": "失败类别",
    "results": "结果",
    "world": "世界",
    "repair process": "修复过程",
    "task solve": "任务求解",
    "experiment results": "实验结果",
    "experiment": "实验",
    "visualization": "可视化",
}

_CHIP_PREFIXES = {
    ">fault: ": ">故障: ",
    ">group: ": ">难度组: ",
    ">scene-": ">场景-",
    ">goal: ": ">目标: ",
    ">failure: ": ">失败类别: ",
    ">backend: ": ">backend: ",
}


def _variants(text: str) -> list[str]:
    forms = []
    for candidate in (
        text,
        html.escape(text, quote=False),
        html.escape(text, quote=True),
    ):
        if candidate not in forms:
            forms.append(candidate)
    return forms


_PATTERNS = [(re.compile(pattern), repl) for pattern, repl in ZH_PATTERNS]
_PHRASES = sorted(
    (
        (variant, zh)
        for en, zh in {**ZH_PHRASES, **_CHIP_PREFIXES}.items()
        for variant in _variants(en)
    ),
    key=lambda pair: -len(pair[0]),
)
_TERMS = [
    (f">{variant}<", f">{zh}<")
    for en, zh in ZH_TERMS.items()
    for variant in _variants(en)
]


def to_chinese(page: str) -> str:
    """
    Translate one rendered HTML page (or fragment) to Chinese.
    """
    for pattern, repl in _PATTERNS:
        page = pattern.sub(repl, page)
    for en, zh in _PHRASES:
        page = page.replace(en, zh)
    for en, zh in _TERMS:
        page = page.replace(en, zh)
    return page
