"""
A small web viewer for the per-run stage logs written by :mod:`runlog`.

It only reads a runs root (default ``runs/``, or ``$RESYM_RUNS_DIR``) and serves a
browsable UI: an index of runs, and per-run pages grouped by pipeline stage (A_world /
B_repair / C_solve). Known artifacts get purpose-built, compact renderings (object
domains collapse to a per-type summary, LLM prompts/responses wrap and fold, task
results show plan + verdict + metrics); anything unfamiliar falls back to pretty JSON,
so it keeps working when demos add new artifacts.

The viewer needs nothing from the CRAM stack, so it runs on the host:

uv run --extra viewer python -m resym.observability.viewer runs/ uv run --extra viewer
python -m resym.observability.viewer --port 8000
"""

from __future__ import annotations

import argparse
import html
import itertools
import json
import os
import re
from urllib.parse import urlencode
from collections import Counter
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, redirect, request, send_file, url_for

from resym.core.grounding_model import GroundingFactoryCandidate, GroundingFactorySpec
from resym.observability.i18n import to_chinese
from resym.planning.events import PipelineEvent
from krrood.adapters.json_serializer import from_json
from resym.core.capability_model import CapabilityContract
from resym.platform.capability_contract_review import (
    CapabilityContractCandidate,
    CapabilityContractWorkspace,
)
from resym.platform.coraplex_catalog import (
    approve_realization_candidate,
    discover_coraplex_capability_contract_drafts,
    initialize_coraplex_capabilities,
)
from resym.platform.coraplex_realizations import (
    CapabilityReviewStatus,
    CoraplexRealizationWorkspace,
    RealizationCandidate,
)
from resym.platform.grounding_catalog import (
    GroundingFactoryCatalog,
    GroundingFactoryWorkspace,
    GroundingVocabulary,
    GroundingVocabularyCandidate,
    initialize_grounding_factories,
)

STAGE_TITLES = {
    "A_world": "Stage A · world",
    "B_repair": "Stage B · repair process",
    "C_solve": "Stage C · task solve",
    "D_visualization": "Stage D · visualization",
}

PIPELINE_STEPS = (
    (
        "A_world",
        "A · World",
        "Load the scene, extract the typed planning objects.",
        "always",
    ),
    (
        "B_repair",
        "B · Repair",
        "An LLM proposes a library patch — only when a symbol gap made "
        "planning fail.",
        "no model gap in this run, so no repair happened",
    ),
    (
        "C_solve",
        "C · Solve",
        "Ground predicates → plan → execute → verify, once per task.",
        "always",
    ),
    (
        "D_visualization",
        "D · Visualization",
        "RViz recording of the execution.",
        None,
    ),
)
"""
The pipeline map at the top of a run page: (stage dir, label, what it does, and what an
absent stage means — None hides absent stages).
"""

FILE_EXPLAINERS = {
    "events.jsonl": "stage log — timestamped events this stage emitted",
    "universe.json": "object universe — the typed objects extracted from the world",
    "goal.json": "task goal — the literals that must end up TRUE",
    "domain.pddl": "PDDL domain — the operators, projected for the planner",
    "problem.pddl": "PDDL problem — this scene's objects and grounded truth",
    "result.json": "task result — plan, execution verdict, cost counters",
    "trace.jsonl": "execution trace — the event stream the live page renders",
    "llm_transcript.jsonl": (
        "the LLM call log — every prompt and reply, kept as the audit trail"
    ),
    "provenance.json": "reproducibility record — code version and command line",
    "source.patch": "uncommitted changes at run time (empty when the tree was clean)",
    "rviz_recording.json": "video recording metadata",
    "ffmpeg.log": "video encoder log",
}
"""
One-line identity for known artifact names, shown next to the filename.
"""

STAGE_EXPLAINERS = {
    "A_world": (
        "The typed objects this run planned over, extracted from the "
        "annotated world model."
    ),
    "B_repair": (
        "The repair process, recorded raw: the LLM call log keeps every "
        "prompt, reply, and retry. This is the how — what became of each "
        "proposal is settled by the curator and recorded with the run."
    ),
    "C_solve": (
        "One folder per task (named scene_task, e.g. apartment_open): the "
        "goal, the generated PDDL, the plan, and the execution trace."
    ),
    "D_visualization": "RViz recordings of the execution.",
}

WORKING_DIR_NAMES = {"work", "versions", "held_out", "admission", "proposal"}
"""
Per-episode working trees (probe outputs, version stores, suite scratch): thousands of
intermediate files that would swamp the page, so the run view names them instead of
rendering them.
"""

_STYLE = """
:root{--bg:#f5f6f8;--fg:#1f2430;--muted:#6b7280;--line:#e3e6ea;--accent:#2563eb;
      --ok-bg:#dcfce7;--ok-fg:#166534;--bad-bg:#fee2e2;--bad-fg:#991b1b}
*{box-sizing:border-box}
body{font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;color:var(--fg);background:var(--bg)}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
header{position:sticky;top:0;z-index:5;background:#111827;color:#fff;padding:11px 20px;
       display:flex;gap:16px;align-items:center;box-shadow:0 1px 4px rgba(0,0,0,.2)}
header .home{color:#93c5fd;font-weight:600}
header nav{margin-left:auto;display:flex;gap:14px;font-size:13px}
header nav a{color:#cbd5e1}
main{max-width:1080px;margin:0 auto;padding:22px 20px 60px}
h2.stage{margin:30px 0 4px;font-size:18px;display:flex;align-items:center;gap:10px}
h2.stage::before{content:"";width:5px;height:20px;background:var(--accent);border-radius:3px}
p.explain{color:var(--muted);font-size:12.5px;margin:0 0 10px;max-width:70ch}
h3.sub{margin:16px 0 2px;font-size:14.5px}
.metric.t-true b{color:var(--ok-fg)}.metric.t-false b{color:var(--bad-fg)}
.file{margin:22px 0}
.file>.name{margin-bottom:6px;padding:3px 0 3px 10px;border-left:3px solid #94a3b8;
            font:600 14px system-ui,sans-serif;color:var(--fg)}
.file>.name code{background:#eef2ff;padding:1px 7px;border-radius:4px;
                 font:600 13px ui-monospace,Menlo,monospace}
.file>.name .whatis{font:italic 12.5px system-ui,sans-serif;color:var(--muted)}
.card{background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px 15px;margin:6px 0}
pre{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:6px;overflow:auto;
    font:12.5px/1.5 ui-monospace,Menlo,monospace;margin:6px 0}
pre.wrap{white-space:pre-wrap;word-break:break-word;overflow:visible}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid var(--line);padding:5px 9px;text-align:left}
th{background:#f3f4f6;position:sticky;top:0}
.tablewrap{max-height:320px;overflow:auto;border:1px solid var(--line);border-radius:6px;margin-top:6px}
.chip{display:inline-block;background:#eef2ff;color:#3730a3;border-radius:12px;
      padding:1px 9px;margin:2px 4px 2px 0;font-size:12px;white-space:nowrap}
.metric{display:inline-block;background:#f1f5f9;border:1px solid var(--line);border-radius:6px;
        padding:3px 9px;margin:3px 6px 0 0;font-size:12.5px}
.metric b{color:var(--fg)}
.badge{display:inline-block;padding:1px 9px;border-radius:11px;font-size:12px;margin-left:6px;font-weight:600}
.ok{background:var(--ok-bg);color:var(--ok-fg)}.bad{background:var(--bad-bg);color:var(--bad-fg)}
.warn{background:#fef3c7;color:#92400e}
.verdict{border-left:4px solid #f59e0b;padding-left:14px}
.verdict .badge{margin-left:0;margin-right:8px;font-size:13px;padding:3px 12px}
.pager{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin:10px 0;font-size:13px}
.pager a{border:1px solid var(--line);border-radius:6px;padding:3px 10px;background:#fff}
.pager a.on{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:700}
.timeline{display:flex;gap:3px;flex-wrap:wrap}
.tl-seg{background:#e0e7ff;border-radius:5px;padding:3px 9px;font-size:12px;
        white-space:nowrap;overflow:hidden;min-width:52px}
.tl-seg b{font-weight:600}
.facets{display:flex;flex-direction:column;gap:8px}
.facet{display:flex;gap:6px;align-items:center;flex-wrap:wrap;font-size:13px}
.facet .lbl{color:var(--muted);min-width:72px;font-size:12px;text-transform:uppercase;
            letter-spacing:.04em}
.facet a{border:1px solid var(--line);border-radius:14px;padding:2px 11px;background:#fff}
.facet a.on{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
.facet a.on:hover{text-decoration:none}
.facet select{border:1px solid var(--line);border-radius:6px;padding:3px 8px;font:13px inherit;
              background:#fff;max-width:100%}
.langswitch{margin-left:14px;white-space:nowrap}
.langswitch a.lang{border:1px solid var(--line);padding:2px 9px;font-size:12px;color:var(--muted)}
.langswitch a.lang:first-child{border-radius:6px 0 0 6px;border-right:0}
.langswitch a.lang:last-child{border-radius:0 6px 6px 0}
.langswitch a.lang.on{background:var(--accent);border-color:var(--accent);color:#fff}
.langswitch a.lang:hover{text-decoration:none}
.openlink{display:inline-block;margin-top:8px;font-weight:600}
details.turn{margin:9px 0;border-left:2px solid var(--line)}
details.turn[open]{border-left-color:var(--accent)}
details.turn>summary{font-weight:650;color:var(--fg);background:#eef2f7;
                     padding:5px 11px;border-radius:0 6px 6px 0}
details.turn[open]>summary{background:#e3ebfd}
details.turn>:not(summary){margin-left:22px;margin-right:6px}
.prompt-secs{margin:4px 0 4px 14px}
details.psec{margin:3px 0;padding-left:9px;border-left:2px solid var(--line)}
details.psec[open]{border-left-color:#a5b4fc}
details.psec>summary{font-size:12.5px;color:var(--fg)}
pre.psec-body{max-height:340px;overflow:auto}
details.earlier>summary{color:var(--muted)}
details.earlier{border-left:2px dotted var(--line);padding-left:9px}
.case{border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:10px;
      padding:11px 14px;margin:16px 0;background:#fbfcfe}
.case .card{margin:8px 0;box-shadow:none;border-style:dashed}
.case-head{margin-bottom:4px}
.case-head .muted{font-size:12.5px;display:block;margin-top:3px}
.badbox{background:var(--bad-bg);color:var(--bad-fg);border:1px solid #fca5a5;
        border-radius:6px;padding:8px 12px}
.run{display:flex;justify-content:space-between;align-items:center;padding:12px 15px;margin:8px 0;
     background:#fff;border:1px solid var(--line);border-radius:8px}
.run:hover{border-color:var(--accent);text-decoration:none}
.run .meta{color:var(--muted);font-size:12.5px;margin-top:2px}
.run b{font:13px ui-monospace,Menlo,monospace}
ol.plan{margin:6px 0;padding-left:22px}ol.plan li{margin:2px 0}
code{background:#eef2ff;padding:1px 6px;border-radius:4px;font:12.5px ui-monospace,Menlo,monospace}
details{margin:5px 0}details>summary{cursor:pointer;color:var(--accent);font-size:13px;user-select:none}
.evt{display:flex;gap:10px;align-items:baseline;padding:6px 0;border-bottom:1px solid var(--line)}
.evt:last-child{border-bottom:0}
.evt .t{color:var(--muted);font:11.5px ui-monospace,monospace;white-space:nowrap}
.evt .n{font-weight:600}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:2px 14px;font-size:13px}
.kv div.k{color:var(--muted)}
.muted{color:var(--muted)}
.live-head{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:14px}
.phase-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:0 0 14px}
.phase-step{padding:7px 10px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--muted)}
.phase-step.active{border-color:var(--accent);color:var(--accent);font-weight:700}
.phase-step.done{border-color:#86efac;background:var(--ok-bg);color:var(--ok-fg)}
.live-grid{display:grid;grid-template-columns:minmax(240px,.8fr) minmax(380px,1.4fr);gap:14px}
.live-grid h3{font-size:14px;margin:0 0 9px}.live-plan{list-style:none;margin:0;padding:0}
.live-plan li{display:flex;gap:8px;align-items:flex-start;padding:8px 5px;border-bottom:1px solid var(--line)}
.live-plan li:last-child{border-bottom:0}.live-plan .mark{width:18px;text-align:center;font-weight:700}
.live-plan .action-summary{margin-top:3px;font-size:11.5px;color:var(--muted)}
.live-plan .active{color:var(--accent)}.flow{display:grid;gap:7px}.flow .arrow{color:var(--muted);padding-left:8px}
.check{display:grid;grid-template-columns:90px 1fr max-content;gap:8px;padding:6px 0;border-bottom:1px solid var(--line)}
.check:last-child{border-bottom:0}.check-pass{color:var(--ok-fg)}.check-fail{color:var(--bad-fg)}
.truth-true{color:var(--ok-fg)}.truth-false{color:var(--bad-fg)}
.live-status{font-weight:700}.timeline{max-height:230px;overflow:auto}
.live-plan details{margin-top:2px}
.live-plan details>summary{font-size:11.5px;color:var(--accent);cursor:pointer;user-select:none}
.stepbody{margin:5px 0 2px;padding:7px 9px;border:1px solid var(--line);border-radius:6px;
          background:#fafbfc;font-size:12px;display:grid;gap:4px}
video.recording{width:100%;max-height:420px;border-radius:6px;background:#000;margin-top:6px}
td.cell-ok{background:var(--ok-bg);color:var(--ok-fg)}
td.cell-bad{background:var(--bad-bg);color:var(--bad-fg)}
td.cell-muted{color:var(--muted)}
.pipemap{display:flex;gap:6px;align-items:stretch;flex-wrap:wrap;margin:14px 0 6px}
.pipestep{flex:1;min-width:170px;background:#fff;border:1px solid var(--line);
          border-radius:8px;padding:9px 12px;font-size:12.5px;color:var(--muted)}
.pipestep b{display:block;color:var(--fg);font-size:13px;margin-bottom:2px}
a.pipestep:hover{border-color:var(--accent);text-decoration:none}
.pipestep.absent{opacity:.55;background:transparent}
.pipearrow{align-self:center;color:var(--muted);font-size:15px}
.pddl-sub{font:700 11.5px system-ui,sans-serif;color:var(--muted);
          text-transform:uppercase;letter-spacing:.05em;margin:12px 0 5px}
.action-card{border-left:3px solid var(--accent)}
.hero{background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:0 0 18px}
.hero h1{margin:0 0 6px;font-size:20px}
.hero p{margin:0;color:var(--muted);font-size:13.5px;max-width:78ch}
.navcards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:0 0 26px}
.navcard{display:block;background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.navcard:hover{border-color:var(--accent);text-decoration:none}
.navcard b{display:block;font-size:15px;margin-bottom:4px}
.navcard span{color:var(--muted);font-size:12.5px;line-height:1.5;display:block}
[data-tip]{position:relative;cursor:help;border-bottom:1px dotted #94a3b8}
[data-tip]:hover::after{content:attr(data-tip);position:absolute;left:0;top:100%;
  margin-top:5px;z-index:30;width:270px;white-space:normal;
  background:#111827;color:#f1f5f9;padding:8px 11px;border-radius:7px;
  font:12px/1.5 system-ui,sans-serif;font-weight:400;text-align:left;
  box-shadow:0 4px 14px rgba(0,0,0,.3);pointer-events:none}
th[data-tip]:hover::after{left:auto;right:0}
.rate-cell{display:flex;align-items:center;gap:8px;min-width:200px}
.rate-bar{position:relative;flex:1;height:10px;background:#e2e8f0;border-radius:5px}
.rate-bar .fill{position:absolute;left:0;top:0;height:100%;border-radius:5px;background:var(--accent);display:block}
.rate-bar .fill.bad{background:#dc2626}
.rate-bar .ci{position:absolute;top:-2px;height:14px;display:block;
              border-left:1.5px solid #334155;border-right:1.5px solid #334155;opacity:.55}
.rate-num{font:12px ui-monospace,Menlo,monospace;white-space:nowrap;min-width:88px;text-align:right}
.tabs{display:flex;gap:8px;margin:0 0 16px;flex-wrap:wrap}
.tab{padding:7px 18px;border:1px solid var(--line);border-radius:8px;background:#fff;
     cursor:pointer;font:600 13.5px system-ui,sans-serif;color:var(--muted)}
.tab.active{border-color:var(--accent);color:var(--accent);background:#eff6ff}
.libpane.hidden{display:none}
details.libsec{margin:10px 0;background:#fff;border:1px solid var(--line);border-radius:8px;padding:4px 15px}
details.libsec>summary{cursor:pointer;padding:8px 0;font-size:14px;user-select:none}
details.libsec[open]>summary{border-bottom:1px solid var(--line);margin-bottom:8px}
.graph-legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;margin-bottom:6px}
.graph-legend span{display:inline-flex;align-items:center;gap:5px}
.graph-legend i{display:inline-block;width:22px;height:0;border-top:2.5px solid}
svg.libgraph{display:block}
svg.libgraph .gnode rect{fill:#fff;stroke:#94a3b8;stroke-width:1.2;cursor:pointer}
svg.libgraph .gnode.kind-op rect{stroke:var(--accent)}
svg.libgraph .gnode.kind-binding rect{stroke:#d97706}
svg.libgraph .gnode.kind-cap rect{stroke:#7c3aed}
svg.libgraph .gnode text{font:11.5px ui-monospace,Menlo,monospace;fill:var(--fg);pointer-events:none}
svg.libgraph .gcol{font:600 12px system-ui,sans-serif;fill:var(--muted)}
svg.libgraph .gedge{fill:none;stroke-width:1.7;opacity:.8}
svg.libgraph .gedge.pre{stroke:#2563eb}
svg.libgraph .gedge.add{stroke:#16a34a}
svg.libgraph .gedge.del{stroke:#dc2626}
svg.libgraph .gedge.maps{stroke:#d97706}
svg.libgraph .gedge.bind{stroke:#7c3aed;stroke-dasharray:5 3}
svg.libgraph .dim{opacity:.1}
.flash{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}
.lib-head{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:6px}
.lib-head b{font-size:14.5px}
.lib-row{display:grid;grid-template-columns:110px 1fr;gap:4px 12px;padding:3px 0;font-size:13px}
.lib-row .k{color:var(--muted)}
code.diff-old{background:var(--bad-bg);color:var(--bad-fg)}
code.diff-new{background:var(--ok-bg);color:var(--ok-fg)}
.diff-name{font-weight:600}
.diff-add .diff-name{color:var(--ok-fg)}.diff-del .diff-name{color:var(--bad-fg)}
.diff-chg .diff-name{color:#a16207}
.diff-entry{padding:5px 0;border-bottom:1px solid var(--line)}
.diff-entry:last-child{border-bottom:0}
.catalog-summary{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
.capability-entry{border-left:4px solid #94a3b8}
.capability-entry[data-status=ready]{border-left-color:#16a34a}
.capability-entry[data-status=adapter-missing]{border-left-color:#d97706}
details.capability-entry{background:#fff;border:1px solid var(--line);
    border-left-width:4px;border-radius:8px;padding:7px 14px;margin:8px 0}
details.capability-entry>summary{cursor:pointer;display:flex;gap:10px;
    align-items:baseline;flex-wrap:wrap;user-select:none}
details.capability-entry>summary .sig-text{color:var(--muted);
    font:12.5px ui-monospace,Menlo,monospace}
details.capability-entry>summary .badge{margin-left:auto}
.cap-detail{border-top:1px solid var(--line);margin-top:8px;padding-top:8px}
.cap-cat{margin:22px 0 8px;font-size:15px}
.cap-group:first-of-type .cap-cat{margin-top:10px}
.field{display:inline-flex;align-items:baseline;gap:5px}
.k-inline{color:var(--muted);font-size:11px;white-space:nowrap}
.role-chip{display:inline-block;padding:1px 9px;border-radius:9px;border:1px solid;
           font:12px ui-monospace,Menlo,monospace;white-space:nowrap}
.role-chip.rc0{background:#eff6ff;color:#1d4ed8;border-color:#bfdbfe}
.role-chip.rc1{background:#f0fdf4;color:#15803d;border-color:#bbf7d0}
.role-chip.rc2{background:#fffbeb;color:#b45309;border-color:#fde68a}
.role-chip.rc3{background:#faf5ff;color:#7e22ce;border-color:#e9d5ff}
.role-chip.rc4{background:#fdf2f8;color:#be185d;border-color:#fbcfe8}
.role-hl{outline:2px solid #334155;outline-offset:1px}
.sigline{font:14.5px ui-monospace,Menlo,monospace;margin:2px 0 8px;line-height:2}
.role-row{display:flex;gap:10px;align-items:baseline;padding:3px 6px;border-radius:6px;font-size:13px}
.role-row.role-hl{outline:none;background:#f1f5f9}
.role-note{color:var(--muted)}
.cap-links{display:flex;gap:8px 16px;flex-wrap:wrap;align-items:baseline;
           border-top:1px solid var(--line);margin-top:9px;padding-top:8px;font-size:13px}
.cap-links .k{color:var(--muted)}
.anatomy-lane{font-size:12px;color:var(--muted);margin:4px 0 6px;font-weight:600}
.anatomy-grid{display:grid;gap:6px}
.anatomy-grid.design{grid-template-columns:repeat(4,minmax(0,1fr))}
.anatomy-grid.runtime{grid-template-columns:repeat(3,minmax(0,1fr))}
.anatomy-box{background:#fff;border:1px solid var(--line);border-radius:8px;padding:7px 10px;min-width:0}
.anatomy-box.junction{border-color:#7c3aed}
.anatomy-box .what{font-size:11px;color:var(--muted)}
.anatomy-box code{font-size:12px;word-break:break-word}
.anatomy-join{text-align:right;color:var(--muted);font-size:12px;padding:5px 8px}
@media(max-width:760px){.live-grid{grid-template-columns:1fr}.phase-strip{grid-template-columns:1fr 1fr}.anatomy-grid.design,.anatomy-grid.runtime{grid-template-columns:1fr 1fr}}
"""


def create_app(
    runs_root: Path,
    library_dir: Path | None = None,
    grounding_workspace: GroundingFactoryWorkspace | None = None,
    grounding_vocabulary: GroundingVocabulary | None = None,
    realization_workspace: CoraplexRealizationWorkspace | None = None,
    contract_workspace: CapabilityContractWorkspace | None = None,
) -> Flask:
    runs_root = runs_root.resolve()
    if library_dir is not None:
        library_dir = library_dir.resolve()
    app = Flask(__name__)

    def _current_grounding_vocabulary() -> GroundingVocabulary:
        if (
            grounding_workspace is not None
            and grounding_workspace.vocabulary_candidates()
        ):
            return grounding_workspace.reviewed_vocabulary()
        return grounding_vocabulary or GroundingVocabulary()

    def _lang() -> str:
        value = request.args.get("lang") or request.cookies.get("resym_lang")
        return "zh" if value == "zh" else "en"

    def _localize(fragment: str) -> str:
        return to_chinese(fragment) if _lang() == "zh" else fragment

    @app.after_request
    def _remember_language(response):
        # an explicit ?lang= choice sticks for every later page
        if "lang" in request.args:
            response.set_cookie("resym_lang", _lang(), max_age=31536000)
        return response

    def _lang_switch() -> str:
        current = _lang()
        links = []
        for code, name in (("en", "EN"), ("zh", "中文")):
            args = {k: v for k, v in request.args.items() if k != "lang"}
            args["lang"] = code
            links.append(
                f"<a class='lang{' on' if code == current else ''}' "
                f"href='{html.escape(request.path + '?' + urlencode(args))}'>"
                f"{name}</a>"
            )
        return f"<span class=langswitch>{''.join(links)}</span>"

    def _page(title: str, body: str, nav: str = "") -> str:
        grounding_link = (
            f"<a href='{url_for('grounding_factories')}'>grounding factories</a>"
            if grounding_workspace is not None
            else ""
        )
        page = (
            f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<style>{_STYLE}</style>"
            f"<header><a class=home href='{url_for('index')}'>&#9635; runs</a>"
            f"<span>{html.escape(title)}</span><nav>{nav}"
            f"<a href='{url_for('system_libraries')}'>system library</a>"
            f"<a href='{url_for('capability_catalog')}'>capabilities</a>"
            f"{grounding_link}"
            f"{_lang_switch()}</nav></header>"
            f"<main>{body}</main>"
        )
        return _localize(page)

    def _safe_run_dir(name: str) -> Path:
        run_dir = (runs_root / name).resolve()
        if run_dir.parent != runs_root or not run_dir.is_dir():
            abort(404)
        return run_dir

    def _media_url(path: Path) -> str | None:
        try:
            relative = path.resolve().relative_to(runs_root)
        except ValueError:
            return None
        return url_for("media", relative=str(relative))

    @app.route("/media/<path:relative>")
    def media(relative: str):
        target = (runs_root / relative).resolve()
        if runs_root not in target.parents or not target.is_file():
            abort(404)
        # conditional=True enables HTTP range requests, so the browser can
        # seek inside long RViz recordings without downloading them whole.
        return send_file(target, conditional=True)

    @app.route("/")
    def index() -> str:
        if not runs_root.is_dir():
            return _page(
                "runs",
                f"<p>No runs root at <code>{html.escape(str(runs_root))}</code>.</p>",
            )
        runs = sorted(
            (p for p in runs_root.iterdir() if p.is_dir()),
            key=lambda p: p.name,
            reverse=True,
        )
        if not runs:
            return _page("runs", "<p>No runs recorded yet.</p>")
        latest = runs[0]
        grounding_card = (
            f"<a class=navcard href='{url_for('grounding_factories')}'><b>✓ Grounding factories</b>"
            "<span>Review agent-authored EQL candidates and inspect the "
            "current locally approved predicate queries.</span></a>"
            if grounding_workspace is not None
            else ""
        )
        rows = [
            "<div class=hero><h1>reSym run viewer</h1>"
            "<p>reSym repairs a robot's symbolic planning model when a "
            "missing or wrong symbol makes a task fail. This viewer shows "
            "everything the pipeline records: the symbol libraries it "
            "plans with, each run's plan and execution trace, and the "
            "repair decisions and provenance.</p></div>"
            "<div class=navcards>"
            f"<a class=navcard href='{url_for('latest_live')}'><b>⚡ Live execution</b>"
            "<span>Follow the most recent task as it runs: grounding "
            "counts, the plan step by step, and every Boolean "
            "check.</span></a>"
            f"<a class=navcard href='{url_for('system_libraries')}'><b>🧩 System library</b>"
            "<span>The shipped symbol libraries and capability contracts, "
            "with an interactive graph of how predicates, operators and "
            "capabilities connect.</span></a>"
            f"<a class=navcard href='{url_for('capability_catalog')}'><b>⚙ Capability catalog</b>"
            "<span>All reviewed capability contracts, their realization "
            "status, required robot resources, and native Coraplex "
            "actions.</span></a>"
            f"{grounding_card}"
            f"<a class=navcard href='{url_for('run_page', name=latest.name)}'><b>🕐 Latest run</b>"
            f"<span><code>{html.escape(latest.name)}</code> — its goal, "
            "artifacts and admission results.</span></a>"
            "</div>",
            _section(
                "Runs",
                "Every recorded run, newest first. Open one for its artifacts, "
                "execution trace, and the libraries it used.",
            ),
        ]
        metas = [(run, _load_json(run / "run.json") or {}) for run in runs]
        label_counts = Counter(str(m.get("label") or "?") for _, m in metas)
        selected_label = str(request.args.get("label") or "")
        if selected_label not in label_counts:
            selected_label = ""
        if len(label_counts) > 1:
            rows.append(
                "<div class='card facets'>"
                + _facet_chips(
                    "run type",
                    sorted(label_counts.items(), key=lambda kv: -kv[1]),
                    "label",
                    {"label": selected_label},
                )
                + "</div>"
            )
        for run, meta in metas:
            if selected_label and str(meta.get("label") or "?") != selected_label:
                continue
            summary = meta.get("summary", {})
            metadata = meta.get("metadata") or {}
            badge = _run_row_status(run, meta)
            context_chips = "".join(
                f"<span class=chip>{html.escape(str(value))}</span>"
                for value in (
                    metadata.get("demo") or meta.get("label"),
                    meta.get("scene") or metadata.get("scene"),
                    summary.get("backend") or metadata.get("backend"),
                )
                if value
            )
            meta_bits = [f"started {html.escape(str(meta.get('started_at', '?')))}"]
            duration = _duration_text(meta)
            if duration:
                meta_bits.append(html.escape(duration))
            rows.append(
                f"<a class=run href='{url_for('run_page', name=run.name)}'>"
                f"<span><b>{html.escape(run.name)}</b> {context_chips}"
                f"<div class=meta>{' &middot; '.join(meta_bits)}</div></span>"
                f"<span>{badge}</span></a>"
            )
        return _page("runs", "".join(rows))

    @app.route("/run/<name>")
    def run_page(name: str) -> str:
        run_dir = _safe_run_dir(name)
        subdirectories = sorted(p.name for p in run_dir.iterdir() if p.is_dir())
        stages = [s for s in subdirectories if s not in WORKING_DIR_NAMES]
        skipped = [s for s in subdirectories if s in WORKING_DIR_NAMES]
        result_files = sorted(
            (p for p in run_dir.iterdir() if p.is_file() and p.name != "run.json"),
            key=lambda p: (
                _RESULT_FILE_ORDER.get(p.name, len(_RESULT_FILE_ORDER)),
                p.name,
            ),
        )

        def _artifact_link(path: Path) -> str:
            return url_for(
                "run_artifact",
                name=name,
                relative=str(path.relative_to(run_dir)),
            )

        nav = "".join(
            f"<a href='#{html.escape(s)}'>{html.escape(STAGE_TITLES.get(s, s).split(' · ')[-1])}</a>"
            for s in (["results"] if result_files else []) + stages
        )
        if any(run_dir.glob("library_*.json")) or skipped:
            nav = f"<a href='{url_for('run_libraries', name=name)}'>libraries</a>" + nav
        if _latest_trace(run_dir) is not None:
            nav = f"<a href='{url_for('live_run', name=name)}'>live execution</a>" + nav
        parts = [_render_run_meta(run_dir), _render_pipeline_map(stages)]
        meta = _load_json(run_dir / "run.json") or {}
        parts.append(_task_verdict_card(run_dir, meta))
        if result_files:
            parts.append(
                _section(
                    "Results",
                    "Top-level artifacts of this run: reports, library "
                    "snapshots, recordings.",
                    anchor="results",
                )
            )
            for path in result_files:
                parts.append(
                    _file_block(
                        path,
                        _render_file(
                            path,
                            media_url=_media_url,
                            detail_url=_artifact_link(path),
                        ),
                    )
                )
        for stage in stages:
            parts.append(
                _section(
                    STAGE_TITLES.get(stage, stage),
                    STAGE_EXPLAINERS.get(stage, ""),
                    anchor=stage,
                )
            )
            parts.append(
                _render_stage(
                    run_dir / stage,
                    media_url=_media_url,
                    detail_url_for=_artifact_link,
                )
            )
        if skipped:
            parts.append(
                "<p class=muted>heavy per-episode working trees (probe "
                "scenes, version stores, suite scratch) are kept on disk "
                "but not rendered: "
                + ", ".join(f"<code>{html.escape(s)}/</code>" for s in skipped)
                + "</p>"
            )
        return _page(name, "".join(parts), nav=nav)

    @app.route("/run/<name>/artifact/<path:relative>")
    def run_artifact(name: str, relative: str) -> str:
        run_dir = _safe_run_dir(name)
        target = (run_dir / relative).resolve()
        if run_dir not in target.parents or not target.is_file():
            abort(404)
        body = _file_block(
            target,
            _render_file(target, media_url=_media_url, query=request.args),
            display_name=str(target.relative_to(run_dir)),
        )
        nav = f"<a href='{url_for('run_page', name=name)}'>run overview</a>"
        return _page(f"{name} · {target.name}", body, nav=nav)

    @app.route("/live")
    def latest_live():
        run_dir = _latest_run(runs_root)
        if run_dir is None:
            return _page(
                "live execution",
                _live_container(
                    "<p>Waiting for a run…</p>", url_for("latest_live_fragment")
                ),
            )
        trace = _latest_trace(run_dir)
        records = (
            [record for record in _iter_jsonl(trace) if isinstance(record, dict)]
            if trace is not None
            else []
        )
        task_name = (
            f"{run_dir.name} / {trace.parent.name}"
            if trace is not None
            else f"{run_dir.name} / waiting for task"
        )
        body = _render_live_execution(records, task_name)
        return _page(
            "live execution",
            _live_container(body, url_for("latest_live_fragment")),
        )

    @app.route("/live/fragment")
    def latest_live_fragment() -> str:
        run_dir = _latest_run(runs_root)
        if run_dir is None:
            return _localize("<p>Waiting for a run…</p>")
        trace = _latest_trace(run_dir)
        if trace is None:
            return _localize(
                _render_live_execution([], f"{run_dir.name} / waiting for task")
            )
        records = [record for record in _iter_jsonl(trace) if isinstance(record, dict)]
        return _localize(
            _render_live_execution(records, f"{run_dir.name} / {trace.parent.name}")
        )

    @app.route("/run/<name>/live")
    def live_run(name: str) -> str:
        run_dir = _safe_run_dir(name)
        trace = _latest_trace(run_dir)
        records = (
            [record for record in _iter_jsonl(trace) if isinstance(record, dict)]
            if trace is not None
            else []
        )
        task_name = trace.parent.name if trace is not None else "task"
        body = _render_live_execution(records, task_name)
        nav = f"<a href='{url_for('run_page', name=name)}'>artifacts</a>"
        return _page(
            f"{name} · live",
            _live_container(body, url_for("live_run_fragment", name=name)),
            nav=nav,
        )

    @app.route("/run/<name>/live/fragment")
    def live_run_fragment(name: str) -> str:
        run_dir = _safe_run_dir(name)
        trace = _latest_trace(run_dir)
        if trace is None:
            return _localize(_render_live_execution([], "task"))
        records = [record for record in _iter_jsonl(trace) if isinstance(record, dict)]
        return _localize(_render_live_execution(records, trace.parent.name))

    @app.route("/library")
    def system_libraries() -> str:
        if library_dir is None or not library_dir.is_dir():
            return _page(
                "system library",
                "<p class=muted>No library directory configured; pass "
                "<code>--library-dir</code> or keep a <code>library/</code> "
                "directory beside the runs root.</p>",
            )
        files = sorted(library_dir.glob("*.json"))
        if not files:
            return _page(
                "system library",
                f"<p class=muted>No libraries in <code>{html.escape(str(library_dir))}</code>.</p>",
            )
        parts = [
            "<p class=explain>The shipped symbol libraries and capability "
            "contracts — the system layer, independent of any run. One tab "
            "per library; the graph shows how predicates, operators and "
            "capabilities connect.</p>",
            _render_chain_anatomy(),
        ]
        buttons = []
        panes = []
        for index, path in enumerate(files):
            active = " active" if index == 0 else ""
            hidden = "" if index == 0 else " hidden"
            buttons.append(
                f"<button class='tab{active}' data-pane='libtab-{index}'>"
                f"{html.escape(path.stem)}</button>"
            )
            panes.append(
                f"<div id='libtab-{index}' class='libpane{hidden}'>"
                f"{_render_symbol_library(_load_json(path))}</div>"
            )
        parts.append(f"<div class=tabs>{''.join(buttons)}</div>")
        parts.extend(panes)
        parts.append(
            "<script>document.querySelectorAll('.tab').forEach(function(b){"
            "b.addEventListener('click',function(){"
            "document.querySelectorAll('.tab').forEach(function(x){x.classList.remove('active');});"
            "document.querySelectorAll('.libpane').forEach(function(p){p.classList.add('hidden');});"
            "b.classList.add('active');"
            "document.getElementById(b.dataset.pane).classList.remove('hidden');"
            "});});</script>"
        )
        return _page("system library", "".join(parts))

    @app.route("/capabilities")
    def capability_catalog() -> str:
        from resym.platform.coraplex_catalog import (
            coraplex_capability_catalog,
        )

        return _page(
            "capability catalog",
            _render_capability_catalog(
                coraplex_capability_catalog(
                    initialize_coraplex_capabilities(
                        _reviewed_contracts(), realization_workspace
                    )
                ),
                _operator_usage_by_capability(library_dir),
                realization_candidates=(
                    ()
                    if realization_workspace is None
                    else realization_workspace.candidates()
                ),
                contract_candidates=(
                    ()
                    if contract_workspace is None
                    else contract_workspace.candidates()
                ),
            ),
        )

    def _reviewed_contracts() -> tuple[CapabilityContract, ...]:
        """
        Contracts the shipped libraries carry plus those the workspace admitted.
        """
        contracts = {
            contract.uid: contract for contract in _contracts_in_libraries(library_dir)
        }
        if contract_workspace is not None:
            for contract in contract_workspace.approved_contracts():
                contracts.setdefault(contract.uid, contract)
        return tuple(contracts[uid] for uid in sorted(contracts))

    @app.post("/capabilities/contracts/<candidate_id>/approve")
    def approve_contract(candidate_id: str):
        if contract_workspace is None:
            abort(404)
        reviewer = str(request.form.get("reviewer") or "").strip()
        if not reviewer:
            abort(400)
        contract_workspace.approve(
            candidate_id,
            reviewer,
            discover_coraplex_capability_contract_drafts(),
            review_note=str(request.form.get("review_note") or "").strip() or None,
        )
        return redirect(url_for("capability_catalog"))

    @app.post("/capabilities/contracts/<candidate_id>/reject")
    def reject_contract(candidate_id: str):
        if contract_workspace is None:
            abort(404)
        reviewer = str(request.form.get("reviewer") or "").strip()
        review_note = str(request.form.get("review_note") or "").strip()
        if not reviewer or not review_note:
            abort(400)
        contract_workspace.reject(candidate_id, reviewer, review_note)
        return redirect(url_for("capability_catalog"))

    @app.post("/capabilities/realizations/<candidate_id>/approve")
    def approve_realization(candidate_id: str):
        if realization_workspace is None:
            abort(404)
        reviewer = str(request.form.get("reviewer") or "").strip()
        if not reviewer:
            abort(400)
        approve_realization_candidate(
            realization_workspace,
            candidate_id,
            reviewer,
            _reviewed_contracts(),
            review_note=str(request.form.get("review_note") or "").strip() or None,
        )
        return redirect(url_for("capability_catalog"))

    @app.post("/capabilities/realizations/<candidate_id>/reject")
    def reject_realization(candidate_id: str):
        if realization_workspace is None:
            abort(404)
        reviewer = str(request.form.get("reviewer") or "").strip()
        review_note = str(request.form.get("review_note") or "").strip()
        if not reviewer or not review_note:
            abort(400)
        realization_workspace.reject(candidate_id, reviewer, review_note)
        return redirect(url_for("capability_catalog"))

    @app.route("/grounding-factories")
    def grounding_factories() -> str:
        if grounding_workspace is None:
            abort(404)
        return _page(
            "grounding factories",
            _render_grounding_factories(
                grounding_workspace, _current_grounding_vocabulary()
            ),
        )

    @app.post("/grounding-factories/<candidate_id>/approve")
    def approve_grounding_factory(candidate_id: str):
        if grounding_workspace is None:
            abort(404)
        reviewer = str(request.form.get("reviewer") or "").strip()
        if not reviewer:
            abort(400)
        grounding_workspace.approve(
            candidate_id,
            reviewer=reviewer,
            vocabulary=_current_grounding_vocabulary(),
            review_note=str(request.form.get("review_note") or "").strip() or None,
        )
        return redirect(url_for("grounding_factories"))

    @app.post("/grounding-factories/<candidate_id>/reject")
    def reject_grounding_factory(candidate_id: str):
        if grounding_workspace is None:
            abort(404)
        reviewer = str(request.form.get("reviewer") or "").strip()
        review_note = str(request.form.get("review_note") or "").strip()
        if not reviewer or not review_note:
            abort(400)
        grounding_workspace.reject(candidate_id, reviewer, review_note)
        return redirect(url_for("grounding_factories"))

    @app.post("/grounding-vocabulary/<path:qualified_name>/approve")
    def approve_grounding_vocabulary(qualified_name: str):
        if grounding_workspace is None:
            abort(404)
        reviewer = str(request.form.get("reviewer") or "").strip()
        if not reviewer:
            abort(400)
        grounding_workspace.approve_vocabulary(
            qualified_name,
            reviewer,
            str(request.form.get("review_note") or "").strip() or None,
        )
        return redirect(url_for("grounding_factories"))

    @app.post("/grounding-vocabulary/<path:qualified_name>/reject")
    def reject_grounding_vocabulary(qualified_name: str):
        if grounding_workspace is None:
            abort(404)
        reviewer = str(request.form.get("reviewer") or "").strip()
        review_note = str(request.form.get("review_note") or "").strip()
        if not reviewer or not review_note:
            abort(400)
        grounding_workspace.reject_vocabulary(qualified_name, reviewer, review_note)
        return redirect(url_for("grounding_factories"))

    @app.route("/run/<name>/libraries")
    def run_libraries(name: str) -> str:
        run_dir = _safe_run_dir(name)
        parts = []
        snapshots = sorted(run_dir.glob("library_*.json"))
        if snapshots:
            parts.append(
                _section(
                    "Library snapshots",
                    "The symbol library this run actually used — usually "
                    "the state it started from. Compare with the system "
                    "layer under 'system library' in the header.",
                )
            )
            for path in snapshots:
                parts.append(
                    f"<div class=file><div class=name>{html.escape(path.name)}</div>"
                    f"{_render_symbol_library(_load_json(path))}</div>"
                )
        stores_html = _render_version_stores(run_dir)
        if stores_html:
            parts.append(stores_html)
        if not parts:
            parts = [
                "<p class=muted>This run recorded no library snapshots "
                "or version stores.</p>"
            ]
        nav = f"<a href='{url_for('run_page', name=name)}'>artifacts</a>"
        return _page(f"{name} · libraries", "".join(parts), nav=nav)

    @app.route("/healthz")
    def healthz() -> str:
        return "ok"

    return app


def _render_pipeline_map(present_stages: list[str]) -> str:
    """
    The run's place in the pipeline: which stages ran, what each does, and what an
    absent stage means.

    Present stages link to their section.
    """
    steps = []
    for stage, label, does, absent_meaning in PIPELINE_STEPS:
        if stage in present_stages:
            steps.append(
                f"<a class=pipestep href='#{html.escape(stage)}'>"
                f"<b>{html.escape(label)}</b>{html.escape(does)}</a>"
            )
        elif absent_meaning == "always":
            steps.append(
                f"<div class='pipestep absent'><b>{html.escape(label)}</b>"
                f"{html.escape(does)}</div>"
            )
        elif absent_meaning is not None:
            steps.append(
                f"<div class='pipestep absent'><b>{html.escape(label)}</b>"
                f"not present: {html.escape(absent_meaning)}</div>"
            )
    return (
        "<div class=pipemap>"
        + "<span class=pipearrow>→</span>".join(steps)
        + "</div><p class=explain>The reSym pipeline for this run — grayed "
        "stages did not occur. Click a stage to jump to its artifacts.</p>"
    )


def _file_block(path: Path, rendered: str, display_name: str | None = None) -> str:
    """
    A file heading with its one-line identity, above its rendering.
    """
    name = display_name if display_name is not None else path.name
    whatis = FILE_EXPLAINERS.get(path.name)
    whatis_html = f" <span class=whatis>{html.escape(whatis)}</span>" if whatis else ""
    return (
        f"<div class=file><div class=name><code>{html.escape(name)}</code>"
        f"{whatis_html}</div>{rendered}</div>"
    )


def _section(title: str, explainer: str = "", anchor: str = "") -> str:
    """
    A stage heading with an optional one-line plain-language explainer.
    """
    anchor_attr = f" id='{html.escape(anchor)}'" if anchor else ""
    heading = f"<h2 class=stage{anchor_attr}>{html.escape(title)}</h2>"
    if explainer:
        heading += f"<p class=explain>{html.escape(explainer)}</p>"
    return heading


# -- live execution ----------------------------------------------------------


def _live_container(body: str, fragment_url: str) -> str:
    """
    Wrap a live view and refresh only its contents, without page flicker.

    The fragment is only swapped in when it actually changed, and folded sections the
    reader opened (all ``details`` carry stable ids) plus the scroll position survive
    the swap.
    """
    escaped_url = json.dumps(fragment_url)
    return (
        f"<div id=live-root>{body}</div>"
        "<script>"
        "const liveRoot=document.getElementById('live-root');"
        f"const liveUrl={escaped_url};"
        "let lastFragment=liveRoot.innerHTML;"
        "async function refreshLive(){try{"
        "const response=await fetch(liveUrl,{cache:'no-store'});"
        "if(!response.ok)return;"
        "const fragment=await response.text();"
        "if(fragment===lastFragment)return;"
        "const opened=new Set(Array.from(liveRoot.querySelectorAll('details[open]'))"
        ".map(node=>node.id).filter(Boolean));"
        "const x=window.scrollX,y=window.scrollY;"
        "liveRoot.innerHTML=fragment;"
        # Only after the swap landed: an exception above must not convince
        # later polls that this fragment is already on screen.
        "lastFragment=fragment;"
        "for(const id of opened){const node=document.getElementById(id);"
        "if(node)node.open=true;}"
        "window.scrollTo(x,y);"
        "}catch(error){console.warn('live refresh failed',error);}}"
        "setInterval(refreshLive,500);"
        "</script>"
    )


def _latest_run(runs_root: Path):
    if not runs_root.is_dir():
        return None
    runs = [path for path in runs_root.iterdir() if path.is_dir()]
    return max(runs, key=lambda path: path.name) if runs else None


def _latest_trace(run_dir: Path):
    traces = list((run_dir / "C_solve").glob("*/trace.jsonl"))
    if not traces:
        return None
    return max(traces, key=lambda path: path.stat().st_mtime_ns)


def _render_live_execution(
    records: list[dict], task_name: str, compact: bool = False
) -> str:
    state = _reduce_pipeline_events(records)
    goal = " · ".join(item.get("display", "") for item in state["goal"])
    status_class = (
        "ok"
        if state["status_kind"] == "ok"
        else ("bad" if state["status_kind"] == "bad" else "")
    )
    status = html.escape(state["status"])
    status_html = (
        f"<span class='badge {status_class} live-status'>{status}</span>"
        if status_class
        else f"<span class='live-status muted'>{status}</span>"
    )
    round_text = f"round {state['round']}" if state["round"] else "waiting"
    grounding = state["grounding"]
    metrics = ""
    if grounding:
        metrics = "".join(
            f"<span class='metric{css}'><b>{label}</b>: {html.escape(str(grounding.get(key, 0)))}</span>"
            for key, label, css in (
                ("evaluations", "predicate evaluations", ""),
                ("true", "TRUE", " t-true"),
                ("false", "FALSE", " t-false"),
            )
        )

    plan_rows = []
    for index, action in enumerate(state["plan"]):
        if index in state["completed"]:
            mark, label_class = "✓", "truth-true"
        elif index in state["failed"]:
            mark, label_class = "✕", "truth-false"
        elif index == state["active"]:
            mark, label_class = "▶", "active"
        else:
            mark, label_class = "○", "muted"
        detail = state["action_details"].get(index, {})
        summary_parts = []
        request = detail.get("request") or {}
        capability = request.get("capability_ref") or {}
        if capability:
            summary_parts.append(str(capability.get("uid", "")))
        if detail.get("coraplex_action"):
            summary_parts.append(str(detail["coraplex_action"]))
        platform = detail.get(PipelineEvent.PLATFORM_RESULT) or {}
        if platform:
            summary_parts.append(str(platform.get("status", "")))
        summary = " → ".join(part for part in summary_parts if part)
        summary_html = (
            f"<div class=action-summary>{html.escape(summary)}</div>" if summary else ""
        )
        plan_rows.append(
            f"<li><span class='mark {label_class}'>{mark}</span>"
            f"<div><code>{html.escape(str(action.get('display', '')))}</code>"
            f"{summary_html}{_step_details(index, detail)}</div></li>"
        )
    plan_html = "".join(plan_rows) or "<li class=muted>waiting for planner</li>"

    current = state["current_action"]
    grounded = html.escape(str((current or {}).get("display", "waiting")))
    request = state["request"] or {}
    binding_rows = []
    for role, binding in _binding_pairs(request.get("role_bindings")):
        value = binding.get("value", "")
        rendered = (
            f"?{value}" if _binding_is_parameter(binding.get("source")) else value
        )
        binding_rows.append(
            f"<span class=chip>{html.escape(str(role))} ← {html.escape(str(rendered))}</span>"
        )
    capability = request.get("capability_ref") or {}
    capability_text = "waiting"
    if capability:
        capability_text = (
            f"{capability.get('uid', '')}@{capability.get('version', '1')}"
        )
    argument_rows = "".join(
        f"<span class=chip>{html.escape(str(role))}={html.escape(str(value))}</span>"
        for role, value in (request.get("arguments") or {}).items()
    )
    coraplex_action = html.escape(str(state["coraplex_action"] or "waiting"))
    realization_name = html.escape(str(state["realization"] or "waiting"))
    platform = state[PipelineEvent.PLATFORM_RESULT] or {}
    platform_text = "waiting"
    if platform:
        platform_text = f"{platform.get('status', '')}: {platform.get('code', '')}"

    check_rows = []
    for check in state["checks"]:
        truth = str(check.get("truth", "not evaluated"))
        required = str(check.get("required", "not evaluated"))
        satisfied = check.get("satisfied") is True
        symbol = "✓" if satisfied else "✕"
        literal = (check.get("literal") or {}).get("display", "")
        reason = check.get("reason")
        detail = f"{literal} · expected {required.upper()} · observed {truth.upper()}"
        if reason:
            detail += f" ({reason})"
        check_rows.append(
            f"<div class=check><span>{html.escape(check.get('kind', 'check'))}</span>"
            f"<span>{html.escape(detail)}</span>"
            f"<b class='{'check-pass' if satisfied else 'check-fail'}'>"
            f"{symbol} {'PASS' if satisfied else 'FAIL'}</b></div>"
        )
    checks_html = "".join(check_rows) or "<span class=muted>no checks yet</span>"

    timeline = _render_event_rows(records[-12:])
    if compact:
        title = ""
        banner = (
            "<p class=explain>Execution replay: this task's recorded event "
            "stream, reduced to the same panels as the live execution page "
            "— shown in its final state.</p>"
        )
        head_label = "Execution replay"
    else:
        title = "<h2 style='margin:0 0 14px'>Live symbolic execution</h2>"
        banner = _liveness_banner(state, records)
        head_label = task_name
    return (
        title + banner + f"<div class=live-head><div><b>{html.escape(head_label)}</b>"
        f"<div class=muted>goal: {html.escape(goal or 'waiting')} · {round_text}</div>"
        f"<div>{metrics}</div></div>{status_html}</div>"
        + _render_phase_strip(state["phase"])
        + "<div class=live-grid>"
        "<section class=card><h3>Grounded plan</h3>"
        "<p class=explain>✓ done · ▶ active · ○ pending. Open a step record "
        "for its full platform chain and checks.</p>"
        f"<ol class=live-plan>{plan_html}</ol></section>"
        "<section>"
        "<div class=card><h3>Selected action</h3>"
        "<p class=explain>How the active plan step becomes platform motion, "
        "layer by layer.</p><div class=flow>"
        f"<div><span class=muted>Grounded Operator</span><br><code>{grounded}</code></div>"
        f"<div class=arrow>↓ OperatorExecutionBinding</div><div>{''.join(binding_rows) or '<span class=muted>waiting</span>'}</div>"
        f"<div class=arrow>↓ CapabilityContract</div><div><code>{html.escape(capability_text)}</code></div>"
        f"<div class=arrow>↓ ExecutionRequest</div><div>{argument_rows or '<span class=muted>waiting</span>'}</div>"
        f"<div class=arrow>↓ PlatformSkillRealization</div><div><code>{realization_name}</code></div>"
        f"<div class=arrow>↓ Coraplex Action</div><div><code>{coraplex_action}</code></div>"
        f"<div class=arrow>↓ PlatformExecutionResult</div><div><code>{html.escape(platform_text)}</code></div>"
        "</div></div>"
        "<div class=card><h3>Checks</h3>"
        "<p class=explain>Truth recomputed around the active action: what "
        "each precondition, effect and goal literal was expected to be, "
        "and what the world actually says.</p>"
        f"{checks_html}</div>"
        "</section></div>"
        "<h2 class=stage>Recent events</h2>"
        "<p class=explain>The raw tail of the trace this whole view is "
        "rendered from (trace.jsonl) — the last 12 entries, for "
        "debugging.</p>"
        f"<div class='card timeline'>{timeline}</div>"
    )


def _step_details(index: int, detail: dict) -> str:
    """
    A folded per-step record: the full operator→platform chain and every check this plan
    step went through, available once the step has run.
    """
    request = detail.get("request") or {}
    platform = detail.get(PipelineEvent.PLATFORM_RESULT) or {}
    checks = detail.get("checks") or []
    if not (request or detail.get("coraplex_action") or platform or checks):
        return ""
    lines = []
    bindings = _binding_pairs(request.get("role_bindings"))
    if bindings:
        chips = "".join(
            f"<span class=chip>{html.escape(str(role))} ← "
            f"{html.escape(('?' + str(binding.get('value', ''))) if _binding_is_parameter(binding.get('source')) else str(binding.get('value', '')))}</span>"
            for role, binding in bindings
        )
        lines.append(f"<div>{chips}</div>")
    capability = request.get("capability_ref") or {}
    if capability:
        lines.append(
            f"<div><span class=muted>capability</span> "
            f"<code>{html.escape(str(capability.get('uid', '')))}@{html.escape(str(capability.get('version', '1')))}</code></div>"
        )
    if detail.get("coraplex_action"):
        lines.append(
            f"<div><span class=muted>coraplex</span> "
            f"<code>{html.escape(str(detail['coraplex_action']))}</code></div>"
        )
    if detail.get("realization"):
        lines.append(
            f"<div><span class=muted>realization</span> "
            f"<code>{html.escape(str(detail['realization']))}</code></div>"
        )
    if platform:
        lines.append(
            f"<div><span class=muted>platform</span> "
            f"<code>{html.escape(str(platform.get('status', '')))}: {html.escape(str(platform.get('code', '')))}</code></div>"
        )
    for check in checks:
        satisfied = check.get("satisfied") is True
        literal = (check.get("literal") or {}).get("display", "")
        truth = str(check.get("truth", "not-recorded")).upper()
        reason = check.get("reason")
        text = f"{check.get('kind', 'check')}: {literal} — {truth}"
        if reason:
            text += f" ({reason})"
        lines.append(
            f"<div class='{'check-pass' if satisfied else 'check-fail'}'>"
            f"{'✓' if satisfied else '✕'} {html.escape(text)}</div>"
        )
    return (
        f"<details id='step-{index}'><summary>step record</summary>"
        f"<div class=stepbody>{''.join(lines)}</div></details>"
    )


STALE_TRACE_SECONDS = 30
"""
A non-terminal trace with no event for this long is flagged as stalled.
"""


def _liveness_banner(state: dict, records: list[dict]) -> str:
    """
    Say plainly whether this view is live, finished, or stalled — the page follows the
    newest trace file and cannot see processes.
    """
    if not records:
        return ""
    last_at = str(records[-1].get("at", ""))
    if state["status_kind"] == "ok":
        return (
            "<div class=card><span class='badge ok'>finished</span> "
            f"<span class=muted>This task completed at {html.escape(last_at)} — "
            "shown here as its final state for review. Start a new task and "
            "this page follows it automatically.</span></div>"
        )
    if state["status_kind"] == "bad":
        return (
            "<div class=card><span class='badge bad'>ended</span> "
            f"<span class=muted>This task stopped at {html.escape(last_at)} — "
            "shown here as its final state for review.</span></div>"
        )
    try:
        age = (datetime.now() - datetime.fromisoformat(last_at)).total_seconds()
    except ValueError:
        return ""
    if age > STALE_TRACE_SECONDS:
        return (
            "<div class=card><span class=badge style='background:#fef3c7;color:#92400e'>stalled?</span> "
            f"<span class=muted>No new events since {html.escape(last_at)} and the "
            "task never reported an end — it may have been interrupted, or a "
            "long computation (e.g. grounding) is still running.</span></div>"
        )
    return ""


def _reduce_pipeline_events(records: list[dict]) -> dict:
    state = {
        "goal": [],
        "round": None,
        "grounding": None,
        "plan": [],
        "active": None,
        "completed": set(),
        "failed": set(),
        "current_action": None,
        "request": None,
        "realization": None,
        "coraplex_action": None,
        PipelineEvent.PLATFORM_RESULT: None,
        "checks": [],
        "action_details": {},
        "phase": "grounding",
        "status": "waiting",
        "status_kind": "",
    }
    for record in records:
        event = record.get("event")
        if event == PipelineEvent.TASK_STARTED:
            state["goal"] = record.get("goal") or []
            state["status"] = "grounding"
            state["phase"] = "grounding"
        elif event == PipelineEvent.ROUND_STARTED:
            state["round"] = record.get("round")
            state["active"] = None
            state["current_action"] = None
            state["request"] = None
            state["realization"] = None
            state["coraplex_action"] = None
            state[PipelineEvent.PLATFORM_RESULT] = None
            state["checks"] = []
        elif event == PipelineEvent.GROUNDING_COMPLETED:
            state["grounding"] = record
            state["status"] = "planning"
            state["phase"] = "plan"
        elif event == PipelineEvent.GROUNDING_FAILED:
            state["status"] = "grounding failed"
            state["status_kind"] = "bad"
            state["phase"] = "grounding"
        elif event == PipelineEvent.PLANNING_STARTED:
            state["status"] = "planning"
        elif event == PipelineEvent.PLAN_GENERATED:
            state["plan"] = record.get("actions") or []
            state["action_details"] = {
                index: {
                    "action": action,
                    "request": None,
                    "realization": None,
                    "coraplex_action": None,
                    PipelineEvent.PLATFORM_RESULT: None,
                    "checks": [],
                }
                for index, action in enumerate(state["plan"])
            }
            state["completed"] = set()
            state["failed"] = set()
            state["active"] = None
            state["current_action"] = None
            state["status"] = "plan ready"
            state["phase"] = "plan"
        elif event == PipelineEvent.PLAN_PREVIEW_STARTED:
            seconds = record.get("seconds")
            state["status"] = (
                f"reviewing plan ({seconds:g}s)"
                if isinstance(seconds, (int, float))
                else "reviewing plan"
            )
            state["phase"] = "plan"
        elif event == PipelineEvent.PLAN_PREVIEW_COMPLETED:
            state["status"] = "execution starting"
            state["phase"] = "execution"
        elif event == PipelineEvent.ACTION_STARTED:
            state["active"] = record.get("action_index")
            state["current_action"] = record.get("action")
            state["request"] = None
            state["realization"] = None
            state["coraplex_action"] = None
            state[PipelineEvent.PLATFORM_RESULT] = None
            state["checks"] = []
            state["status"] = "checking preconditions"
            state["phase"] = "execution"
        elif event in (
            PipelineEvent.PRECONDITION_CHECKED,
            PipelineEvent.EFFECT_CHECKED,
        ):
            check = dict(record)
            check["kind"] = "precondition" if event.startswith("pre") else "effect"
            state["checks"].append(check)
            detail = state["action_details"].get(record.get("action_index"))
            if detail is not None:
                detail["checks"].append(check)
            if event == PipelineEvent.EFFECT_CHECKED:
                state["phase"] = "verification"
        elif event == PipelineEvent.EXECUTION_REQUEST_CREATED:
            state["request"] = record
            detail = state["action_details"].get(record.get("action_index"))
            if detail is not None:
                detail["request"] = record
            state["status"] = "request ready"
        elif event == PipelineEvent.PLATFORM_EXECUTION_STARTED:
            state["realization"] = record.get("realization")
            detail = state["action_details"].get(record.get("action_index"))
            if detail is not None:
                detail["realization"] = record.get("realization")
            state["status"] = "Coraplex executing"
        elif event == PipelineEvent.CORAPLEX_ACTION_SELECTED:
            state["coraplex_action"] = record.get("action")
            detail = state["action_details"].get(state["active"])
            if detail is not None:
                detail["coraplex_action"] = record.get("action")
        elif event == PipelineEvent.PLATFORM_RESULT:
            state[PipelineEvent.PLATFORM_RESULT] = record
            detail = state["action_details"].get(record.get("action_index"))
            if detail is not None:
                detail[PipelineEvent.PLATFORM_RESULT] = record
            state["status"] = "verifying effects"
            state["phase"] = "verification"
        elif event == PipelineEvent.ACTION_COMPLETED:
            state["completed"].add(record.get("action_index"))
            state["status"] = "action verified"
        elif event == PipelineEvent.EXECUTION_STOPPED:
            state["failed"].add(record.get("action_index"))
            state["status"] = str(record.get("violation") or "execution stopped")
            state["status_kind"] = "bad"
        elif event == PipelineEvent.GOAL_LITERAL_CHECKED:
            check = dict(record)
            check["kind"] = "goal"
            state["checks"].append(check)
            state["phase"] = "verification"
        elif event == PipelineEvent.REPLANNING_STARTED:
            state["status"] = "replanning"
            state["status_kind"] = ""
        elif event == PipelineEvent.TASK_SUCCEEDED:
            state["status"] = "succeeded"
            state["status_kind"] = "ok"
            state["phase"] = "complete"
        elif event in (
            "task_failed",
            PipelineEvent.REPLANNING_LIMIT_EXCEEDED,
            PipelineEvent.REPEATED_PLAN_REFUSED,
        ):
            state["status"] = str(record.get("error") or event.replace("_", " "))
            state["status_kind"] = "bad"
    return state


def _render_phase_strip(active_phase: str) -> str:
    phases = (
        ("grounding", "1 · Model evaluation"),
        ("plan", "2 · Plan ready"),
        ("execution", "3 · Executing"),
        ("verification", "4 · Verifying"),
    )
    order = {name: index for index, (name, _) in enumerate(phases)}
    active_index = order.get(active_phase, len(phases))
    items = []
    for index, (name, label) in enumerate(phases):
        css_class = "done" if index < active_index else ""
        if name == active_phase:
            css_class = "active"
        if active_phase == "complete":
            css_class = "done"
        items.append(f"<div class='phase-step {css_class}'>{html.escape(label)}</div>")
    return f"<div class=phase-strip>{''.join(items)}</div>"


def _render_event_rows(records: list[dict]) -> str:
    rows = []
    for record in records:
        when = str(record.get("at", ""))
        name = str(record.get("event", ""))
        sequence = record.get("seq")
        label = f"#{sequence}" if sequence is not None else when
        extra = " · ".join(
            f"{html.escape(str(key))}: {html.escape(str(value))}"
            for key, value in record.items()
            if key not in {"at", "seq", "event"}
            and value is not None
            and not isinstance(value, (dict, list))
        )
        rows.append(
            f"<div class=evt><span class=t>{html.escape(label)}</span>"
            f"<span class=n>{html.escape(name)}</span>"
            f"<span class=muted>{extra}</span></div>"
        )
    return "".join(rows) or "<span class=muted>waiting for events</span>"


# -- per-run header -----------------------------------------------------------


def _format_seconds(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _stage_timeline(run_dir: Path, meta: dict) -> str:
    """
    Where the run's wall-clock went: one proportional segment per stage, from the run
    start to each stage's last logged event.
    """
    try:
        started = datetime.fromisoformat(str(meta["started_at"]))
    except (KeyError, TypeError, ValueError):
        return ""
    boundaries = []
    for stage in STAGE_TITLES:
        events = run_dir / stage / "events.jsonl"
        if not events.is_file():
            continue
        last = None
        for record in _iter_jsonl(events):
            if isinstance(record, dict) and record.get("at"):
                last = str(record["at"])
        if not last:
            continue
        try:
            boundaries.append((stage, datetime.fromisoformat(last)))
        except ValueError:
            continue
    boundaries.sort(key=lambda pair: pair[1])
    segments = []
    previous = started
    for stage, end in boundaries:
        seconds = (end - previous).total_seconds()
        if seconds > 0:
            segments.append((stage, seconds))
            previous = end
    if len(segments) < 2:
        return ""
    parts = "".join(
        f"<div class=tl-seg style='flex:{max(seconds, 1):.0f}' data-tip=\""
        + html.escape(
            "from the run start (or the previous stage's last event) to "
            "this stage's last logged event",
            quote=True,
        )
        + f'"><b>{html.escape(STAGE_TITLES.get(stage, stage))}</b> '
        f"{html.escape(_format_seconds(seconds))}</div>"
        for stage, seconds in segments
    )
    return (
        "<div class=timeline-wrap><p class=explain style='margin:10px 0 4px'>"
        "where the time went</p>"
        f"<div class=timeline>{parts}</div></div>"
    )


def _task_verdict_card(run_dir: Path, meta: dict) -> str:
    """
    The headline card of a task run: did the task succeed, with the plan it ran — the
    answer first, before any artifact listing.
    """
    summary = meta.get("summary") or {}
    succeeded = summary.get("succeeded")
    if succeeded is None:
        return ""
    tasks = []
    solve = run_dir / "C_solve"
    if solve.is_dir():
        for task_dir in sorted(p for p in solve.iterdir() if p.is_dir()):
            result = _load_json(task_dir / "result.json") or {}
            plan = result.get("plan") or []
            steps = " &rarr; ".join(
                f"<code>{html.escape(step.strip('()').split()[0])}</code>"
                for step in plan
                if isinstance(step, str) and step.strip("()").split()
            )
            executed = (result.get("execution") or {}).get("succeeded")
            state = ""
            if executed is True:
                state = "executed &amp; verified"
            elif executed is False:
                state = "execution violated"
            tasks.append(
                f"<div><b>{html.escape(task_dir.name)}</b>"
                + (f" — plan: {steps}" if steps else "")
                + (f" <span class=muted>&middot; {state}</span>" if state else "")
                + "</div>"
            )
    badge = (
        "<span class='badge ok'>&#10003; succeeded</span>"
        if succeeded
        else "<span class='badge bad'>&#10007; failed</span>"
    )
    duration = _duration_text(meta)
    return (
        "<div class='card verdict'>"
        + badge
        + "<b>task outcome</b>"
        + (
            f" <span class=muted>&middot; {html.escape(duration)}</span>"
            if duration
            else ""
        )
        + (f"<div style='margin-top:8px'>{''.join(tasks)}</div>" if tasks else "")
        + "</div>"
    )


def _render_run_meta(run_dir: Path) -> str:
    meta = _load_json(run_dir / "run.json") or {}
    if not meta:
        return ""
    duration = _duration_text(meta)
    kv = {
        "label": meta.get("label"),
        "started": meta.get("started_at"),
        "ended": meta.get("ended_at"),
        "duration": duration,
        "scene": meta.get("scene"),
    }
    grid = "".join(
        f"<div class=k>{html.escape(k)}</div><div>{html.escape(str(v))}</div>"
        for k, v in kv.items()
        if v is not None
    )
    summary = meta.get("summary")
    summary_html = ""
    if summary:
        summary_html = "<h3 style='margin:10px 0 4px'>summary</h3>" + _summary_chips(
            summary
        )
    timeline = _stage_timeline(run_dir, meta)
    return (
        f"<div class=card><div class=kv>{grid}</div>{summary_html}" f"{timeline}</div>"
    )


def _summary_chips(summary: dict) -> str:
    chips = []
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            continue
        chips.append(
            f"<span class=metric><b>{html.escape(str(key))}</b>: {html.escape(str(value))}</span>"
        )
    plan = summary.get("retry_plan")
    if isinstance(plan, list) and plan:
        chips.append(_plan_html(plan))
    return "".join(chips) or "<span class=muted>(see run.json)</span>"


# -- stage / file dispatch ----------------------------------------------------


_TASK_FILE_ORDER = {
    "goal.json": 0,
    "domain.pddl": 1,
    "problem.pddl": 2,
    "result.json": 3,
    "trace.jsonl": 4,
}
"""
Task artifacts in pipeline order (goal → PDDL → outcome → trace), not alphabetically.
"""

_RESULT_FILE_ORDER: dict[str, int] = {}
"""
Optional ordering of recognized run-level artifacts.
"""


def _render_stage(stage_dir: Path, media_url=None, detail_url_for=None) -> str:
    files = sorted(
        (p for p in stage_dir.rglob("*") if p.is_file()),
        key=lambda p: (
            str(p.parent),
            _TASK_FILE_ORDER.get(p.name, len(_TASK_FILE_ORDER)),
            p.name,
        ),
    )
    if not files:
        return "<p class=muted>(empty)</p>"
    blocks = []
    for path in files:
        rel = path.relative_to(stage_dir)
        blocks.append(
            _file_block(
                path,
                _render_file(
                    path,
                    media_url=media_url,
                    detail_url=(
                        detail_url_for(path) if detail_url_for is not None else None
                    ),
                ),
                display_name=str(rel),
            )
        )
    return "".join(blocks)


_INLINE_BYTE_LIMIT = 200 * 1024
"""
Raw content larger than this is summarized on the run overview and rendered in full only
on its own artifact page.
"""


def _too_big_stub(path: Path, detail_url: str, what: str) -> str:
    size_mib = path.stat().st_size / (1024 * 1024)
    return (
        "<div class=card>"
        f"<span class=metric><b>size</b>: {size_mib:.1f} MiB</span> "
        f"<span class=muted>{html.escape(what)}</span>"
        f"<div><a class=openlink href='{html.escape(detail_url)}'>"
        "open it on its own page &rarr;</a></div></div>"
    )


def _render_file(path: Path, media_url=None, detail_url=None, query=None) -> str:
    name, suffix = path.name, path.suffix.lower()
    if name == "universe.json":
        return _render_universe(_load_json(path))
    if name == "result.json":
        return _render_result(_load_json(path))
    if name == "goal.json":
        return _render_goal(_load_json(path))
    if name.startswith("library_") and suffix == ".json":
        return _render_symbol_library(_load_json(path))
    if name == "provenance.json":
        return _render_provenance(_load_json(path))
    if suffix == ".jsonl":
        if name == "trace.jsonl":
            records = [
                record for record in _iter_jsonl(path) if isinstance(record, dict)
            ]
            return _render_live_execution(records, path.parent.name, compact=True)
        if name == "llm_transcript.jsonl":
            return _render_transcript(path, detail_url=detail_url, query=query)
        if detail_url and path.stat().st_size > _INLINE_BYTE_LIMIT:
            return _too_big_stub(path, detail_url, "event log")
        return _render_events(path)
    if suffix == ".pddl":
        return _render_pddl(path)
    if suffix == ".json":
        if detail_url and path.stat().st_size > _INLINE_BYTE_LIMIT:
            return _too_big_stub(path, detail_url, "JSON document")
        return _collapsible_json(_load_json(path))
    if suffix in {".mp4", ".webm"}:
        size_mib = path.stat().st_size / (1024 * 1024)
        source = media_url(path) if media_url is not None else None
        player = (
            f"<video class=recording controls preload=metadata "
            f"src='{html.escape(source)}'></video>"
            if source
            else f"<div><code>{html.escape(str(path))}</code></div>"
        )
        return (
            "<div class=card><b>video recording</b> "
            f"<span class=muted>{size_mib:.1f} MiB</span>{player}</div>"
        )
    if detail_url and path.stat().st_size > _INLINE_BYTE_LIMIT:
        return _too_big_stub(path, detail_url, "text file")
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return "<p class=muted>(empty file)</p>"
    if name.endswith("_report.txt"):
        return (
            "<div class=card><p class=explain>Plain-text report generated by "
            "the run.</p>"
            "<details><summary>show the text report</summary>"
            f"<pre class=wrap>{html.escape(text)}</pre></details></div>"
        )
    return f"<pre class=wrap>{html.escape(text)}</pre>"


# -- PDDL ---------------------------------------------------------------------


def _parse_sexp(text: str):
    """
    One tolerant s-expression read (PDDL files are s-expressions).
    """
    stripped = "\n".join(line.split(";", 1)[0] for line in text.splitlines())
    tokens = stripped.replace("(", " ( ").replace(")", " ) ").split()

    def read(index: int):
        if tokens[index] == "(":
            node = []
            index += 1
            while tokens[index] != ")":
                child, index = read(index)
                node.append(child)
            return node, index + 1
        return tokens[index], index + 1

    node, _ = read(0)
    return node


def _sexp_text(node) -> str:
    if isinstance(node, list):
        return "(" + " ".join(_sexp_text(child) for child in node) + ")"
    return str(node)


def _pddl_literal_chips(expression) -> str:
    """
    A precondition/effect expression flattened to literal chips.
    """
    if not isinstance(expression, list):
        return f"<span class=chip>{html.escape(str(expression))}</span>"
    literals = expression[1:] if expression and expression[0] == "and" else [expression]
    return (
        "".join(
            f"<span class=chip>{html.escape(_sexp_text(literal))}</span>"
            for literal in literals
        )
        or "<span class=muted>—</span>"
    )


def _render_pddl(path: Path) -> str:
    """
    Domain and problem files summarized structurally; the raw text stays one fold away.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    raw = (
        f"<details><summary>show raw PDDL</summary>"
        f"<pre class=wrap>{html.escape(text)}</pre></details>"
    )
    try:
        tree = _parse_sexp(text)
        kind = tree[1][0] if isinstance(tree[1], list) else None
        if kind == "domain":
            body = _pddl_domain_summary(tree)
        elif kind == "problem":
            body = _pddl_problem_summary(tree)
        else:
            raise ValueError(kind)
    except (IndexError, ValueError, TypeError):
        return f"<pre class=wrap>{html.escape(text)}</pre>"
    return f"<div class=card>{body}{raw}</div>"


def _pddl_domain_summary(tree) -> str:
    predicates = []
    actions = []
    for section in tree[2:]:
        if not isinstance(section, list) or not section:
            continue
        if section[0] == ":predicates":
            predicates = [
                _sexp_text(entry) for entry in section[1:] if isinstance(entry, list)
            ]
        elif section[0] == ":action":
            fields = {"name": section[1]}
            for key_index in range(2, len(section) - 1, 2):
                fields[str(section[key_index])] = section[key_index + 1]
            actions.append(fields)
    domain_name = tree[1][1] if len(tree[1]) > 1 else "?"
    predicate_chips = "".join(
        f"<span class=chip>{html.escape(p)}</span>" for p in predicates
    )
    action_cards = "".join(
        "<div class='card action-card'>"
        f"<div class=lib-head><b><code>({html.escape(str(a['name']))} "
        f"{html.escape(' '.join(str(p) for p in a.get(':parameters', [])))})</code></b></div>"
        f"<div class=lib-row><span class=k>precondition</span>"
        f"<span>{_pddl_literal_chips(a.get(':precondition'))}</span></div>"
        f"<div class=lib-row><span class=k>effect</span>"
        f"<span>{_pddl_literal_chips(a.get(':effect'))}</span></div>"
        "</div>"
        for a in actions
    )
    return (
        f"<div class=lib-head><b>domain <code>{html.escape(str(domain_name))}</code></b></div>"
        "<p class=explain>What the planner was given: the predicate "
        "vocabulary and the projected actions. The cram-type-… predicates "
        "are static type atoms standing in for the CRAM class hierarchy.</p>"
        f"<div class=pddl-sub>Predicates ({len(predicates)})</div>"
        f"<div>{predicate_chips}</div>"
        f"<div class=pddl-sub>Actions ({len(actions)})</div>"
        f"{action_cards}"
    )


def _pddl_problem_summary(tree) -> str:
    objects = []
    init = []
    goal = None
    for section in tree[2:]:
        if not isinstance(section, list) or not section:
            continue
        if section[0] == ":objects":
            objects = [str(o) for o in section[1:]]
        elif section[0] == ":init":
            init = [entry for entry in section[1:] if isinstance(entry, list)]
        elif section[0] == ":goal":
            goal = section[1] if len(section) > 1 else None
    atom_groups: dict[str, list] = {}
    for atom in init:
        atom_groups.setdefault(str(atom[0]), []).append(atom)
    group_rows = "".join(
        f"<details><summary><code>{html.escape(predicate)}</code> × {len(atoms)}"
        + (
            " <span class=muted>(static type atoms)</span>"
            if predicate.startswith("cram-type-")
            else ""
        )
        + "</summary>"
        + "".join(
            f"<span class=chip>{html.escape(_sexp_text(atom))}</span>" for atom in atoms
        )
        + "</details>"
        for predicate, atoms in sorted(
            atom_groups.items(), key=lambda item: (-len(item[1]), item[0])
        )
    )
    problem_name = tree[1][1] if len(tree[1]) > 1 else "?"
    return (
        f"<div class=lib-head><b>problem <code>{html.escape(str(problem_name))}</code></b>"
        f"<span class=metric><b>objects</b>: {len(objects)}</span>"
        f"<span class=metric><b>init atoms</b>: {len(init)}</span></div>"
        "<p class=explain>This scene, written down for the planner: the "
        "object names, every grounded atom that held (grouped by "
        "predicate), and the goal.</p>"
        f"<div class=pddl-sub>Goal</div>"
        f"<div>{_pddl_literal_chips(goal)}</div>"
        f"<div class=pddl-sub>Init atoms by predicate</div>"
        f"{group_rows}"
    )


# -- purpose-built renderers --------------------------------------------------


def _render_universe(data) -> str:
    if not data:
        return "<div class='card muted'>(unreadable)</div>"
    objects = data.get("objects", [])
    by_type: dict[str, list[dict]] = {}
    for entry in objects:
        by_type.setdefault(str(entry.get("type", "?")), []).append(entry)
    groups = []
    for type_ref, members in sorted(
        by_type.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        instance_chips = "".join(
            f"<span class=chip title='{html.escape(str(m.get('body', '')))}'>"
            f"{html.escape(str(m.get('name', '')))}</span>"
            for m in sorted(members, key=lambda m: str(m.get("name", "")))
        )
        groups.append(
            f"<details><summary>{_short_type(type_ref)} × {len(members)}"
            f"</summary>{instance_chips}</details>"
        )
    total = data.get("object_count", len(objects))
    return (
        f"<div class=card><b>{total} objects</b> in {len(by_type)} types"
        "<p class=explain>Grouped by type — open a type for its instances "
        "(hover an instance for the world body it denotes).</p>"
        f"{''.join(groups)}</div>"
    )


def _render_result(data) -> str:
    if not data:
        return "<div class='card muted'>(unreadable)</div>"
    if data.get("error"):
        return f"<div class='card'><div class=badbox>&#10007; {html.escape(str(data['error']))}</div></div>"
    execution = data.get("execution") or {}
    ok = execution.get("succeeded")
    badge = ""
    if ok is not None:
        badge = f"<span class='badge {'ok' if ok else 'bad'}'>{'executed' if ok else 'violated'}</span>"
    plan = data.get("plan") or []
    plan_html = _plan_html(plan) if plan else "<p class=muted>(no plan)</p>"
    metrics = "".join(
        f"<span class=metric title='{html.escape(hint)}'><b>{label}</b>: "
        f"{html.escape(str(data[key]))}</span>"
        for key, label, hint in (
            (
                "evaluation_count",
                "predicate evaluations",
                "how many predicate truth procedures grounding recomputed "
                "on the world model",
            ),
            ("grounding_seconds", "grounding seconds", "time spent recomputing truth"),
            ("planning_seconds", "planning seconds", "Fast Downward search time"),
            (
                "replanning_rounds",
                "replanning rounds",
                "how many failed plans were retried with new grounding",
            ),
        )
        if data.get(key) is not None
    )
    violation = ""
    if ok is False and execution.get("violation"):
        violation = (
            f"<div class=badbox>{html.escape(str(execution['violation']))}"
            f" @ {html.escape(str(execution.get('violated_action', '')))}</div>"
        )
    return (
        f"<div class=card>{badge}"
        "<p class=explain>How the task went: the plan the planner found, "
        "whether execution and checks succeeded, and what it cost (hover "
        "a counter for its meaning).</p>"
        f"{plan_html}{violation}<div>{metrics}</div></div>"
    )


def _render_goal(data) -> str:
    goal = (data or {}).get("goal", [])
    chips = "".join(f"<span class=chip>{html.escape(str(g))}</span>" for g in goal)
    return f"<div class=card>{chips or '<span class=muted>(no goal)</span>'}</div>"


_LIBRARY_RENDER_IDS = itertools.count()
"""
Distinct prefix per rendered library so node ids stay unique per page.
"""

_GRAPH_NODE_WIDTH = 200
_GRAPH_NODE_HEIGHT = 30
_GRAPH_ROW_STEP = 44
_GRAPH_COLUMNS = (12, 300, 588, 876)

_GRAPH_SCRIPT = """
<script>
(function(){
  if(window.__libGraphWiring){window.__libGraphWiring();return;}
  function setDim(g,id){
    var keep={};
    if(id){
      keep[id]=1;
      g.querySelectorAll('.gedge').forEach(function(p){
        if(p.dataset.from===id)keep[p.dataset.to]=1;
        if(p.dataset.to===id)keep[p.dataset.from]=1;
      });
    }
    g.querySelectorAll('.gedge').forEach(function(p){
      p.classList.toggle('dim',!!id&&p.dataset.from!==id&&p.dataset.to!==id);
    });
    g.querySelectorAll('.gnode').forEach(function(n){
      n.classList.toggle('dim',!!id&&!keep[n.dataset.node]);
    });
  }
  function wire(g){
    if(g.dataset.wired)return; g.dataset.wired='1';
    g.addEventListener('mouseover',function(e){
      var n=e.target.closest('.gnode');
      if(n&&!g.dataset.pin)setDim(g,n.dataset.node);
    });
    g.addEventListener('mouseleave',function(){if(!g.dataset.pin)setDim(g,null);});
    g.addEventListener('click',function(e){
      var n=e.target.closest('.gnode');
      if(!n){g.dataset.pin='';setDim(g,null);return;}
      var id=n.dataset.node;
      if(g.dataset.pin===id){g.dataset.pin='';setDim(g,null);return;}
      g.dataset.pin=id; setDim(g,id);
      var target=document.getElementById(g.dataset.prefix+'-entry-'+id);
      if(target){
        var d=target.closest('details');
        while(d){d.open=true;d=d.parentElement?d.parentElement.closest('details'):null;}
        target.scrollIntoView({behavior:'smooth',block:'center'});
        target.classList.add('flash');
        setTimeout(function(){target.classList.remove('flash');},1400);
      }
    });
  }
  window.__libGraphWiring=function(){
    document.querySelectorAll('svg.libgraph').forEach(wire);
  };
  window.__libGraphWiring();
})();
</script>
"""


_ROLE_COLOR_COUNT = 5
"""
Distinct role-chip colors; roles beyond that cycle.
"""

_ROLE_LINK_SCRIPT = (
    "<script>(function(){if(window._roleLinkWired)return;window._roleLinkWired=1;"
    "document.addEventListener('mouseover',function(e){"
    "var t=e.target.closest('[data-rk]');if(!t)return;var k=t.dataset.rk;"
    "document.querySelectorAll('[data-rk]').forEach(function(n){"
    "if(n.dataset.rk===k)n.classList.add('role-hl');});});"
    "document.addEventListener('mouseout',function(e){"
    "if(!e.target.closest('[data-rk]'))return;"
    "document.querySelectorAll('.role-hl').forEach(function(n){"
    "n.classList.remove('role-hl');});});})();</script>"
)
"""
Hovering any role chip highlights the same role everywhere in its card.
"""


def _contract_role_indexes(contract: dict) -> dict[str, int]:
    return {
        str(role.get("name", "")): index
        for index, role in enumerate(contract.get("roles") or [])
    }


def _role_chip(contract_uid: str, name: str, index: int) -> str:
    key = html.escape(f"{contract_uid}:{name}")
    return (
        f"<span class='role-chip rc{index % _ROLE_COLOR_COUNT}' "
        f"data-rk='{key}'>{html.escape(name)}</span>"
    )


def _contract_signature(contract: dict) -> str:
    """
    The contract as a call signature: its label's verb plus role chips.
    """
    uid = str(contract.get("uid", ""))
    label = str(contract.get("label", ""))
    verb = label.rsplit(".", 1)[-1] or label
    chips = ", ".join(
        _role_chip(uid, name, index)
        for name, index in _contract_role_indexes(contract).items()
    )
    return f"<span class=sigline>{html.escape(verb)}({chips})</span>"


def _contract_signature_text(contract: dict) -> str:
    """
    The signature as plain text, for one-line summaries.
    """
    label = str(contract.get("label", ""))
    verb = label.rsplit(".", 1)[-1] or label
    names = ", ".join(_contract_role_indexes(contract))
    return f"{verb}({names})"


def _highlight_roles(text: str, contract_uid: str, indexes: dict[str, int]) -> str:
    """
    Escape ``text`` and replace role-name words with matching chips.
    """
    escaped = html.escape(str(text))
    if not indexes:
        return escaped
    names = sorted(indexes, key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(name) for name in names) + r")\b"
    return re.sub(
        pattern,
        lambda match: _role_chip(contract_uid, match.group(1), indexes[match.group(1)]),
        escaped,
    )


def _render_chain_anatomy(open_by_default: bool = True) -> str:
    """
    Trace the generic data flow from predicate to robot.

    The design-time lane holds what a symbol library stores; the run-time lane holds
    what dispatching a single plan step creates.
    """

    def box(kind: str, code: str, what: str, junction: bool = False) -> str:
        classes = "anatomy-box junction" if junction else "anatomy-box"
        return (
            f"<div class='{classes}'><div class=what>{kind}</div>"
            f"<code>{html.escape(code)}</code>"
            f"<div class=what>{what}</div></div>"
        )

    return (
        f"<details class=libsec{' open' if open_by_default else ''}>"
        "<summary><b>How the pieces connect</b></summary>"
        "<div class=anatomy-lane>design time — stored in the symbol library</div>"
        "<div class='anatomy-grid design'>"
        + box("predicate", "relation(?object)", "a named yes/no question")
        + box("operator", "task-action ?actor ?object", "preconditions and effects")
        + box("binding", "?actor→actor, ?object→patient", "maps parameters to roles")
        + box(
            "CapabilityContract",
            "CapabilityInterface",
            "reviewed platform interface",
            junction=True,
        )
        + "</div>"
        "<div class=anatomy-join>dispatching one plan step fills the roles "
        "with grounded objects ↓</div>"
        "<div class=anatomy-lane>run time — created per dispatched action</div>"
        "<div class='anatomy-grid runtime'>"
        + box(
            "ExecutionRequest",
            "actor=robot, patient=object",
            "one message to the platform",
        )
        + box("adapter", "→ CoraplexAction(…)", "translates to a native action")
        + box(
            "verification",
            "relation(?object) re-checked",
            "by the predicate's truth procedure",
        )
        + "</div></details>"
    )


_OPERATOR_USAGE_CACHE: dict[tuple, dict[str, list[str]]] = {}
"""
Single-entry memo keyed by every library file's (path, mtime).
"""


def _contracts_in_libraries(library_dir: Path | None) -> tuple[CapabilityContract, ...]:
    """
    Every capability contract the shipped symbol libraries carry, once per uid.
    """
    contracts: dict[str, CapabilityContract] = {}
    if library_dir is None or not library_dir.is_dir():
        return ()
    for path in sorted(library_dir.glob("*.json")):
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        for entry in data.get("capability_contracts", ()):
            contract = from_json(entry)
            contracts.setdefault(contract.uid, contract)
    return tuple(contracts[uid] for uid in sorted(contracts))


def _operator_usage_by_capability(library_dir: Path | None) -> dict[str, list[str]]:
    """
    Which shipped operators bind each capability, as ``uid -> labels``.
    """
    usage: dict[str, list[str]] = {}
    if library_dir is None or not library_dir.is_dir():
        return usage
    files = sorted(library_dir.glob("*.json"))
    cache_key = tuple((str(path), path.stat().st_mtime_ns) for path in files)
    cached = _OPERATOR_USAGE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    for path in files:
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        for operator in data.get("operators") or []:
            binding = operator.get("execution_binding") or {}
            uid = str((binding.get("capability_ref") or {}).get("uid", ""))
            if not uid:
                continue
            label = f"{operator.get('name', '?')} ({path.stem})"
            entries = usage.setdefault(uid, [])
            if label not in entries:
                entries.append(label)
    _OPERATOR_USAGE_CACHE.clear()
    _OPERATOR_USAGE_CACHE[cache_key] = usage
    return usage


_CAPABILITY_CATEGORIES: tuple[tuple[str, str, str], ...] = (
    ("navigation", "Moving around", "base motion toward goals"),
    (
        "articulation",
        "Opening and closing",
        "drawers, doors and other jointed parts",
    ),
    ("attention", "Looking", "pointing the robot's sensors at something"),
    ("perception", "Perceiving", "detecting objects in the scene"),
    (
        "manipulation",
        "Handling objects",
        "reaching, grasping, picking up, placing, transporting",
    ),
    ("robot", "Body posture", "gripper, arm, torso and carry postures"),
    ("material", "Working with material", "mixing, pouring, cutting"),
)
"""
Display groups for the catalog, keyed by the label prefix before the dot.
"""


def _render_grounding_factories(
    workspace: GroundingFactoryWorkspace,
    vocabulary: GroundingVocabulary,
) -> str:
    """
    Render local candidates and current locally approved implementations.
    """
    candidates = workspace.candidates()
    catalog = GroundingFactoryCatalog.load(workspace=workspace)
    specifications = tuple(catalog)
    local_count = len(workspace.specifications())
    pending_count = sum(
        str(candidate.review_status) == "pending-review" for candidate in candidates
    )
    candidate_cards = [
        _render_grounding_candidate(candidate) for candidate in candidates
    ]
    approved_cards = [
        _render_approved_grounding_factory(specification)
        for specification in specifications
    ]
    unavailable_cards = "".join(
        "<div class=card><b><code>"
        f"{html.escape(uid)}</code></b> "
        "<span class='badge warn'>unavailable</span>"
        f"<div class=muted>{html.escape(reason)}</div></div>"
        for uid, reason in sorted(catalog.unavailable.items())
    )
    vocabulary_candidates = workspace.vocabulary_candidates()
    vocabulary_candidate_cards = "".join(
        _render_grounding_vocabulary_candidate(item) for item in vocabulary_candidates
    )
    vocabulary_rows = "".join(
        "<tr><td><code>"
        f"{html.escape(entry.qualified_name)}</code></td><td>"
        f"{html.escape(entry.signature)}</td><td>"
        f"{html.escape(entry.kind.value)}</td><td><code>"
        f"{html.escape(entry.source_checksum)}</code></td></tr>"
        for entry in vocabulary.entries
    )
    return (
        "<div class=hero><h1>Grounding factories</h1>"
        "<p>Agent-authored EQL remains non-executable until a human approves it. "
        "Approval materializes one local Python module; publishing it to a shared "
        "repository is a separate decision.</p></div>"
        "<div class=catalog-summary>"
        f"<span class=metric><b>candidates</b>: {len(candidates)}</span>"
        f"<span class=metric><b>pending review</b>: {pending_count}</span>"
        f"<span class=metric><b>approved catalog</b>: {len(specifications)}</span>"
        f"<span class=metric><b>approved local</b>: {local_count}</span></div>"
        "<h2>Review queue</h2>"
        + ("".join(candidate_cards) or "<p class=muted>No candidates.</p>")
        + "<h2>Current approved catalog</h2>"
        + ("".join(approved_cards) or "<p class=muted>No approved factories.</p>")
        + (
            f"<h2>Unavailable factories</h2>{unavailable_cards}"
            if unavailable_cards
            else ""
        )
        + "<h2>Scanned EQL review queue</h2>"
        + (
            vocabulary_candidate_cards
            or "<p class=muted>No source scan has been synchronized.</p>"
        )
        + "<h2>Scanned EQL vocabulary</h2>"
        + (
            "<div class=tablewrap><table><tr><th>symbol</th><th>signature</th>"
            "<th>kind</th><th>source checksum</th></tr>"
            f"{vocabulary_rows}</table></div>"
            if vocabulary_rows
            else "<p class=muted>No EQL symbols discovered.</p>"
        )
    )


def _render_grounding_vocabulary_candidate(
    candidate: GroundingVocabularyCandidate,
) -> str:
    """
    Render one source-discovered EQL symbol and its review controls.
    """
    entry = candidate.entry
    status = str(candidate.review_status)
    controls = ""
    if status == "pending-review":
        approve_url = url_for(
            "approve_grounding_vocabulary", qualified_name=entry.qualified_name
        )
        reject_url = url_for(
            "reject_grounding_vocabulary", qualified_name=entry.qualified_name
        )
        controls = (
            f"<form method=post action='{html.escape(approve_url)}'>"
            "<input name=reviewer required placeholder='reviewer'> "
            "<input name=review_note placeholder='review note'> "
            "<button type=submit>Approve vocabulary symbol</button></form>"
            f"<form method=post action='{html.escape(reject_url)}'>"
            "<input name=reviewer required placeholder='reviewer'> "
            "<input name=review_note required placeholder='rejection reason'> "
            "<button type=submit>Reject</button></form>"
        )
    return (
        "<div class=card><b><code>"
        f"{html.escape(entry.qualified_name)}</code></b> "
        f"<span class='badge'>{html.escape(status)}</span>"
        f"<div>{html.escape(entry.signature)} · {html.escape(entry.kind.value)}</div>"
        + (
            f"<div class=muted>{html.escape(entry.documentation)}</div>"
            if entry.documentation
            else ""
        )
        + "<div class=muted>source: <code>"
        + html.escape(entry.source_file)
        + "</code></div>"
        + "<div class=muted>source checksum: <code>"
        f"{html.escape(entry.source_checksum)}</code></div>{controls}</div>"
    )


def _render_grounding_candidate(candidate: GroundingFactoryCandidate) -> str:
    """
    Render one pending or reviewed EQL source proposal.
    """
    status = str(candidate.review_status)
    roles = "".join(
        f"<span class=chip>{html.escape(role.name)}: "
        f"{html.escape(role.symbol_type.short_name)}</span>"
        for role in candidate.roles
    )
    evidence = (
        "".join(
            f"<span class=chip>{html.escape(item)}</span>"
            for item in candidate.evidence
        )
        or "<span class=muted>none recorded</span>"
    )
    parameters = (
        "".join(
            f"<span class=chip>{html.escape(parameter.name)}: "
            f"{html.escape(parameter.value_type.value)}</span>"
            for parameter in candidate.parameters
        )
        or "<span class=muted>none</span>"
    )
    review = (
        _render_grounding_review_form(candidate)
        if status == "pending-review"
        else (
            "<div class=lib-row><span class=k>reviewed by</span><span>"
            f"{html.escape(candidate.reviewed_by or '—')}</span></div>"
            "<div class=lib-row><span class=k>review note</span><span>"
            f"{html.escape(candidate.review_note or '—')}</span></div>"
        )
    )
    badge_class = "warn" if status == "pending-review" else ""
    return (
        "<details class=libsec open><summary><b>"
        f"{html.escape(candidate.semantic_name)}</b> "
        f"<code>{html.escape(candidate.candidate_id)}</code> "
        f"<span class='badge {badge_class}'>{html.escape(status)}</span></summary>"
        "<div class=lib-row><span class=k>proposed uid</span><span><code>"
        f"{html.escape(candidate.proposed_uid)}</code></span></div>"
        f"<div class=lib-row><span class=k>roles</span><span>{roles}</span></div>"
        f"<div class=lib-row><span class=k>parameters</span><span>{parameters}</span></div>"
        "<div class=lib-row><span class=k>generated by</span><span>"
        f"{html.escape(candidate.generated_by)}</span></div>"
        "<div class=lib-row><span class=k>rationale</span><span>"
        f"{html.escape(candidate.rationale)}</span></div>"
        f"<div class=lib-row><span class=k>evidence</span><span>{evidence}</span></div>"
        "<details><summary>proposed native EQL source</summary><pre class=wrap>"
        f"{html.escape(candidate.source_code)}</pre></details>{review}</details>"
    )


def _render_grounding_review_form(candidate: GroundingFactoryCandidate) -> str:
    """
    Render explicit approve and reject operations for one pending candidate.
    """
    approve_url = url_for(
        "approve_grounding_factory", candidate_id=candidate.candidate_id
    )
    reject_url = url_for(
        "reject_grounding_factory", candidate_id=candidate.candidate_id
    )
    return (
        "<div class=card><b>Human review</b>"
        f"<form method=post action='{html.escape(approve_url)}'>"
        "<input name=reviewer required placeholder='reviewer'> "
        "<input name=review_note placeholder='review note'> "
        "<button type=submit>Approve and materialize</button></form>"
        f"<form method=post action='{html.escape(reject_url)}'>"
        "<input name=reviewer required placeholder='reviewer'> "
        "<input name=review_note required placeholder='rejection reason'> "
        "<button type=submit>Reject</button></form></div>"
    )


def _render_approved_grounding_factory(
    specification: GroundingFactorySpec,
) -> str:
    """
    Render one active source-backed grounding factory.
    """
    origin_label = (
        "approved-local"
        if specification.origin.value == "local"
        else specification.origin.value
    )
    parameters = (
        ", ".join(
            f"{parameter.name}: {parameter.value_type.value}"
            for parameter in specification.parameters
        )
        or "none"
    )
    return (
        "<div class=card><b>"
        f"{html.escape(specification.semantic_name)}</b> "
        f"<span class='badge ok'>{html.escape(origin_label)}</span>"
        "<div class=lib-row><span class=k>factory uid</span><span><code>"
        f"{html.escape(specification.uid)}</code></span></div>"
        "<div class=lib-row><span class=k>implementation</span><span><code>"
        f"{html.escape(specification.implementation_ref)}</code></span></div>"
        "<div class=lib-row><span class=k>checksum</span><span><code>"
        f"{html.escape(specification.implementation_checksum)}</code></span></div>"
        "<div class=lib-row><span class=k>revision</span><span>"
        f"{html.escape(specification.active_revision_id)}</span></div>"
        "<div class=lib-row><span class=k>parameters</span><span>"
        f"{html.escape(parameters)}</span></div>"
        "<div class=lib-row><span class=k>reviewed by</span><span>"
        f"{html.escape(specification.reviewed_by)}</span></div></div>"
    )


def _render_capability_catalog(
    data: dict,
    operator_usage: dict[str, list[str]] | None = None,
    realization_candidates: tuple[RealizationCandidate, ...] = (),
    contract_candidates: tuple[CapabilityContractCandidate, ...] = (),
) -> str:
    """
    The reviewed contracts, grouped by what kind of activity they cover.

    Each contract is one flat, fully labeled card; ``operator_usage`` maps contract UIDs
    to the shipped operators bound to them, so every card also shows its task-side
    neighbours.
    """
    operator_usage = operator_usage or {}
    summary = data.get("summary") or {}
    entries = data.get("contracts") or []
    pending_actions = data.get("pending_actions") or []
    counts = Counter(
        str((entry.get("realization") or {}).get("status", "unknown"))
        for entry in entries
    )
    filters = [
        ("all", len(entries)),
        *(sorted(counts.items())),
    ]
    filter_buttons = "".join(
        f"<button class='tab{' active' if status == 'all' else ''}' "
        f"data-cap-status='{html.escape(status)}'>{html.escape(status)} ({count})</button>"
        for status, count in filters
    )

    def render_entry(entry: dict) -> str:
        contract = entry.get("contract") or {}
        uid = str(contract.get("uid", ""))
        realization = entry.get("realization") or {}
        status = str(realization.get("status", "unknown"))
        verification = str(realization.get("effect_verification", "unknown"))
        actions = realization.get("actions") or []
        action_chips = (
            "".join(
                "<span class=chip title='requires "
                + html.escape(", ".join(action.get("required_resources") or []))
                + "'>"
                + html.escape(str(action.get("action_class", ""))).rsplit(".", 1)[-1]
                + "</span>"
                for action in actions
            )
            or "<span class=muted>no native action mapped</span>"
        )
        used_by = (
            "".join(
                f"<a href='/library'><span class=chip>{html.escape(label)}</span></a>"
                for label in operator_usage.get(uid, [])
            )
            or "<span class=muted>none in the shipped libraries</span>"
        )
        extras = (
            "<div class=lib-row><span class=k>implemented by</span>"
            f"<span>{action_chips} <span class=muted>· effect verification: "
            f"{html.escape(verification)}</span></span></div>"
            "<div class=cap-links><span class=k>used by operators</span>"
            f"<span>{used_by}</span></div>"
        )
        badge = (
            f"<span class='badge {'ok' if status == 'ready' else ''}'>"
            f"{html.escape(status)}</span>"
        )
        return (
            f"<details class=capability-entry id='cap-{html.escape(uid)}' "
            f"data-status='{html.escape(status)}'>"
            "<summary>"
            f"<b>{html.escape(str(contract.get('label', '')))}</b>"
            f"<span class=sig-text>{html.escape(_contract_signature_text(contract))}</span>"
            f"{badge}</summary>"
            + _render_contract(contract, extras=extras, card_class="cap-detail")
            + "</details>"
        )

    grouped: dict[str, list[dict]] = {}
    for entry in entries:
        label = str((entry.get("contract") or {}).get("label", ""))
        grouped.setdefault(label.split(".", 1)[0], []).append(entry)
    sections = []
    known_prefixes = {prefix for prefix, _, _ in _CAPABILITY_CATEGORIES}
    categories = list(_CAPABILITY_CATEGORIES) + [
        (prefix, prefix, "")
        for prefix in sorted(grouped)
        if prefix not in known_prefixes
    ]
    for prefix, title, description in categories:
        members = grouped.get(prefix)
        if not members:
            continue
        note = f" — {description}" if description else ""
        sections.append(
            "<div class=cap-group>"
            f"<div class=cap-cat><b>{html.escape(title)}</b> "
            f"<span class=muted>({len(members)}){html.escape(note)} · "
            f"<code>{html.escape(prefix)}.*</code></span></div>"
            + "".join(render_entry(entry) for entry in members)
            + "</div>"
        )
    contract_section = ""
    if contract_candidates:
        pending_contracts = sum(
            candidate.review_status is CapabilityReviewStatus.PENDING
            for candidate in contract_candidates
        )
        contract_section = (
            "<details class=libsec open><summary><b>Contract candidates</b> "
            f"<span class=muted>({pending_contracts} pending)</span></summary>"
            + "".join(
                _render_contract_candidate(candidate)
                for candidate in contract_candidates
            )
            + "</details>"
        )
    candidate_section = ""
    if realization_candidates:
        pending_candidates = sum(
            candidate.review_status is CapabilityReviewStatus.PENDING
            for candidate in realization_candidates
        )
        candidate_section = (
            "<details class=libsec open><summary><b>Realization candidates</b> "
            f"<span class=muted>({pending_candidates} pending)</span></summary>"
            + "".join(
                _render_realization_candidate(candidate)
                for candidate in realization_candidates
            )
            + "</details>"
        )
    pending_section = ""
    if pending_actions:
        pending_rows = "".join(
            "<div class=lib-row><span class=k>"
            + html.escape(str(action.get("action_class", ""))).rsplit(".", 1)[-1]
            + "</span><span><code>"
            + html.escape(str(action.get("source_id", "")))
            + "</code><br><span class=muted>"
            + html.escape(str(action.get("summary", "")))
            + "</span></span></div>"
            for action in pending_actions
        )
        pending_section = (
            "<details class=libsec><summary><b>Actions awaiting semantic review</b> "
            f"<span class=muted>({len(pending_actions)})</span></summary>"
            f"{pending_rows}</details>"
        )
    script = (
        "<script>document.querySelectorAll('[data-cap-status]').forEach(function(b){"
        "b.addEventListener('click',function(){"
        "document.querySelectorAll('[data-cap-status]').forEach(function(x){x.classList.remove('active');});"
        "b.classList.add('active');const wanted=b.dataset.capStatus;"
        "document.querySelectorAll('.capability-entry').forEach(function(c){"
        "c.hidden=wanted!=='all'&&c.dataset.status!==wanted;});"
        "document.querySelectorAll('.cap-group').forEach(function(g){"
        "g.hidden=!g.querySelector('.capability-entry:not([hidden])');});});});"
        # a #cap-<uid> link from the library page lands on a folded entry:
        # unfold it so the jump shows the contract, not a closed row
        "function resymOpenHash(){var h=decodeURIComponent(location.hash.slice(1));"
        "if(!h)return;var el=document.getElementById(h);"
        "if(el&&el.classList.contains('capability-entry')){el.open=true;"
        "el.scrollIntoView();}}"
        "window.addEventListener('hashchange',resymOpenHash);resymOpenHash();"
        "</script>"
    )
    return (
        "<div class=hero><h1>Coraplex capability catalog</h1>"
        "<p>Every reviewed semantic interface and its native implementation "
        "evidence. READY means an ExecutionRequest adapter exists. Dispatch also "
        "requires a compatible robot and grounded effect predicates in the current "
        "task; task-required marks contracts whose predicates are not bundled.</p></div>"
        f"{_render_chain_anatomy(open_by_default=False)}"
        "<div class=catalog-summary>"
        f"<span class=metric><b>contracts</b>: {summary.get('contracts', 0)}</span>"
        f"<span class=metric><b>Coraplex actions</b>: {summary.get('actions', 0)}</span>"
        f"<span class=metric><b>adapter ready</b>: {summary.get('ready', 0)}</span>"
        f"<span class=metric><b>built-in verification</b>: "
        f"{summary.get('built_in_verification', 0)}</span>"
        f"<span class=metric><b>awaiting review</b>: "
        f"{summary.get('pending_review', 0)}</span></div>"
        f"{contract_section}{candidate_section}{pending_section}"
        f"<div class=tabs>{filter_buttons}</div>{''.join(sections)}"
        f"{script}{_ROLE_LINK_SCRIPT}"
    )


def _render_contract_candidate(candidate: CapabilityContractCandidate) -> str:
    """
    Render one proposed contract and, while pending, its review controls.
    """
    status = candidate.review_status
    contract = candidate.contract
    roles = "".join(
        f"<li><code>{html.escape(role.name)}</code>: "
        + html.escape(
            ", ".join(item.short_name for item in role.accepted_symbol_types)
            or " | ".join(role.allowed_values)
        )
        + ("" if role.required else " <span class=muted>(optional)</span>")
        + "</li>"
        for role in contract.roles
    )
    controls = ""
    if status is CapabilityReviewStatus.PENDING:
        approve_url = url_for("approve_contract", candidate_id=candidate.candidate_id)
        reject_url = url_for("reject_contract", candidate_id=candidate.candidate_id)
        controls = (
            f"<form method=post action='{html.escape(approve_url)}'>"
            "<input name=reviewer required placeholder='reviewer'> "
            "<input name=review_note placeholder='review note'> "
            "<button type=submit>Approve contract</button></form>"
            f"<form method=post action='{html.escape(reject_url)}'>"
            "<input name=reviewer required placeholder='reviewer'> "
            "<input name=review_note required placeholder='rejection reason'> "
            "<button type=submit>Reject</button></form>"
        )
    badge_class = "warn" if status is CapabilityReviewStatus.PENDING else ""
    return (
        f"<div class=card id='contract-{html.escape(candidate.candidate_id)}'>"
        f"<b><code>{html.escape(candidate.candidate_id)}</code></b> "
        f"<span class='badge {badge_class}'>{html.escape(status.value)}</span>"
        f"<div><code>{html.escape(contract.uid)}</code> · {html.escape(contract.label)}"
        f" · v{html.escape(contract.version)}</div>"
        f"<ul>{roles}</ul>"
        f"<div>effects: {html.escape(', '.join(contract.verifiable_effect_names))}"
        f" · success when <code>{html.escape(contract.success_relation)}</code></div>"
        f"<div class=muted>evidence: "
        f"{html.escape(', '.join(candidate.action_source_ids) or 'none')}</div>"
        f"<div class=muted>{html.escape(candidate.generated_by)}: "
        f"{html.escape(candidate.rationale)}</div>"
        + (
            f"<div class=muted>{html.escape(status.value)} by "
            f"{html.escape(candidate.reviewed_by or '')}"
            + (
                f" — {html.escape(candidate.review_note)}"
                if candidate.review_note
                else ""
            )
            + "</div>"
            if candidate.reviewed_by
            else ""
        )
        + f"{controls}</div>"
    )


def _render_realization_candidate(candidate: RealizationCandidate) -> str:
    """
    Render one proposed realization and, while pending, its review controls.
    """
    status = candidate.review_status
    realization = candidate.realization
    sources = "".join(
        f"<li><code>{html.escape(source.parameter)}</code> ← "
        f"{html.escape(source.kind.value)} <code>{html.escape(source.value)}</code></li>"
        for source in realization.parameter_sources
    )
    condition = ""
    if realization.applies_when is not None:
        condition = (
            f"<div class=muted>when <code>{html.escape(realization.applies_when.role)}</code>"
            + (
                f" = <code>{html.escape(realization.applies_when.value)}</code>"
                if realization.applies_when.value is not None
                else " is bound"
            )
            + "</div>"
        )
    controls = ""
    if status is CapabilityReviewStatus.PENDING:
        approve_url = url_for(
            "approve_realization", candidate_id=candidate.candidate_id
        )
        reject_url = url_for("reject_realization", candidate_id=candidate.candidate_id)
        controls = (
            f"<form method=post action='{html.escape(approve_url)}'>"
            "<input name=reviewer required placeholder='reviewer'> "
            "<input name=review_note placeholder='review note'> "
            "<button type=submit>Approve realization</button></form>"
            f"<form method=post action='{html.escape(reject_url)}'>"
            "<input name=reviewer required placeholder='reviewer'> "
            "<input name=review_note required placeholder='rejection reason'> "
            "<button type=submit>Reject</button></form>"
        )
    badge_class = "warn" if status is CapabilityReviewStatus.PENDING else ""
    return (
        f"<div class=card id='realization-{html.escape(candidate.candidate_id)}'>"
        f"<b><code>{html.escape(candidate.candidate_id)}</code></b> "
        f"<span class='badge {badge_class}'>{html.escape(status.value)}</span>"
        f"<div><code>{html.escape(realization.capability_uid)}</code> ← "
        f"<code>{html.escape(candidate.action_source_id)}</code></div>"
        f"{condition}<ul>{sources}</ul>"
        f"<div class=muted>{html.escape(candidate.generated_by)}: "
        f"{html.escape(candidate.rationale)}</div>"
        + (
            f"<div class=muted>{html.escape(status.value)} by "
            f"{html.escape(candidate.reviewed_by or '')}"
            + (
                f" — {html.escape(candidate.review_note)}"
                if candidate.review_note
                else ""
            )
            + "</div>"
            if candidate.reviewed_by
            else ""
        )
        + f"{controls}</div>"
    )


def _graph_label(text: str, limit: int = 27) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _render_library_graph(data: dict, prefix: str) -> str:
    """
    The library as one interactive layered graph: predicates feed operators
    (preconditions), operators write predicates (effects) and connect through explicit
    bindings to capability contracts.

    Hover traces a symbol; click pins the trace and jumps to the definition below.
    """
    predicates = [str(p.get("name", "")) for p in data.get("predicates") or []]
    operators = data.get("operators") or []
    contracts = data.get("capability_contracts") or []
    operator_names = [str(op.get("name", "")) for op in operators]
    binding_names = [str(op.get("name", "")) for op in operators]
    contract_uids = [str(c.get("uid", "")) for c in contracts]
    columns = (
        [("pred:" + n, n) for n in predicates],
        [("op:" + n, n) for n in operator_names],
        [("binding:" + n, n + " binding") for n in binding_names],
        [("cap:" + uid, uid.removeprefix("resym:")) for uid in contract_uids],
    )
    if not any(columns):
        return ""
    max_rows = max(len(column) for column in columns)
    height = 46 + max_rows * _GRAPH_ROW_STEP
    positions: dict[str, tuple[float, float]] = {}
    node_parts = []
    for column_index, column in enumerate(columns):
        x = _GRAPH_COLUMNS[column_index]
        start_y = 40 + (max_rows - len(column)) * _GRAPH_ROW_STEP / 2
        for row, (node_id, label) in enumerate(column):
            y = start_y + row * _GRAPH_ROW_STEP
            positions[node_id] = (x, y + _GRAPH_NODE_HEIGHT / 2)
            kind = node_id.split(":", 1)[0]
            node_parts.append(
                f"<g class='gnode kind-{kind}' data-node='{html.escape(node_id)}'>"
                f"<rect x={x} y={y:.0f} rx=7 width={_GRAPH_NODE_WIDTH} height={_GRAPH_NODE_HEIGHT}></rect>"
                f"<text x={x + _GRAPH_NODE_WIDTH / 2:.0f} y={y + _GRAPH_NODE_HEIGHT / 2 + 4:.0f} "
                f"text-anchor=middle>{html.escape(_graph_label(label))}</text>"
                f"<title>{html.escape(label)}</title></g>"
            )
    edge_parts = []

    def edge(source: str, target: str, kind: str, tooltip: str) -> None:
        if source not in positions or target not in positions:
            return
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        start_x = x1 + _GRAPH_NODE_WIDTH if x1 < x2 else x1
        end_x = x2 if x1 < x2 else x2 + _GRAPH_NODE_WIDTH
        edge_parts.append(
            f"<path class='gedge {kind}' data-from='{html.escape(source)}' "
            f"data-to='{html.escape(target)}' "
            f"marker-end='url(#{prefix}-arrow-{kind})' "
            f"d='M{start_x},{y1:.0f} C{start_x + 58},{y1:.0f} {end_x - 58},{y2:.0f} {end_x},{y2:.0f}'>"
            f"<title>{html.escape(tooltip)}</title></path>"
        )

    for operator in operators:
        name = str(operator.get("name", ""))
        for literal in operator.get("preconditions") or []:
            edge(
                "pred:" + str(literal.get("predicate", "")),
                "op:" + name,
                "pre",
                f"{name} requires {literal.get('predicate')} (precondition)",
            )
        for literal in operator.get("add_effects") or []:
            edge(
                "op:" + name,
                "pred:" + str(literal.get("predicate", "")),
                "add",
                f"{name} makes {literal.get('predicate')} true (add effect)",
            )
        for literal in operator.get("delete_effects") or []:
            edge(
                "op:" + name,
                "pred:" + str(literal.get("predicate", "")),
                "del",
                f"{name} makes {literal.get('predicate')} false (delete effect)",
            )
        binding = operator.get("execution_binding") or {}
        capability = (binding.get("capability_ref") or {}).get("uid")
        if capability:
            edge(
                "op:" + name,
                "binding:" + name,
                "maps",
                f"{name} resolves parameters through its execution binding",
            )
            edge(
                "binding:" + name,
                "cap:" + str(capability),
                "bind",
                f"{name}'s binding targets {capability}",
            )
    arrows = "".join(
        f"<marker id='{prefix}-arrow-{kind}' viewBox='0 0 10 10' refX=9 refY=5 "
        f"markerWidth=5.5 markerHeight=5.5 orient=auto>"
        f"<path d='M0,0L10,5L0,10z' fill='{color}'/></marker>"
        for kind, color in (
            ("pre", "#2563eb"),
            ("add", "#16a34a"),
            ("del", "#dc2626"),
            ("maps", "#d97706"),
            ("bind", "#7c3aed"),
        )
    )
    headers = "".join(
        f"<text class=gcol x={x + _GRAPH_NODE_WIDTH / 2:.0f} y=22 text-anchor=middle>{label}</text>"
        for x, label in zip(
            _GRAPH_COLUMNS,
            ("predicates", "operators", "execution bindings", "capabilities"),
        )
    )
    legend = (
        "<div class=graph-legend>"
        "<span><i style='border-color:#2563eb'></i>precondition</span>"
        "<span><i style='border-color:#16a34a'></i>add effect</span>"
        "<span><i style='border-color:#dc2626'></i>delete effect</span>"
        "<span><i style='border-color:#d97706'></i>parameter mapping</span>"
        "<span><i style='border-color:#7c3aed;border-top-style:dashed'></i>capability binding</span>"
        "</div>"
    )
    width = _GRAPH_COLUMNS[-1] + _GRAPH_NODE_WIDTH + 12
    return (
        f"<div class=card>{legend}"
        f"<div style='overflow-x:auto'>"
        f"<svg class=libgraph data-prefix='{prefix}' width={width} height={height:.0f} "
        f"viewBox='0 0 {width} {height:.0f}' xmlns='http://www.w3.org/2000/svg'>"
        f"<defs>{arrows}</defs>{headers}{''.join(edge_parts)}{''.join(node_parts)}</svg></div>"
        "<p class=explain>Hover a symbol to trace its connections; click it "
        "to pin the trace and jump to its definition below.</p>"
        f"</div>{_GRAPH_SCRIPT}"
    )


def _short_type(reference) -> str:
    """
    ``module:QualName`` shortened to the class name, full ref on hover.
    """
    if isinstance(reference, dict):
        reference = reference.get("python_type_ref", "?")
    text = str(reference)
    short = text.rsplit(":", 1)[-1].rsplit(".", 1)[-1]
    return f"<span title='{html.escape(text)}'>{html.escape(short)}</span>"


def _binding_pairs(role_bindings):
    """
    Role bindings as (role, binding) pairs, from a mapping or a pair list.
    """
    if isinstance(role_bindings, dict):
        return list(role_bindings.items())
    return [tuple(pair) for pair in role_bindings or []]


def _binding_is_parameter(source) -> bool:
    """
    The binding source, whether serialized as an enum object or a bare string.
    """
    if isinstance(source, dict):
        return source.get("name") == "PARAMETER"
    return str(source).lower() == "parameter"


def _literal_chips(literals) -> str:
    chips = []
    for literal in literals or []:
        prefix = "not " if literal.get("negated") else ""
        arguments = ", ".join(f"?{a}" for a in literal.get("arguments") or [])
        chips.append(
            f"<span class=chip>{html.escape(prefix + str(literal.get('predicate', '')))}"
            f"({html.escape(arguments)})</span>"
        )
    return "".join(chips) or "<span class=muted>—</span>"


def _render_symbol_library(data) -> str:
    """
    A symbol library rendered for reading: capability contracts with their roles and
    ontology alignment, predicates with their truth procedures, operators with
    preconditions/effects and their bindings.
    """
    if not isinstance(data, dict) or "predicates" not in data:
        return _collapsible_json(data)
    prefix = f"lib{next(_LIBRARY_RENDER_IDS)}"
    predicates = data.get("predicates") or []
    operators = data.get("operators") or []
    contracts = data.get("capability_contracts") or []
    parts = [
        "<div class=card>"
        "<span class=metric><b>format</b>: krrood json</span>"
        f"<span class=metric><b>predicates</b>: {len(predicates)}</span>"
        f"<span class=metric><b>operators</b>: {len(operators)}</span>"
        f"<span class=metric><b>capability contracts</b>: {len(contracts)}</span>"
        "</div>",
        _render_library_graph(data, prefix),
    ]
    if predicates:
        rows = "".join(
            f"<tr id='{prefix}-entry-pred:{html.escape(str(p.get('name', '')))}'>"
            f"<td><code>{html.escape(str(p.get('name', '')))}</code></td>"
            f"<td>{_stable_ref_cell({'uid': p.get('uid'), 'version': p.get('version')})}</td>"
            f"<td>{' × '.join(_short_type(t) for t in p.get('parameter_types') or []) or '<span class=muted>0-ary</span>'}</td>"
            f"<td>{_truth_procedure_cell(p)}</td>"
            f"<td>{'fluent' if p.get('fluent') else 'static'}</td>"
            f"<td>{_provenance_cell(p.get('provenance'))}</td>"
            "</tr>"
            for p in predicates
        )
        parts.append(
            "<details class=libsec open><summary><b>Predicates</b> "
            f"({len(predicates)}) <span class=muted>— each stores how to "
            "decide, never truth values</span></summary>"
            "<div class=tablewrap><table>"
            "<tr><th>predicate</th><th>stable reference</th><th>signature</th><th>truth procedure</th>"
            "<th>kind</th><th>source</th></tr>"
            f"{rows}</table></details>"
        )
    if operators:
        cards = "".join(
            f"<div id='{prefix}-entry-op:{html.escape(str(op.get('name', '')))}'>"
            f"{_render_operator(op)}</div>"
            for op in operators
        )
        parts.append(
            "<details class=libsec><summary><b>Operators</b> "
            f"({len(operators)}) <span class=muted>— symbolic actions and "
            "their capability bindings</span></summary>"
            "<p class=explain>Typed parameters, preconditions, effects — and "
            "the binding that maps parameters and constants onto one "
            f"capability's roles.</p>{cards}</details>"
        )
        binding_cards = "".join(
            f"<div id='{prefix}-entry-binding:{html.escape(str(op.get('name', '')))}'>"
            f"{_render_execution_binding(op)}</div>"
            for op in operators
        )
        parts.append(
            "<details class=libsec><summary><b>Operator execution bindings</b> "
            f"({len(operators)}) <span class=muted>— maps operator-local "
            "parameters to stable capability roles</span></summary>"
            "<p class=explain>A binding is a reusable semantic adapter. "
            "Grounding later replaces its parameter references with concrete "
            f"object IDs to create an ExecutionRequest.</p>{binding_cards}</details>"
        )
    if contracts:
        cards = "".join(
            f"<div id='{prefix}-entry-cap:{html.escape(str(c.get('uid', '')))}'>"
            + _render_contract(
                c,
                extras=(
                    "<div class=cap-links>"
                    f"<a href='/capabilities#cap-{html.escape(str(c.get('uid', '')))}'>"
                    "view in the capability catalog →</a></div>"
                ),
            )
            + "</div>"
            for c in contracts
        )
        parts.append(
            "<details class=libsec><summary><b>Capability contracts</b> "
            f"({len(contracts)}) <span class=muted>— frozen platform "
            "interfaces; the repair loop may never change them</span></summary>"
            "<p class=explain>What the platform can physically do: semantic "
            "roles, a success condition, and the effects that can be "
            f"verified afterwards.</p>{cards}</details>"
        )
    parts.append(_ROLE_LINK_SCRIPT)
    return "".join(parts)


def _truth_procedure_cell(predicate: dict) -> str:
    grounding_plan = predicate.get("grounding_plan")
    if isinstance(grounding_plan, dict):
        factory_uid = html.escape(str(grounding_plan.get("factory_uid", "?")))
        checksum = html.escape(
            str(grounding_plan.get("approved_factory_checksum", "?"))
        )
        details = [f"factory <code>{factory_uid}</code>"]
        if grounding_plan.get("negated"):
            details.append("negated")
        parameters = grounding_plan.get("parameters")
        if isinstance(parameters, dict):
            parameter_items = parameters.items()
        elif isinstance(parameters, list):
            parameter_items = (
                item for item in parameters if isinstance(item, list) and len(item) == 2
            )
        else:
            parameter_items = ()
        rendered_parameters = ", ".join(
            f"{html.escape(str(name))}={html.escape(str(value))}"
            for name, value in parameter_items
        )
        if rendered_parameters:
            details.append(f"parameters <code>{rendered_parameters}</code>")
        details.append(f"checksum <code>{checksum}</code>")
        procedure_ref = predicate.get("truth_procedure_ref")
        if isinstance(procedure_ref, dict):
            details.append(_stable_ref_cell(procedure_ref))
        return "grounding plan · " + " · ".join(details)
    return "<span class=muted>none</span>"


def _stable_ref_cell(reference) -> str:
    if not isinstance(reference, dict):
        return "<span class=muted>legacy name</span>"
    uid = html.escape(str(reference.get("uid", "?")))
    version = html.escape(str(reference.get("version", "1")))
    return f"<code>{uid}@{version}</code>"


def _predicate_ref_label(reference) -> str:
    if isinstance(reference, dict):
        return str(
            reference.get("local_name") or reference.get("uid", "").rsplit("/", 1)[-1]
        )
    return str(reference)


def _provenance_cell(provenance) -> str:
    if not isinstance(provenance, dict):
        return "<span class=muted>?</span>"
    source = str(provenance.get("source", "?"))
    admitted = provenance.get("admitted_at")
    backend = provenance.get("proposal_backend")
    text = source
    if backend:
        text += f" · {backend}"
    if admitted:
        text += f" · admitted {admitted}"
    return html.escape(text)


def _render_operator(operator: dict) -> str:
    parameters = " ".join(
        f"?{html.escape(str(name))}:" + _short_type(symbol_type)
        for name, symbol_type in operator.get("parameters") or []
    )
    binding = operator.get("execution_binding") or {}
    binding_html = "<span class=muted>no binding</span>"
    if binding:
        capability = binding.get("capability_ref") or {}
        chips = "".join(
            f"<span class=chip>{html.escape(str(role))} ← "
            + html.escape(
                f"?{value.get('value', '')}"
                if _binding_is_parameter(value.get("source"))
                else str(value.get("value", ""))
            )
            + "</span>"
            for role, value in _binding_pairs(binding.get("role_bindings"))
        )
        binding_html = (
            f"<code>{html.escape(str(capability.get('uid', '?')))}"
            f"@{html.escape(str(capability.get('version', '1')))}</code> {chips}"
        )
    return (
        "<div class=card>"
        f"<div class=lib-head><b><code>({html.escape(str(operator.get('name', '')))}"
        f"{' ' if parameters else ''}{parameters})</code></b>"
        f"<span class=muted>{_provenance_cell(operator.get('provenance'))}</span></div>"
        f"<div class=lib-row><span class=k>preconditions</span><span>{_literal_chips(operator.get('preconditions'))}</span></div>"
        f"<div class=lib-row><span class=k>add effects</span><span>{_literal_chips(operator.get('add_effects'))}</span></div>"
        f"<div class=lib-row><span class=k>delete effects</span><span>{_literal_chips(operator.get('delete_effects'))}</span></div>"
        f"<div class=lib-row><span class=k>binding</span><span>{binding_html}</span></div>"
        "</div>"
    )


def _render_execution_binding(operator: dict) -> str:
    binding = operator.get("execution_binding") or {}
    capability = binding.get("capability_ref") or {}
    rows = []
    for role, value in _binding_pairs(binding.get("role_bindings")):
        source = (
            "parameter" if _binding_is_parameter(value.get("source")) else "constant"
        )
        raw = str(value.get("value", ""))
        mapped = f"?{raw}" if source == "parameter" else raw
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(role))}</code></td>"
            f"<td>{html.escape(source)}</td>"
            f"<td><code>{html.escape(mapped)}</code></td>"
            "</tr>"
        )
    return (
        "<div class=card>"
        f"<div class=lib-head><b>{html.escape(str(operator.get('name', '')))}</b>"
        "<span class=muted>OperatorExecutionBinding</span></div>"
        f"<div class=lib-row><span class=k>capability</span><span><code>"
        f"{html.escape(str(capability.get('uid', '?')))}@"
        f"{html.escape(str(capability.get('version', '1')))}</code></span></div>"
        "<div class=tablewrap><table><tr><th>semantic role</th>"
        f"<th>source</th><th>operator value</th></tr>{''.join(rows)}</table></div>"
        f"<div class=lib-row><span class=k>proposal source</span><span>"
        f"{html.escape(str(binding.get('proposal_source', 'seed')))}</span></div>"
        "</div>"
    )


def _render_contract(
    contract: dict,
    extras: str = "",
    badge: str = "",
    card_class: str = "card",
    card_attributes: str = "",
) -> str:
    """
    A capability contract as one flat, fully labeled card.

    Every field says what it is: the header carries "name" and "id" tags,
    and each body line is a labeled row. The label's verb and the roles form
    the signature; the success relation is rendered with the same colored
    role chips, so the connection between the name, the role list and the
    relation is visible instead of implied.
    """
    uid = str(contract.get("uid", ""))
    indexes = _contract_role_indexes(contract)
    role_rows = []
    for index, role in enumerate(contract.get("roles") or []):
        name = str(role.get("name", ""))
        notes = ["required" if role.get("required", True) else "optional"]
        accepted = ", ".join(
            _short_type(t) for t in role.get("accepted_symbol_types") or []
        )
        if accepted:
            notes.append(f"accepts {accepted}")
        values = " | ".join(role.get("allowed_values") or [])
        if values:
            notes.append(f"one of <code>{html.escape(values)}</code>")
        role_rows.append(
            f"<div class=role-row data-rk='{html.escape(f'{uid}:{name}')}'>"
            f"{_role_chip(uid, name, index)}"
            f"<span class=role-note>{' · '.join(notes)}</span></div>"
        )
    success = _highlight_roles(str(contract.get("success_relation", "")), uid, indexes)
    effects = "".join(
        f"<span class=chip title='{html.escape(str(effect))}'>"
        f"{html.escape(_predicate_ref_label(effect))}</span>"
        for effect in contract.get("verifiable_effects") or []
    )
    constants = "".join(
        f"<span class=chip>{html.escape(f'{effect}: {role}={value}')}</span>"
        for effect, role, value in (
            tuple(entry) for entry in contract.get("effect_role_values") or []
        )
    )
    alignment = contract.get("ontology_alignment") or {}
    alignment_html = ""
    if alignment:
        target = str(alignment.get("target_iri", ""))
        target_name = target.rsplit("#", 1)[-1] or target
        reviewed = (
            "human-reviewed"
            if alignment.get("human_reviewed")
            else f"decided by {alignment.get('decided_by', '?')}"
        )
        alignment_html = (
            "<details><summary>ontology: "
            f"{html.escape(str(alignment.get('relation', '?')))} of "
            f"{html.escape(target_name)} ({html.escape(reviewed)})</summary>"
            "<div class=lib-row><span class=k>target</span>"
            f"<span><code>{html.escape(target)}</code> · {html.escape(str(alignment.get('source_version', '')))}</span></div>"
            "<div class=lib-row><span class=k>rationale</span>"
            f"<span>{html.escape(str(alignment.get('rationale', '')))}</span></div>"
            "</details>"
        )
    return (
        f"<div class='{card_class}'{' ' if card_attributes else ''}{card_attributes}>"
        "<div class=lib-head>"
        f"<span class=field><span class=k-inline>name</span>"
        f"<b>{html.escape(str(contract.get('label', '')))}</b></span>"
        f"<span class=field><span class=k-inline>stable id</span>"
        f"<code>{html.escape(uid)}@{html.escape(str(contract.get('version', '1')))}</code></span>"
        f"{badge}</div>"
        f"<div class=lib-row><span class=k>what it does</span>"
        f"<span>{_contract_signature(contract)}</span></div>"
        f"<div class=lib-row><span class=k>success when</span>"
        f"<span><code>{success}</code></span></div>"
        f"<div class=lib-row><span class=k>roles</span>"
        f"<span>{''.join(role_rows) or '<span class=muted>—</span>'}</span></div>"
        f"<div class=lib-row><span class=k>verifiable effects</span>"
        f"<span>{effects or '<span class=muted>—</span>'} "
        "<span class=muted>— re-checked after execution by each "
        "predicate's truth procedure</span></span></div>"
        + (
            f"<div class=lib-row><span class=k>effect constants</span><span>{constants}</span></div>"
            if constants
            else ""
        )
        + alignment_html
        + extras
        + "</div>"
    )


# -- library diffs and version stores -----------------------------------------

_LIBRARY_SECTIONS = (
    ("predicates", "name"),
    ("operators", "name"),
    ("capability_contracts", "uid"),
)


def _render_library_diff(before, after) -> str:
    """
    Added / removed / changed symbols between two library snapshots.
    """
    if not isinstance(before, dict) or not isinstance(after, dict):
        return "<p class=muted>(snapshots unreadable)</p>"
    parts = []
    for section, key in _LIBRARY_SECTIONS:
        old = {str(entry.get(key)): entry for entry in before.get(section) or []}
        new = {str(entry.get(key)): entry for entry in after.get(section) or []}
        added = [name for name in new if name not in old]
        removed = [name for name in old if name not in new]
        changed = [name for name in new if name in old and new[name] != old[name]]
        if not (added or removed or changed):
            continue
        entries = []
        for name in added:
            entries.append(
                f"<div class='diff-entry diff-add'><span class=diff-name>+ {html.escape(name)}</span>"
                f"<details><summary>added entry</summary>{_json_block(new[name])}</details></div>"
            )
        for name in removed:
            entries.append(
                f"<div class='diff-entry diff-del'><span class=diff-name>− {html.escape(name)}</span>"
                f"<details><summary>removed entry</summary>{_json_block(old[name])}</details></div>"
            )
        for name in changed:
            entries.append(
                f"<div class='diff-entry diff-chg'><span class=diff-name>~ {html.escape(name)}</span>"
                f"{_field_diff(old[name], new[name])}</div>"
            )
        parts.append(
            f"<div class=card><div class=lib-head><b>{html.escape(section.replace('_', ' '))}</b>"
            f"<span class=metric><b>+</b>{len(added)}</span>"
            f"<span class=metric><b>−</b>{len(removed)}</span>"
            f"<span class=metric><b>~</b>{len(changed)}</span></div>"
            f"{''.join(entries)}</div>"
        )
    if not parts:
        return "<p class=muted>no differences</p>"
    legend = (
        "<p class=explain>+ added · − removed · ~ changed "
        "(each changed field shown as old → new).</p>"
    )
    return legend + "".join(parts)


def _field_diff(old: dict, new: dict) -> str:
    lines = []
    for field_name in sorted(set(old) | set(new)):
        if old.get(field_name) == new.get(field_name):
            continue
        before = json.dumps(old.get(field_name), ensure_ascii=False)
        after = json.dumps(new.get(field_name), ensure_ascii=False)
        if len(before) > 220:
            before = before[:220] + "…"
        if len(after) > 220:
            after = after[:220] + "…"
        lines.append(
            f"<div><b>{html.escape(field_name)}</b>: "
            f"<code class=diff-old>{html.escape(before)}</code> → "
            f"<code class=diff-new>{html.escape(after)}</code></div>"
        )
    return "".join(lines)


VERSION_STORE_RENDER_LIMIT = 40
"""
Maximum version stores rendered on one run page.
"""


def _render_version_stores(run_dir: Path) -> str:
    """
    Every versioned library store below a run, each with its commit chain and the diff
    between consecutive versions.
    """
    stores: dict[Path, list[Path]] = {}
    for path in sorted(run_dir.glob("**/versions/v[0-9]*.json")):
        stores.setdefault(path.parent.parent, []).append(path)
    if not stores:
        return ""
    parts = [
        _section(
            "Library version stores",
            "Versioned symbol-library snapshots. Open a diff to see exactly "
            "which symbols were added, removed, or changed.",
        )
    ]
    rendered = list(stores.items())[:VERSION_STORE_RENDER_LIMIT]
    for store_root, version_paths in rendered:
        relative = store_root.relative_to(run_dir)
        quarantine = _load_json(store_root / "quarantine.json") or {}
        parts.append(
            f"<div class=file><div class=name>{html.escape(str(relative))}</div>"
        )
        previous = None
        for path in version_paths:
            record = _load_json(path) or {}
            meta = record.get("meta") or {}
            role_label = str(meta.get("role") or "")
            chips = "".join(
                f"<span class=metric><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</span>"
                for k, v in meta.items()
                if not isinstance(v, (dict, list))
            )
            badge = (
                "<span class='badge bad'>quarantined</span>"
                if path.stem in quarantine
                else ""
            )
            parts.append(
                f"<div class=card><div class=lib-head><b>{html.escape(path.stem)}</b>"
                + (
                    f"<span class=muted>— {html.escape(role_label)}</span>"
                    if role_label
                    else ""
                )
                + f"{badge}</div>{chips}"
            )
            library = record.get("library")
            if previous is not None:
                parts.append(
                    f"<details><summary>diff {html.escape(previous[0])} → "
                    f"{html.escape(path.stem)}</summary>"
                    f"{_render_library_diff(previous[1], library)}</details>"
                )
            parts.append(
                f"<details><summary>full library at {html.escape(path.stem)}</summary>"
                f"{_render_symbol_library(library)}</details></div>"
            )
            previous = (path.stem, library)
        parts.append("</div>")
    if len(stores) > len(rendered):
        parts.append(
            f"<p class=muted>{len(stores) - len(rendered)} more version "
            f"stores not rendered (limit {VERSION_STORE_RENDER_LIMIT}).</p>"
        )
    return "".join(parts)


def _render_transcript(path: Path, detail_url=None, query=None) -> str:
    """
    Render an append-only language-model exchange log without task assumptions.
    """
    records = [record for record in _iter_jsonl(path) if isinstance(record, dict)]
    if not records:
        return "<p class=muted>(no language-model exchanges)</p>"
    rows = []
    for index, record in enumerate(records, start=1):
        agent = html.escape(str(record.get("agent_name") or "model"))
        attempt = html.escape(str(record.get("attempt") or 1))
        prompt = html.escape(str(record.get("prompt") or ""))
        response = html.escape(str(record.get("response") or ""))
        error = record.get("parse_error")
        error_html = f"<p class=bad>{html.escape(str(error))}</p>" if error else ""
        rows.append(
            "<details class=turn>"
            f"<summary>exchange {index} · {agent} · attempt {attempt}</summary>"
            f"<h4>prompt</h4><pre class=wrap>{prompt}</pre>"
            f"<h4>response</h4><pre class=wrap>{response}</pre>"
            f"{error_html}</details>"
        )
    return (
        f"<div class=card><span class=metric><b>exchanges</b>: {len(records)}</span>"
        "</div>" + "".join(rows)
    )


def _render_provenance(data) -> str:
    """
    provenance.json as a readable reproducibility card.
    """
    if not isinstance(data, dict):
        return _collapsible_json(data)
    repository = data.get("repository") or {}
    invocation = data.get("invocation") or {}
    clean = repository.get("clean")
    clean_badge = ""
    if clean is not None:
        clean_badge = (
            "<span class='badge ok'>clean tree</span>"
            if clean
            else "<span class='badge bad'>dirty tree</span>"
        )
    commit = str(repository.get("commit", ""))
    grid = "".join(
        f"<div class=k>{html.escape(key)}</div><div>{value}</div>"
        for key, value in (
            (
                "commit",
                f"<code>{html.escape(commit[:12])}</code> on "
                f"<code>{html.escape(str(repository.get('branch', '?')))}</code> {clean_badge}",
            ),
            (
                "invocation",
                f"<code>{html.escape(' '.join(str(a) for a in invocation.get('argv') or []))}</code>",
            ),
        )
    )
    arguments = invocation.get("arguments") or {}
    argument_chips = "".join(
        f"<span class=metric><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</span>"
        for k, v in arguments.items()
        if v is not None and not isinstance(v, (dict, list))
    )
    return (
        "<div class=card>"
        "<p class=explain>What produced this run: the exact code version "
        "and command line, for bit-level reproducibility.</p>"
        f"<div class=kv>{grid}</div><div>{argument_chips}</div>"
        f"<details><summary>full provenance JSON</summary>{_json_block(data)}</details>"
        "</div>"
    )


def _render_events(path: Path) -> str:
    rows = []
    for record in _iter_jsonl(path):
        if not isinstance(record, dict):
            rows.append(f"<div class=evt>{html.escape(str(record))}</div>")
            continue
        when = record.pop("at", "")
        name = record.pop("event", "")
        extra = " · ".join(
            f"{html.escape(str(k))}: {html.escape(str(v))}"
            for k, v in record.items()
            if not isinstance(v, (dict, list))
        )
        rows.append(
            f"<div class=evt><span class=t>{html.escape(str(when))}</span>"
            f"<span class=n>{html.escape(str(name))}</span>"
            f"<span class=muted>{extra}</span></div>"
        )
    return (
        f"<div class=card>{''.join(rows)}</div>"
        if rows
        else "<p class=muted>(no events)</p>"
    )


# -- helpers ------------------------------------------------------------------


def _plan_html(plan) -> str:
    return (
        "<ol class=plan>"
        + "".join(f"<li><code>{html.escape(str(step))}</code></li>" for step in plan)
        + "</ol>"
    )


def _collapsible_json(data) -> str:
    block = _json_block(data)
    if data is None:
        return "<div class='card muted'>(unreadable)</div>"
    # short payloads inline; long ones fold
    if len(json.dumps(data)) <= 400:
        return f"<div class=card>{block}</div>"
    return (
        f"<div class=card><details><summary>show JSON</summary>{block}</details></div>"
    )


def _json_block(data) -> str:
    return f"<pre class=wrap>{html.escape(json.dumps(data, indent=2, ensure_ascii=False))}</pre>"


def _run_row_status(run_dir: Path, meta: dict) -> str:
    """
    The outcome badge for one index row: the answer a reader scans the list for — did
    this run succeed — without opening the run.
    """
    summary = meta.get("summary") or {}
    outcome = summary.get("outcome")
    if outcome in {"success", "failure"}:
        klass = "ok" if outcome == "success" else "bad"
        label = summary.get("label", outcome)
        return f"<span class='badge {klass}'>{html.escape(str(label))}</span>"
    succeeded = summary.get("succeeded")
    if succeeded is not None:
        return (
            "<span class='badge ok'>&#10003; succeeded</span>"
            if succeeded
            else "<span class='badge bad'>&#10007; failed</span>"
        )
    if summary.get("status") == "succeeded":
        return "<span class='badge ok'>finished</span>"
    if not meta.get("ended_at"):
        return (
            "<span class='badge warn' data-tip=\"run.json records no end "
            'time — the run is either still going or was interrupted">'
            "unfinished</span>"
        )
    if summary.get("status"):
        return (
            f"<span class='badge warn'>{html.escape(str(summary['status']))}" "</span>"
        )
    return ""


def _duration_text(meta: dict) -> str | None:
    """
    '4m 28s' / '5h 51m' from run.json start and end stamps.
    """
    try:
        delta = datetime.fromisoformat(str(meta["ended_at"])) - datetime.fromisoformat(
            str(meta["started_at"])
        )
    except (KeyError, TypeError, ValueError):
        return None
    minutes, seconds = divmod(int(delta.total_seconds()), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s"


def _iter_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            yield line


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Web viewer for resym run logs.")
    parser.add_argument(
        "runs_root",
        nargs="?",
        default=os.environ.get("RESYM_RUNS_DIR", "runs"),
        help="Directory holding per-run log folders (default: runs/ or $RESYM_RUNS_DIR).",
    )
    parser.add_argument(
        "--library-dir",
        default=None,
        help=(
            "Directory with the shipped symbol libraries for the /library "
            "page (default: a 'library' directory beside the runs root)."
        ),
    )
    parser.add_argument(
        "--grounding-workspace",
        default=None,
        help=(
            "Local grounding-factory review workspace. When supplied, the Viewer "
            "enables candidate review and local source materialization."
        ),
    )
    parser.add_argument(
        "--realization-workspace",
        default=None,
        help=(
            "Local capability-realization review workspace. When supplied, the "
            "Viewer lists realization candidates for approval."
        ),
    )
    parser.add_argument(
        "--contract-workspace",
        default=None,
        help=(
            "Local capability-contract review workspace. When supplied, the Viewer "
            "lists contract candidates for approval and admitted contracts join the "
            "catalog."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    runs_root = Path(args.runs_root)
    library_dir = (
        Path(args.library_dir)
        if args.library_dir is not None
        else runs_root.resolve().parent / "library"
    )
    grounding_workspace = None
    grounding_vocabulary = None
    if args.grounding_workspace is not None:
        grounding_initialization = initialize_grounding_factories(
            Path(args.grounding_workspace)
        )
        grounding_workspace = grounding_initialization.workspace
        grounding_vocabulary = grounding_initialization.reviewed_vocabulary
    realization_workspace = None
    if args.realization_workspace is not None:
        realization_workspace = CoraplexRealizationWorkspace(
            Path(args.realization_workspace)
        )
    contract_workspace = None
    if args.contract_workspace is not None:
        contract_workspace = CapabilityContractWorkspace(Path(args.contract_workspace))
    app = create_app(
        runs_root,
        library_dir=library_dir,
        grounding_workspace=grounding_workspace,
        grounding_vocabulary=grounding_vocabulary,
        realization_workspace=realization_workspace,
        contract_workspace=contract_workspace,
    )
    print(
        f"serving logs from {Path(args.runs_root).resolve()} at http://{args.host}:{args.port}"
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
