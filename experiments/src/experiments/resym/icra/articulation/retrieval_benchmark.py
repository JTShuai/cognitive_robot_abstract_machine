"""
Drawer-domain development benchmark for the ICRA evaluation.
"""

from collections.abc import Sequence

from resym.knowledge.corpus import DomainFragment
from resym.knowledge.dense import Benchmark
from resym.knowledge.retrieval import RetrievalQuery, tokenize


def drawer_development_benchmark(
    fragments: Sequence[DomainFragment],
) -> Benchmark:
    queries = (
        RetrievalQuery(
            text="missing operator model closed drawer opened handle of ready to open",
            goal_predicates=("closed",),
            causal_predicates=("opened", "closed", "handle-of", "ready-to-open"),
            argument_arities=(1,),
        ),
        RetrievalQuery(
            text="missing operator model opened drawer closed handle of ready to open",
            goal_predicates=("opened",),
            causal_predicates=("opened", "closed", "handle-of", "ready-to-open"),
            argument_arities=(1,),
        ),
        RetrievalQuery(
            text="missing predicate model opened drawer open drawer pull handle",
            goal_predicates=("opened",),
            causal_predicates=("closed", "handle-of"),
            argument_arities=(1,),
        ),
    )
    relevant = frozenset(
        fragment.fragment_id
        for fragment in fragments
        if _mentions_drawer_manipulation(fragment)
    )
    if not relevant:
        raise ValueError("No drawer-manipulation fragments in this corpus")
    return [(query, relevant) for query in queries]


def _mentions_drawer_manipulation(fragment: DomainFragment) -> bool:
    tokens = {
        token
        for name in (*fragment.predicate_names(), *fragment.operator_names())
        for token in tokenize(name)
    }
    return "drawer" in tokens and bool(tokens & {"open", "close", "opened", "closed"})
