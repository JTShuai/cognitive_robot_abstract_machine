"""
Loading and parsing the untrusted declarative corpus (UniDomain Dataset).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Iterator, Optional, Union

Sexp = Union[str, list]


class FragmentParseError(Exception):
    """
    Raised when one atomic-domain record cannot be normalized.
    """

    def __init__(self, fragment_id: str, detail: str):
        super().__init__(f"Fragment '{fragment_id}': {detail}")
        self.fragment_id = fragment_id
        self.detail = detail


@dataclass(frozen=True)
class PredicateFragment:
    """
    One declared predicate of an atomic domain: signature plus gloss.
    """

    name: str
    """
    Predicate name, e.g. ``on_counter``.
    """

    arity: int
    """
    Number of parameters in the declared signature.
    """

    signature: str
    """
    The raw signature string, e.g. ``(on_counter ?o ?c)``.
    """

    gloss: str
    """
    The natural-language description shipped with the corpus.
    """


@dataclass(frozen=True)
class OperatorFragment:
    """One operator of an atomic domain, with its causal structure parsed
    out: which predicates it requires and which it adds or deletes."""

    name: str
    """
    Operator name, e.g. ``pick_from_counter``.
    """

    parameters: tuple[str, ...]
    """
    Parameter variables in declaration order (untyped in this corpus).
    """

    positive_preconditions: tuple[str, ...]
    """
    Predicate names required true.
    """

    negative_preconditions: tuple[str, ...]
    """
    Predicate names required false.
    """

    add_predicates: tuple[str, ...]
    """
    Predicate names the operator makes true.
    """

    delete_predicates: tuple[str, ...]
    """
    Predicate names the operator makes false.
    """

    raw: str
    """
    The verbatim PDDL action text, kept for provenance and adaptation.
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(
            self, "positive_preconditions", tuple(self.positive_preconditions)
        )
        object.__setattr__(
            self, "negative_preconditions", tuple(self.negative_preconditions)
        )
        object.__setattr__(self, "add_predicates", tuple(self.add_predicates))
        object.__setattr__(self, "delete_predicates", tuple(self.delete_predicates))


@dataclass(frozen=True)
class DomainFragment:
    """
    One atomic domain, normalized: the unit of retrieval.

    Untrusted by definition — it carries no grounding plan, no capability binding, and
    no execution guarantee.
    """

    fragment_id: str
    """
    ``<group>/<episode>`` derived from the corpus directory layout.
    """

    predicates: tuple[PredicateFragment, ...]
    operators: tuple[OperatorFragment, ...]

    source_path: str
    """
    Path of the record this fragment was loaded from, relative to the corpus root when
    one was given, so a frozen release stays byte-identical across machines.
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicates", tuple(self.predicates))
        object.__setattr__(self, "operators", tuple(self.operators))

    def index_text(self) -> str:
        """
        Synthesized retrieval text: the corpus ships no instruction, so the searchable
        surface is predicate glosses plus humanized operator names.
        """
        parts = [predicate.gloss for predicate in self.predicates]
        parts += [operator.name.replace("_", " ") for operator in self.operators]
        return ". ".join(part for part in parts if part)

    def predicate_names(self) -> frozenset[str]:
        return frozenset(predicate.name for predicate in self.predicates)

    def operator_names(self) -> frozenset[str]:
        return frozenset(operator.name for operator in self.operators)


def iter_atomic_domain_files(corpus_root: Path) -> Iterator[Path]:
    """
    Every atomic-domain JSON record under the corpus root, sorted for deterministic
    release builds.
    """
    yield from sorted(corpus_root.glob("*/episode_*/atomic_domain.json"))


def fragment_id_for(path: Path) -> str:
    """
    ``<group>/<episode>`` — matches the ``;; source:`` tags of the fused corpus, so
    atomic fragments and fused operators can be joined.
    """
    episode = path.parent.name
    group = path.parent.parent.name
    return f"{group}/{episode.removeprefix('episode_')}"


def load_fragment(path: Path, corpus_root: Optional[Path] = None) -> DomainFragment:
    """
    Normalize one record; malformed content raises :class:`FragmentParseError` with the
    fragment id.
    """
    fragment_id = fragment_id_for(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise FragmentParseError(fragment_id, f"unreadable record: {error}")
    if (
        not isinstance(data, dict)
        or "predicates" not in data
        or "operators" not in data
    ):
        raise FragmentParseError(fragment_id, "record lacks predicates/operators keys")
    predicates = tuple(
        _parse_predicate(fragment_id, signature, gloss)
        for signature, gloss in data["predicates"].items()
    )
    operators = tuple(
        _parse_operator(fragment_id, name, text)
        for name, text in data["operators"].items()
    )
    return DomainFragment(
        fragment_id=fragment_id,
        predicates=predicates,
        operators=operators,
        source_path=str(path.relative_to(corpus_root) if corpus_root else path),
    )


def load_corpus(
    corpus_root: Path,
) -> tuple[list[DomainFragment], list[FragmentParseError]]:
    """
    Load every parseable fragment; parse failures are collected, not raised, so a
    release build can report exactly what it excluded.
    """
    fragments: list[DomainFragment] = []
    failures: list[FragmentParseError] = []
    for path in iter_atomic_domain_files(corpus_root):
        try:
            fragments.append(load_fragment(path, corpus_root))
        except FragmentParseError as error:
            failures.append(error)
    return fragments, failures


def _parse_predicate(fragment_id: str, signature: str, gloss: str) -> PredicateFragment:
    sexp = _parse_sexp(fragment_id, signature)
    if not sexp or not isinstance(sexp[0], str):
        raise FragmentParseError(fragment_id, f"malformed predicate '{signature}'")
    return PredicateFragment(
        name=sexp[0],
        arity=len(sexp) - 1,
        signature=signature.strip(),
        gloss=str(gloss).strip(),
    )


def _parse_operator(fragment_id: str, name: str, text: str) -> OperatorFragment:
    sexp = _parse_sexp(fragment_id, text)
    sections = _action_sections(sexp)
    parameters = tuple(
        token for token in sections.get(":parameters", []) if token.startswith("?")
    )
    positive, negative = _condition_predicates(sections.get(":precondition"))
    adds, deletes = _condition_predicates(sections.get(":effect"))
    return OperatorFragment(
        name=name,
        parameters=parameters,
        positive_preconditions=positive,
        negative_preconditions=negative,
        add_predicates=adds,
        delete_predicates=deletes,
        raw=text.strip(),
    )


def _action_sections(sexp: Sexp) -> dict[str, Sexp]:
    """
    Map ``:parameters``/``:precondition``/``:effect`` keywords to the expression that
    follows them, wherever the action header sits.
    """
    if not isinstance(sexp, list):
        return {}
    tokens = sexp
    if tokens and tokens[0] == ":action":
        tokens = tokens[1:]
    sections: dict[str, Sexp] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if isinstance(token, str) and token.startswith(":") and index + 1 < len(tokens):
            sections[token] = tokens[index + 1]
            index += 2
        else:
            index += 1
    return sections


@dataclass(frozen=True)
class FragmentLiteral:
    """
    One atom of a fragment operator with its argument variables.
    """

    predicate: str
    arguments: tuple[str, ...]
    negated: bool = False


@dataclass(frozen=True)
class OperatorLiterals:
    """
    The literal-level structure of one fragment operator, parsed from its verbatim PDDL
    on demand (the normalized record only stores predicate names).
    """

    preconditions: tuple[FragmentLiteral, ...]
    add_effects: tuple[FragmentLiteral, ...]
    delete_effects: tuple[FragmentLiteral, ...]


def parse_operator_literals(raw: str) -> OperatorLiterals:
    """
    Literal-level view of an operator's verbatim PDDL text; used by the executable
    adapter, which needs variable positions, not just names.
    """
    sexp = _parse_sexp("adaptation", raw)
    sections = _action_sections(sexp)
    preconditions = _condition_literals(sections.get(":precondition"))
    effects = _condition_literals(sections.get(":effect"))
    return OperatorLiterals(
        preconditions=tuple(preconditions),
        add_effects=tuple(literal for literal in effects if not literal.negated),
        delete_effects=tuple(
            FragmentLiteral(literal.predicate, literal.arguments, negated=False)
            for literal in effects
            if literal.negated
        ),
    )


def _condition_literals(sexp: Optional[Sexp]) -> list[FragmentLiteral]:
    literals: list[FragmentLiteral] = []

    def walk(node: Sexp, negated: bool) -> None:
        if not isinstance(node, list) or not node:
            return
        head = node[0]
        if head == "and":
            for child in node[1:]:
                walk(child, negated)
        elif head == "not":
            for child in node[1:]:
                walk(child, not negated)
        elif isinstance(head, str) and not head.startswith(":"):
            literals.append(
                FragmentLiteral(
                    predicate=head,
                    arguments=tuple(
                        token for token in node[1:] if isinstance(token, str)
                    ),
                    negated=negated,
                )
            )

    if sexp is not None:
        walk(sexp, negated=False)
    return literals


def _condition_predicates(
    sexp: Optional[Sexp],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Split a condition tree into (positive, negative) predicate names, descending through
    ``and`` and ``not``.
    """
    positive: list[str] = []
    negative: list[str] = []

    def walk(node: Sexp, negated: bool) -> None:
        if not isinstance(node, list) or not node:
            return
        head = node[0]
        if head == "and":
            for child in node[1:]:
                walk(child, negated)
        elif head == "not":
            for child in node[1:]:
                walk(child, not negated)
        elif isinstance(head, str) and not head.startswith(":"):
            (negative if negated else positive).append(head)

    if sexp is not None:
        walk(sexp, negated=False)
    return tuple(dict.fromkeys(positive)), tuple(dict.fromkeys(negative))


def _parse_sexp(fragment_id: str, text: str) -> Sexp:
    """
    Parse one s-expression; PDDL ``;`` line comments are stripped first (glosses with
    embedded semicolons never reach this parser — they live in JSON values, not in the
    expression text).
    """
    stripped = "\n".join(line.split(";", 1)[0] for line in text.splitlines())
    tokens = re.findall(r"\(|\)|[^\s()]+", stripped)
    if not tokens:
        raise FragmentParseError(fragment_id, "empty expression")
    position = 0

    def parse() -> Sexp:
        nonlocal position
        token = tokens[position]
        position += 1
        if token == "(":
            children: list[Sexp] = []
            while position < len(tokens) and tokens[position] != ")":
                children.append(parse())
            if position >= len(tokens):
                raise FragmentParseError(
                    fragment_id, f"unbalanced parens in: {text[:80]}"
                )
            position += 1
            return children
        if token == ")":
            raise FragmentParseError(fragment_id, f"unexpected ')' in: {text[:80]}")
        return token.lower()

    expression = parse()
    return expression
