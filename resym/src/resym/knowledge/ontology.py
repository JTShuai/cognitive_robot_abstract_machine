"""
Deterministic retrieval over the frozen robot ontologies.

The language model may propose an ontology alignment, but it never reads a mutable web
page and it never decides whether an IRI exists.  This module verifies the frozen
manifest, builds a small lexical index from RDF/XML, and admits only references that are
present in that index.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from urllib.parse import urljoin

from typing_extensions import Iterable, Mapping, Sequence

from resym.core.model import CapabilityContract, OntologyAlignment

RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
XML = "http://www.w3.org/XML/1998/namespace"

ABOUT = f"{{{RDF}}}about"
RESOURCE = f"{{{RDF}}}resource"
XML_BASE = f"{{{XML}}}base"

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKEN = re.compile(r"[a-z0-9]+")


class OntologyEntityKind(Enum):
    CLASS = "class"
    OBJECT_PROPERTY = "object_property"
    DATA_PROPERTY = "data_property"
    ANNOTATION_PROPERTY = "annotation_property"
    NAMED_INDIVIDUAL = "named_individual"


_ENTITY_TAGS = {
    f"{{{OWL}}}Class": OntologyEntityKind.CLASS,
    f"{{{OWL}}}ObjectProperty": OntologyEntityKind.OBJECT_PROPERTY,
    f"{{{OWL}}}DatatypeProperty": OntologyEntityKind.DATA_PROPERTY,
    f"{{{OWL}}}AnnotationProperty": OntologyEntityKind.ANNOTATION_PROPERTY,
    f"{{{OWL}}}NamedIndividual": OntologyEntityKind.NAMED_INDIVIDUAL,
}

PROPERTY_KINDS = frozenset(
    {
        OntologyEntityKind.OBJECT_PROPERTY,
        OntologyEntityKind.DATA_PROPERTY,
        OntologyEntityKind.ANNOTATION_PROPERTY,
    }
)


@dataclass(frozen=True)
class OntologySource:
    source_id: str
    version: str
    provenance: str
    license: str


@dataclass(frozen=True)
class OntologyEntity:
    iri: str
    kinds: frozenset[OntologyEntityKind]
    labels: tuple[str, ...]
    comments: tuple[str, ...]
    parent_iris: tuple[str, ...]
    source_ids: tuple[str, ...]

    @property
    def local_name(self) -> str:
        return _local_name(self.iri)

    def index_text(self) -> str:
        return " ".join(
            (
                self.local_name,
                *self.labels,
                *self.comments,
                *(_local_name(iri) for iri in self.parent_iris),
            )
        )


@dataclass(frozen=True)
class OntologyHit:
    entity: OntologyEntity
    score: float
    matched_terms: tuple[str, ...]


@dataclass
class _EntityAccumulator:
    kinds: set[OntologyEntityKind] = field(default_factory=set)
    labels: set[str] = field(default_factory=set)
    comments: set[str] = field(default_factory=set)
    parent_iris: set[str] = field(default_factory=set)
    source_ids: set[str] = field(default_factory=set)


class OntologyIndex:
    """
    Local lexical index and deterministic admission gate.
    """

    K1 = 1.5
    B = 0.75

    def __init__(
        self,
        entities: Iterable[OntologyEntity],
        sources: Mapping[str, OntologySource],
    ):
        self.entities = tuple(sorted(entities, key=lambda entity: entity.iri))
        self.sources = dict(sources)
        self._by_iri = {entity.iri: entity for entity in self.entities}
        self._documents = [_tokens(entity.index_text()) for entity in self.entities]
        self._frequencies = [Counter(document) for document in self._documents]
        self._lengths = [len(document) for document in self._documents]
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        self._document_frequency: Counter[str] = Counter()
        for frequency in self._frequencies:
            self._document_frequency.update(frequency.keys())

    @classmethod
    def from_directory(cls, root: Path) -> OntologyIndex:
        """
        Load every OWL file declared by ``manifest.json`` after hashing.
        """
        root = root.resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise OntologyNotInstalledError(root)
        manifest = json.loads(manifest_path.read_text())
        accumulators: dict[str, _EntityAccumulator] = {}
        sources: dict[str, OntologySource] = {}

        for source in manifest["sources"]:
            source_id = source["id"]
            version = source.get("release") or source.get("git_commit")
            if not version:
                raise InvalidOntologyManifestError(
                    f"source '{source_id}' has no release or git_commit"
                )
            sources[source_id] = OntologySource(
                source_id=source_id,
                version=version,
                provenance=source["provenance"],
                license=source["license"],
            )
            for file_record in source["files"]:
                path = (root / file_record["path"]).resolve()
                if not path.is_relative_to(root):
                    raise InvalidOntologyManifestError(
                        f"ontology path escapes root: {file_record['path']}"
                    )
                _verify_hash(path, file_record["sha256"])
                _merge_rdf_xml(path, source_id, accumulators)

        entities = (
            OntologyEntity(
                iri=iri,
                kinds=frozenset(entry.kinds),
                labels=tuple(sorted(entry.labels)),
                comments=tuple(sorted(entry.comments)),
                parent_iris=tuple(sorted(entry.parent_iris)),
                source_ids=tuple(sorted(entry.source_ids)),
            )
            for iri, entry in accumulators.items()
        )
        return cls(entities, sources)

    def get(self, iri: str) -> OntologyEntity | None:
        return self._by_iri.get(iri)

    def require(
        self,
        iri: str,
        allowed_kinds: frozenset[OntologyEntityKind] | None = None,
    ) -> OntologyEntity:
        entity = self.get(iri)
        if entity is None:
            raise UnknownOntologyIriError(iri)
        if allowed_kinds is not None and entity.kinds.isdisjoint(allowed_kinds):
            raise OntologyKindMismatchError(iri, allowed_kinds, entity.kinds)
        return entity

    def retrieve(self, query: str, top_k: int = 10) -> list[OntologyHit]:
        """
        Rank entities by labels, local names, definitions, and parents.
        """
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        bm25 = self._bm25(query_tokens)
        maximum = max(bm25, default=0.0)
        query_set = set(query_tokens)
        query_normalized = " ".join(query_tokens)
        hits: list[OntologyHit] = []
        for index, entity in enumerate(self.entities):
            names = (entity.local_name, *entity.labels)
            name_token_sets = [set(_tokens(name)) for name in names]
            name_overlap = max(
                (
                    len(query_set & tokens) / len(query_set)
                    for tokens in name_token_sets
                ),
                default=0.0,
            )
            exact_name = any(
                " ".join(_tokens(name)) == query_normalized for name in names
            )
            text_score = bm25[index] / maximum if maximum > 0 else 0.0
            score = text_score + name_overlap + (2.0 if exact_name else 0.0)
            if score <= 0:
                continue
            matched = tuple(sorted(query_set & set(self._documents[index])))
            hits.append(OntologyHit(entity, score, matched))
        hits.sort(key=lambda hit: (-hit.score, hit.entity.iri))
        return hits[:top_k]

    def retrieve_for_contract(
        self, contract: CapabilityContract, top_k: int = 10
    ) -> list[OntologyHit]:
        """
        Build a reproducible ontology query from one capability contract.
        """
        role_names = tuple(role.name for role in contract.roles)
        role_value_terms = tuple(
            term
            for role in contract.roles
            for term in (
                *(
                    symbol_type.short_name.lower()
                    for symbol_type in role.accepted_symbol_types
                ),
                *role.allowed_values,
            )
        )
        query = " ".join(
            (
                contract.label,
                contract.success_relation,
                *contract.verifiable_effect_names,
                *role_names,
                *role_value_terms,
                *role_value_terms,
            )
        )
        return self.retrieve(query, top_k)

    def admit_alignment(
        self,
        *,
        candidate_iri: str | None,
        relation: str,
        role_mapping: Mapping[str, str] | None = None,
        cited_classes: Sequence[str] = (),
        cited_properties: Sequence[str] = (),
    ) -> OntologyAlignment:
        """
        Validate an LLM proposal and create the persisted alignment.
        """
        for iri in cited_classes:
            self.require(iri, frozenset({OntologyEntityKind.CLASS}))
        for iri in cited_properties:
            self.require(iri, PROPERTY_KINDS)
        for iri in (role_mapping or {}).values():
            self.require(iri)

        if relation == "NO_MATCH":
            if candidate_iri is not None:
                raise InvalidOntologyAlignmentError(
                    "NO_MATCH must not carry a candidate IRI"
                )
            return OntologyAlignment()
        if relation not in {"EXACT_MATCH", "SPECIALIZATION"}:
            raise InvalidOntologyAlignmentError(
                f"relation '{relation}' cannot be admitted"
            )
        if candidate_iri is None:
            raise InvalidOntologyAlignmentError(f"{relation} requires a candidate IRI")

        entity = self.require(candidate_iri, frozenset({OntologyEntityKind.CLASS}))
        source_versions = tuple(
            f"{source_id}@{self.sources[source_id].version}"
            for source_id in entity.source_ids
        )
        return OntologyAlignment(
            relation=relation,
            target_iri=candidate_iri,
            source_version=", ".join(source_versions),
        )

    def _bm25(self, query_tokens: Sequence[str]) -> list[float]:
        scores = [0.0] * len(self.entities)
        count = len(self.entities)
        if count == 0 or self._average_length == 0:
            return scores
        for token in query_tokens:
            containing = self._document_frequency.get(token, 0)
            if containing == 0:
                continue
            inverse_frequency = math.log(
                (count - containing + 0.5) / (containing + 0.5) + 1.0
            )
            for index, frequency in enumerate(self._frequencies):
                occurrences = frequency.get(token, 0)
                if occurrences == 0:
                    continue
                length_norm = (
                    1.0
                    - self.B
                    + self.B * (self._lengths[index] / self._average_length)
                )
                scores[index] += inverse_frequency * (
                    occurrences
                    * (self.K1 + 1.0)
                    / (occurrences + self.K1 * length_norm)
                )
        return scores


class CorruptOntologyError(Exception):
    def __init__(self, path: Path, expected: str, actual: str):
        super().__init__(
            f"Ontology file {path} hashes to {actual}, expected {expected}."
        )


class OntologyNotInstalledError(FileNotFoundError):
    """
    The configured ontology directory has not been installed.
    """

    def __init__(self, root: Path):
        super().__init__(
            f"No ontology installation found at {root}. "
            "Run 'uv run resym-install-ontologies'."
        )


class InvalidOntologyManifestError(Exception):
    pass


class UnknownOntologyIriError(Exception):
    def __init__(self, iri: str):
        super().__init__(f"Ontology IRI does not exist in the frozen index: {iri}")


class OntologyKindMismatchError(Exception):
    def __init__(
        self,
        iri: str,
        expected: frozenset[OntologyEntityKind],
        actual: frozenset[OntologyEntityKind],
    ):
        expected_names = ", ".join(sorted(kind.value for kind in expected))
        actual_names = ", ".join(sorted(kind.value for kind in actual))
        super().__init__(
            f"Ontology IRI {iri} is {actual_names}, expected {expected_names}."
        )


class InvalidOntologyAlignmentError(Exception):
    pass


def _verify_hash(path: Path, expected: str) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise CorruptOntologyError(path, expected, digest)


def _merge_rdf_xml(
    path: Path,
    source_id: str,
    accumulators: dict[str, _EntityAccumulator],
) -> None:
    root = ET.parse(path).getroot()
    base = root.attrib.get(XML_BASE, path.as_uri())
    for element in root:
        kind = _ENTITY_TAGS.get(element.tag)
        raw_iri = element.attrib.get(ABOUT)
        if kind is None or raw_iri is None:
            continue
        iri = urljoin(base, raw_iri)
        entry = accumulators.setdefault(iri, _EntityAccumulator())
        entry.kinds.add(kind)
        entry.source_ids.add(source_id)
        for child in element:
            text = " ".join((child.text or "").split())
            if child.tag == f"{{{RDFS}}}label" and text:
                entry.labels.add(text)
            elif child.tag == f"{{{RDFS}}}comment" and text:
                entry.comments.add(text)
            elif child.tag in {
                f"{{{RDFS}}}subClassOf",
                f"{{{RDFS}}}subPropertyOf",
            }:
                parent = child.attrib.get(RESOURCE)
                if parent:
                    entry.parent_iris.add(urljoin(base, parent))


def _tokens(text: str) -> list[str]:
    separated = _CAMEL_BOUNDARY.sub(" ", text).replace("_", " ").replace("-", " ")
    return _TOKEN.findall(separated.lower())


def _local_name(iri: str) -> str:
    return iri.rsplit("#", 1)[-1].rstrip("/").rsplit("/", 1)[-1]
