"""
Frozen ontology loading, retrieval, and deterministic admission.
"""

from __future__ import annotations

import json

import pytest
from krrood.adapters.json_serializer import from_json, to_json

from resym.platform.capabilities import (
    articulation_capability_contract,
    capability_contracts,
    navigation_capability_contract,
)
from resym.knowledge.ontology import (
    CorruptOntologyError,
    InvalidOntologyAlignmentError,
    OntologyEntityKind,
    OntologyIndex,
    OntologyKindMismatchError,
    OntologyNotInstalledError,
    UnknownOntologyIriError,
)
from resym.knowledge.ontology_alignment import (
    apply_human_alignment_review,
    align_capability_contract,
)
from resym.llm.client import ScriptedCompletionClient
from resym.llm.schemas import OntologyAlignmentProposal
from resym.llm.structured import StructuredCompleter
from resym.llm.transcript import TranscriptRecorder
from resym.knowledge.ontology_installation import ONTOLOGY_DIRECTORY

ONTOLOGY_ROOT = ONTOLOGY_DIRECTORY
OPENING = "http://www.ease-crc.org/ont/SOMA.owl#Opening"
DRAWER = "http://www.ease-crc.org/ont/SOMA.owl#Drawer"
ROBOT = "http://purl.org/ieee1872-owl/cora-bare#Robot"
HAS_JOINT_STATE = "http://www.ease-crc.org/ont/SOMA.owl#hasJointState"
STATE_TRANSITION = "http://www.ease-crc.org/ont/SOMA.owl#StateTransition"
NAVIGATING = "http://www.ease-crc.org/ont/SOMA.owl#Navigating"


@pytest.fixture(scope="module")
def index() -> OntologyIndex:
    if not (ONTOLOGY_ROOT / "manifest.json").is_file():
        pytest.skip("ontology data is not installed")
    return OntologyIndex.from_directory(ONTOLOGY_ROOT)


def test_missing_installation_has_an_actionable_error(tmp_path):
    with pytest.raises(OntologyNotInstalledError):
        OntologyIndex.from_directory(tmp_path)


def test_frozen_files_load_and_merge_duplicate_iris(index):
    assert len(index.entities) > 1_000
    opening = index.require(OPENING)
    assert opening.kinds == frozenset({OntologyEntityKind.CLASS})
    assert opening.source_ids == ("soma",)


def test_manifest_hash_is_checked_before_parsing(tmp_path):
    ontology_file = tmp_path / "tiny.owl"
    ontology_file.write_text(
        '<?xml version="1.0"?>'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:owl="http://www.w3.org/2002/07/owl#">'
        '<owl:Class rdf:about="https://example.test/Thing"/>'
        "</rdf:RDF>"
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "test",
                        "release": "1",
                        "provenance": "test",
                        "license": "test",
                        "files": [{"path": "tiny.owl", "sha256": "0" * 64}],
                    }
                ]
            }
        )
    )
    with pytest.raises(CorruptOntologyError):
        OntologyIndex.from_directory(tmp_path)


@pytest.mark.parametrize(
    ("query", "expected_iri"),
    [("opening", OPENING), ("drawer", DRAWER), ("robot", ROBOT)],
)
def test_exact_entity_name_ranks_first(index, query, expected_iri):
    assert index.retrieve(query, top_k=1)[0].entity.iri == expected_iri


def test_contract_query_surfaces_state_transition(index):
    hits = index.retrieve_for_contract(articulation_capability_contract(), top_k=10)
    assert any(hit.entity.local_name == "StateTransition" for hit in hits)


@pytest.mark.parametrize(
    ("contract", "target_iri"),
    [
        (articulation_capability_contract(), STATE_TRANSITION),
        (navigation_capability_contract(), NAVIGATING),
    ],
)
def test_bundled_contract_ontology_alignments_are_frozen_and_valid(
    index, contract, target_iri
):
    alignment = contract.ontology_alignment
    assert alignment.relation == "SPECIALIZATION"
    assert alignment.target_iri == target_iri
    assert alignment.source_version == "soma@2.1.0"
    assert alignment.decided_by == "llm-agent"
    assert alignment.rationale
    assert alignment.evidence_iris
    for iri in alignment.evidence_iris:
        index.require(iri, frozenset({OntologyEntityKind.CLASS}))


def test_complete_capability_catalog_only_cites_frozen_ontology_classes(index):
    assert len(capability_contracts()) == 20
    for contract in capability_contracts():
        alignment = contract.ontology_alignment
        index.require(alignment.target_iri, frozenset({OntologyEntityKind.CLASS}))
        for iri in alignment.evidence_iris:
            index.require(iri, frozenset({OntologyEntityKind.CLASS}))


def test_contract_query_uses_roles_symbol_types_and_constants(index, monkeypatch):
    captured = {}

    def capture(query, top_k=10):
        captured["query"] = query
        return []

    monkeypatch.setattr(index, "retrieve", capture)

    index.retrieve_for_contract(articulation_capability_contract())

    assert "actor" in captured["query"]
    assert "agent" in captured["query"]
    assert "target_state" in captured["query"]
    assert "OPEN" in captured["query"]


def test_unknown_and_wrong_kind_iris_are_rejected(index):
    with pytest.raises(UnknownOntologyIriError):
        index.require("https://example.invalid/FabricatedCapability")
    with pytest.raises(OntologyKindMismatchError):
        index.require(HAS_JOINT_STATE, frozenset({OntologyEntityKind.CLASS}))


def test_valid_proposal_is_admitted_with_frozen_source_version(index):
    proposal = OntologyAlignmentProposal(
        candidate_iri=OPENING,
        relation="SPECIALIZATION",
        cited_classes=[OPENING, DRAWER],
        cited_properties=[HAS_JOINT_STATE],
    )
    alignment = index.admit_alignment(**proposal.model_dump())
    assert alignment.relation == "SPECIALIZATION"
    assert alignment.target_iri == OPENING
    assert alignment.source_version == "soma@2.1.0"


def test_related_proposal_is_not_admissible(index):
    proposal = OntologyAlignmentProposal(
        candidate_iri=OPENING,
        relation="RELATED",
    )
    with pytest.raises(InvalidOntologyAlignmentError):
        index.admit_alignment(**proposal.model_dump())


def test_no_match_cannot_hide_a_candidate_iri(index):
    with pytest.raises(InvalidOntologyAlignmentError):
        index.admit_alignment(candidate_iri=OPENING, relation="NO_MATCH")


def test_offline_aligner_retrieves_proposes_and_applies(index):
    inspect = json.dumps(
        {
            "tool": "inspect_entity",
            "iri": STATE_TRANSITION,
        }
    )
    submit = json.dumps(
        {
            "tool": "submit_alignment",
            "rationale": "Changing articulation state is a state transition.",
            "proposal": {
                "candidate_iri": STATE_TRANSITION,
                "relation": "SPECIALIZATION",
                "role_mapping": {},
                "cited_classes": [STATE_TRANSITION],
                "cited_properties": [],
            },
        }
    )
    client = ScriptedCompletionClient(responses=[inspect, submit])
    completer = StructuredCompleter(client, TranscriptRecorder())
    contract = articulation_capability_contract()

    result = align_capability_contract(contract, index, completer)

    assert result.admitted
    assert result.alignment.target_iri == STATE_TRANSITION
    aligned = result.apply(contract)
    assert aligned.ontology_alignment.source_version == "soma@2.1.0"
    assert aligned.ontology_alignment.decided_by == "llm-agent"
    assert aligned.ontology_alignment.decision_model == "scripted"
    assert aligned.ontology_alignment.evidence_iris == (STATE_TRANSITION,)
    reloaded = from_json(to_json(aligned))
    assert reloaded.ontology_alignment.rationale == (
        "Changing articulation state is a state transition."
    )
    assert len(result.trajectory) == 2
    assert STATE_TRANSITION in client.received_prompts[0]
    assert contract.success_relation in client.received_prompts[0]
    assert "definitions:" in client.received_prompts[1]


def test_offline_aligner_returns_structured_rejection(index):
    fabricated = "https://example.invalid/FabricatedCapability"
    response = json.dumps(
        {
            "tool": "submit_alignment",
            "rationale": "The fabricated class appears to match.",
            "proposal": {
                "candidate_iri": fabricated,
                "relation": "EXACT_MATCH",
                "role_mapping": {},
                "cited_classes": [fabricated],
                "cited_properties": [],
            },
        }
    )
    completer = StructuredCompleter(
        ScriptedCompletionClient(responses=[response]), TranscriptRecorder()
    )

    result = align_capability_contract(
        articulation_capability_contract(), index, completer, maximum_steps=1
    )

    assert not result.admitted
    assert "does not exist" in result.rejection_reason


def test_human_can_review_and_correct_agent_alignment(index):
    contract = articulation_capability_contract()
    proposal = OntologyAlignmentProposal(
        candidate_iri=STATE_TRANSITION,
        relation="SPECIALIZATION",
        cited_classes=[STATE_TRANSITION],
    )

    reviewed = apply_human_alignment_review(
        contract,
        index,
        proposal,
        reviewer="ontology-curator",
        rationale="Confirmed from the frozen class definition.",
    )

    alignment = reviewed.ontology_alignment
    assert alignment.decided_by == "human"
    assert alignment.human_reviewed
    assert alignment.human_reviewer == "ontology-curator"
