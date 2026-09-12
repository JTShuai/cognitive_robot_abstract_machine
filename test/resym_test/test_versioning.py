"""
Versioned admission, quarantine, and rollback consistency (Gate P0).

Host-runnable: model and store are stdlib-only.
"""

from __future__ import annotations

import json

import pytest
from krrood.adapters.exceptions import MissingTypeError

from resym.core.model import (
    Literal,
    Operator,
    PredicateRef,
    PredicateSymbol,
    Provenance,
    SymbolLibrary,
    contract_violations,
)
from .capability_helpers import capability_contract, execution_binding
from .grounding_helpers import STUB_GROUNDING_PLAN
from resym.repair.versioning import (
    CorruptVersionError,
    EmptyStoreError,
    NonHeadRollbackError,
    NothingToRollBackError,
    UnknownVersionError,
    VersionedLibraryStore,
)

from resym.core.model import SymbolType
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer

DRAWER_TYPE = SymbolType.from_python_type(Drawer)


def seed_library() -> SymbolLibrary:
    library = SymbolLibrary()
    library.add(
        PredicateSymbol(
            name="opened",
            parameter_types=(DRAWER_TYPE,),
            grounding_plan=STUB_GROUNDING_PLAN,
            fluent=True,
        )
    )
    library.add_capability_contract(
        capability_contract(
            "test:Pull",
            (("patient", DRAWER_TYPE),),
            ("opened", "closed"),
        )
    )
    return library


def grown_operator() -> Operator:
    return Operator(
        name="open-drawer",
        parameters=(("d", DRAWER_TYPE),),
        preconditions=(),
        add_effects=(Literal("opened", ("d",)),),
        delete_effects=(),
        execution_binding=execution_binding("test:Pull", (("patient", "d"),)),
        provenance=Provenance(
            source="curation",
            capability_request="learn to open drawers",
            proposal_backend="fixed-pipeline",
            retrieved_ids=("1/100",),
        ),
    )


def test_provenance_and_contracts_survive_the_json_roundtrip():
    library = seed_library()
    library.add(grown_operator())
    reloaded = SymbolLibrary.from_json(library.to_json())
    operator = reloaded.operators["open-drawer"]
    assert operator.provenance.source == "curation"
    assert operator.provenance.retrieved_ids == ("1/100",)
    assert reloaded.capability_contracts["test:Pull"].verifiable_effect_names == (
        "opened",
        "closed",
    )


def test_untyped_payload_is_rejected():
    """
    Legacy libraries without serializer type markers require an explicit offline
    migration.
    """
    v1 = {
        "predicates": [
            {
                "name": "opened",
                "parameter_types": ["drawer"],
                "evaluator": "drawer_opened",
                "fluent": True,
            }
        ],
        "operators": [],
    }
    with pytest.raises(MissingTypeError):
        SymbolLibrary.from_json(v1)


def test_contract_effect_consistency_check():
    contract = capability_contract(
        "test:Pull", (("patient", DRAWER_TYPE),), ("opened",)
    )
    consistent = Operator(
        name="ok",
        parameters=(("d", DRAWER_TYPE),),
        preconditions=(),
        add_effects=(Literal("opened", ("d",)),),
        delete_effects=(),
        execution_binding=execution_binding("test:Pull", (("patient", "d"),)),
    )
    overreaching = Operator(
        name="bad",
        parameters=(("d", DRAWER_TYPE),),
        preconditions=(),
        add_effects=(Literal("opened", ("d",)), Literal("locked", ("d",))),
        delete_effects=(),
        execution_binding=execution_binding("test:Pull", (("patient", "d"),)),
    )
    assert contract_violations(consistent, contract) == []
    (violation,) = contract_violations(overreaching, contract)
    assert "locked" in violation


def test_contract_matches_a_local_predicate_by_stable_reference():
    predicate = PredicateSymbol(
        name="is-open",
        uid="soma:Opened",
        parameter_types=(DRAWER_TYPE,),
        grounding_plan=STUB_GROUNDING_PLAN,
        fluent=True,
    )
    contract = capability_contract("test:Pull", (("patient", DRAWER_TYPE),), ())
    contract = type(contract)(
        uid=contract.uid,
        label=contract.label,
        roles=contract.roles,
        success_relation=contract.success_relation,
        verifiable_effects=(PredicateRef("soma:Opened", local_name="opened"),),
    )
    operator = Operator(
        name="open",
        parameters=(("d", DRAWER_TYPE),),
        preconditions=(),
        add_effects=(Literal("is-open", ("d",)),),
        delete_effects=(),
        execution_binding=execution_binding("test:Pull", (("patient", "d"),)),
    )

    assert contract_violations(operator, contract, {"is-open": predicate}) == []


def test_commit_moves_head_and_links_parent(tmp_path):
    store = VersionedLibraryStore(tmp_path)
    with pytest.raises(EmptyStoreError):
        store.head_version()
    v1 = store.commit(seed_library(), metadata={"note": "seed"})
    grown = seed_library()
    grown.add(grown_operator())
    v2 = store.commit(grown, metadata={"capability_request": "close drawers"})
    assert store.head_version() == v2
    assert store.info(v2).parent == v1
    assert store.info(v1).parent is None
    assert store.info(v2).metadata["capability_request"] == "close drawers"


def test_rollback_restores_parent_content_exactly(tmp_path):
    """
    Version Recoverability: after quarantine-and-rollback the head loads byte-identical
    to the parent version — model, contracts, provenance.
    """
    store = VersionedLibraryStore(tmp_path)
    v1 = store.commit(seed_library())
    grown = seed_library()
    grown.add(grown_operator())
    v2 = store.commit(grown)

    restored_head = store.quarantine_and_roll_back(v2, reason="false admission")
    assert restored_head == v1
    assert store.head_version() == v1
    assert store.head().to_json() == seed_library().to_json()
    assert store.info(v2).quarantined
    assert not store.info(v1).quarantined
    # the quarantined version stays loadable for audit
    assert "open-drawer" in store.load(v2).operators
    # a new commit after rollback branches from the restored parent
    v3 = store.commit(seed_library())
    assert store.info(v3).parent == v1


def test_rollback_without_parent_is_refused(tmp_path):
    store = VersionedLibraryStore(tmp_path)
    v1 = store.commit(seed_library())
    with pytest.raises(NothingToRollBackError):
        store.quarantine_and_roll_back(v1, reason="nope")


def test_rollback_of_inactive_version_is_refused(tmp_path):
    store = VersionedLibraryStore(tmp_path)
    store.commit(seed_library())
    v2 = store.commit(seed_library())
    v3 = store.commit(seed_library())

    with pytest.raises(NonHeadRollbackError, match=f"current head is '{v3}'"):
        store.quarantine_and_roll_back(v2, reason="stale alert")

    assert store.head_version() == v3
    assert not store.info(v2).quarantined


def test_tampered_version_is_refused(tmp_path):
    store = VersionedLibraryStore(tmp_path)
    v1 = store.commit(seed_library())
    path = tmp_path / "versions" / f"{v1}.json"
    record = json.loads(path.read_text())
    record["library"]["predicates"][0]["name"] = "tampered"
    path.write_text(json.dumps(record))
    with pytest.raises(CorruptVersionError):
        store.load(v1)


def test_unknown_version_is_refused(tmp_path):
    store = VersionedLibraryStore(tmp_path)
    with pytest.raises(UnknownVersionError):
        store.load("v9999")
