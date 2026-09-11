"""
Behavioural validation used by the controlled ICRA experiments only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from typing_extensions import Callable, Optional

from krrood.adapters.json_serializer import to_json
from resym.core.model import SymbolLibrary
from resym.repair.curator import Curator
from resym.repair.patch import ModelPatch
from resym.repair.versioning import VersionedLibraryStore


class SuiteGroup(Enum):
    CAPABILITY_POSITIVE = "capability_positive"
    CAPABILITY_NEGATIVE = "capability_negative"
    BOUNDARY = "boundary"
    REGRESSION = "regression"


@dataclass(frozen=True)
class AdmissionTest:
    name: str
    group: SuiteGroup
    context_id: str
    run: Callable[[SymbolLibrary], list[str]]


@dataclass(frozen=True)
class MandatorySuite:
    version: str
    tests: tuple[AdmissionTest, ...]

    def coverage_problems(self, proposal_context_id: str) -> list[str]:
        problems = []
        present = {test.group for test in self.tests}
        for group in SuiteGroup:
            if group not in present:
                problems.append(f"mandatory suite has no {group.value} test")
        if not any(test.context_id != proposal_context_id for test in self.tests):
            problems.append(
                "mandatory suite has no admission context different from the "
                f"proposal context '{proposal_context_id}'"
            )
        return problems


@dataclass
class TestResult:
    name: str
    group: SuiteGroup
    context_id: str
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass
class BehaviouralAdmissionReport:
    static_objections: list[str] = field(default_factory=list)
    suite_problems: list[str] = field(default_factory=list)
    test_results: list[TestResult] = field(default_factory=list)
    suite_version: Optional[str] = None

    @property
    def admitted(self) -> bool:
        return (
            not self.static_objections
            and not self.suite_problems
            and all(result.passed for result in self.test_results)
        )

    def evidence(self) -> dict:
        return {
            "static_objections": list(self.static_objections),
            "suite_problems": list(self.suite_problems),
            "suite_version": self.suite_version,
            "tests": [
                {
                    "name": result.name,
                    "group": result.group.value,
                    "context_id": result.context_id,
                    "failures": list(result.failures),
                }
                for result in self.test_results
            ],
        }


@dataclass
class BehaviouralCurator(Curator):
    """
    Experimental wrapper: static Curator plus controlled scene probes.
    """

    suite: MandatorySuite = field(default_factory=lambda: MandatorySuite("", ()))

    def review(
        self,
        patch: ModelPatch,
        library: SymbolLibrary,
        proposal_context_id: str,
    ) -> BehaviouralAdmissionReport:
        report = BehaviouralAdmissionReport(suite_version=self.suite.version)
        report.static_objections = self.static_review(patch, library)
        if report.static_objections:
            return report
        report.suite_problems = self.suite.coverage_problems(proposal_context_id)
        if report.suite_problems:
            return report
        candidate = self.candidate(patch, library)
        for test in self.suite.tests:
            try:
                failures = test.run(candidate)
            except Exception as error:  # noqa: BLE001 - candidate code fails closed
                failures = [f"test crashed ({type(error).__name__}): {error}"]
            report.test_results.append(
                TestResult(test.name, test.group, test.context_id, failures)
            )
        return report

    def admit(
        self,
        patch: ModelPatch,
        library: SymbolLibrary,
        store: VersionedLibraryStore,
        proposal_context_id: str,
        metadata: Optional[dict] = None,
    ) -> tuple[BehaviouralAdmissionReport, Optional[str]]:
        report = self.review(patch, library, proposal_context_id)
        if not report.admitted:
            return report, None
        version_id = store.commit(
            self.candidate(patch, library),
            metadata={
                **(metadata or {}),
                "proposal_context_id": proposal_context_id,
                "patch": to_json(patch),
                "admission_evidence": report.evidence(),
            },
        )
        return report, version_id
