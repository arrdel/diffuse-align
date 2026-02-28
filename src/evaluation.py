"""
Evaluation — metrics that separate functional task success from conversational fluency.

This is a key contribution of the paper: we argue that existing multi-agent dialogue
evaluations over-weight fluency metrics (BERTScore, perplexity) and under-weight
functional success. DiffuseAlign is designed to maximize functional success, and
we need metrics that capture this.

Metric categories:
    1. Functional Success — did the task actually get done?
    2. Coordination Quality — did agents work efficiently together?
    3. Conversational Fluency — is the generated dialogue natural?
    4. The Gap — the difference between function and fluency
       (positive = functionally strong, negative = fluent but failing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class EpisodeResult:
    """Evaluation results for a single episode."""
    task_id: str
    success: bool
    steps_taken: int
    optimal_steps: int
    agent_actions: Dict[int, List[str]]  # agent_id → action sequence
    generated_utterances: List[str]
    reference_utterances: Optional[List[str]] = None
    redundant_actions: int = 0
    conflicting_actions: int = 0
    delegation_correct: int = 0
    delegation_total: int = 0
    difficulty: str = "moderate"


@dataclass
class EvaluationReport:
    """Aggregate evaluation report across episodes."""
    # Functional metrics
    task_success_rate: float = 0.0
    action_efficiency: float = 0.0
    coordination_score: float = 0.0
    avg_turn_count: float = 0.0
    delegation_accuracy: float = 0.0

    # Fluency metrics
    avg_bertscore: float = 0.0
    avg_coherence: float = 0.0

    # Composite
    functional_fluency_gap: float = 0.0

    # Per-complexity breakdown
    per_complexity: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Raw
    num_episodes: int = 0

    def to_dict(self) -> Dict[str, float]:
        d = {
            "task_success_rate": self.task_success_rate,
            "action_efficiency": self.action_efficiency,
            "coordination_score": self.coordination_score,
            "avg_turn_count": self.avg_turn_count,
            "delegation_accuracy": self.delegation_accuracy,
            "avg_bertscore": self.avg_bertscore,
            "avg_coherence": self.avg_coherence,
            "functional_fluency_gap": self.functional_fluency_gap,
            "num_episodes": self.num_episodes,
        }
        for complexity, metrics in self.per_complexity.items():
            for k, v in metrics.items():
                d[f"{complexity}/{k}"] = v
        return d


class FunctionalMetrics:
    """Computes task-completion and coordination metrics."""

    @staticmethod
    def task_success_rate(results: List[EpisodeResult]) -> float:
        """Fraction of episodes where the task was completed."""
        if not results:
            return 0.0
        return sum(1 for r in results if r.success) / len(results)

    @staticmethod
    def action_efficiency(results: List[EpisodeResult]) -> float:
        """
        Average ratio of optimal steps to actual steps taken.
        1.0 = perfect efficiency; <1.0 = took more steps than necessary.
        """
        if not results:
            return 0.0
        efficiencies = [
            r.optimal_steps / max(r.steps_taken, 1)
            for r in results
        ]
        return float(np.mean(efficiencies))

    @staticmethod
    def coordination_score(results: List[EpisodeResult]) -> float:
        """
        1 - (redundant + conflicting actions) / total actions.
        Higher is better. 1.0 = perfect coordination.
        """
        if not results:
            return 0.0
        scores = []
        for r in results:
            total = r.steps_taken
            if total == 0:
                scores.append(1.0)
                continue
            bad = r.redundant_actions + r.conflicting_actions
            scores.append(1.0 - bad / total)
        return float(np.mean(scores))

    @staticmethod
    def delegation_accuracy(results: List[EpisodeResult]) -> float:
        """Fraction of delegated actions assigned to a capable agent."""
        total_correct = sum(r.delegation_correct for r in results)
        total_delegations = sum(r.delegation_total for r in results)
        if total_delegations == 0:
            return 1.0
        return total_correct / total_delegations

    @staticmethod
    def avg_turn_count(results: List[EpisodeResult]) -> float:
        """Average dialogue turns to task completion."""
        if not results:
            return 0.0
        return float(np.mean([r.steps_taken for r in results]))


class FluencyMetrics:
    """Computes conversational quality metrics."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._bertscore_model = None
        self._nli_model = None

    def _load_bertscore(self):
        """Lazy-load BERTScore."""
        if self._bertscore_model is None:
            try:
                from bert_score import BERTScorer
                self._bertscore_model = BERTScorer(
                    model_type="microsoft/deberta-xlarge-mnli",
                    device=self.device,
                )
            except ImportError:
                print("Warning: bert_score not installed. Using dummy scorer.")
                self._bertscore_model = "dummy"

    def bertscore(
        self,
        candidates: List[str],
        references: List[str],
    ) -> float:
        """Compute average BERTScore F1 between candidates and references."""
        self._load_bertscore()
        if self._bertscore_model == "dummy":
            return 0.5  # Placeholder

        P, R, F1 = self._bertscore_model.score(candidates, references)
        return float(F1.mean())

    def coherence_nli(
        self,
        utterances: List[str],
    ) -> float:
        """
        Compute inter-turn coherence using NLI.

        For each consecutive pair of utterances, check if the second entails
        or is consistent with the first. Average entailment probability.
        """
        if len(utterances) < 2:
            return 1.0

        try:
            from transformers import pipeline
            if self._nli_model is None:
                self._nli_model = pipeline(
                    "text-classification",
                    model="cross-encoder/nli-deberta-v3-base",
                    device=self.device if self.device != "cpu" else -1,
                )
        except ImportError:
            return 0.5  # Placeholder

        scores = []
        for i in range(len(utterances) - 1):
            premise = utterances[i]
            hypothesis = utterances[i + 1]

            try:
                result = self._nli_model(f"{premise} [SEP] {hypothesis}")
                # Extract entailment score
                if isinstance(result, list):
                    result = result[0]
                if result.get("label") == "ENTAILMENT":
                    scores.append(result.get("score", 0.5))
                elif result.get("label") == "NEUTRAL":
                    scores.append(0.5)
                else:  # CONTRADICTION
                    scores.append(0.0)
            except Exception:
                scores.append(0.5)

        return float(np.mean(scores)) if scores else 1.0


class RedundancyDetector:
    """
    Detects redundant actions across agents.

    An action is redundant if two agents perform the same action on the same
    target within a short temporal window.
    """

    @staticmethod
    def detect(
        agent_actions: Dict[int, List[str]],
        window_size: int = 3,
    ) -> int:
        """
        Count redundant action pairs.

        Args:
            agent_actions: agent_id → list of action strings.
            window_size: temporal window for overlap detection.
        Returns:
            Number of redundant action pairs.
        """
        redundant = 0
        agent_ids = sorted(agent_actions.keys())

        for i, aid1 in enumerate(agent_ids):
            for aid2 in agent_ids[i + 1:]:
                actions1 = agent_actions[aid1]
                actions2 = agent_actions[aid2]

                for t1, a1 in enumerate(actions1):
                    for t2, a2 in enumerate(actions2):
                        if abs(t1 - t2) <= window_size and a1 == a2 and a1 != "nop":
                            redundant += 1

        return redundant


class ConflictDetector:
    """
    Detects conflicting actions across agents.

    Actions conflict when they have opposing effects (e.g., one agent opens
    a door while another closes it).
    """

    CONFLICT_PAIRS = {
        ("open", "close"),
        ("pick_up", "put_down"),
        ("heat", "cool"),
        ("navigate", "navigate"),  # Same target = conflict for embodied agents
    }

    @staticmethod
    def detect(
        agent_actions: Dict[int, List[str]],
        window_size: int = 2,
    ) -> int:
        """Count conflicting action pairs."""
        conflicts = 0
        agent_ids = sorted(agent_actions.keys())

        for i, aid1 in enumerate(agent_ids):
            for aid2 in agent_ids[i + 1:]:
                actions1 = agent_actions[aid1]
                actions2 = agent_actions[aid2]

                for t1, a1 in enumerate(actions1):
                    for t2, a2 in enumerate(actions2):
                        if abs(t1 - t2) <= window_size:
                            a1_base = a1.split("(")[0]
                            a2_base = a2.split("(")[0]
                            if (a1_base, a2_base) in ConflictDetector.CONFLICT_PAIRS or \
                               (a2_base, a1_base) in ConflictDetector.CONFLICT_PAIRS:
                                conflicts += 1

        return conflicts


class MultiAgentEvaluator:
    """
    Full evaluation pipeline for multi-agent dialogue systems.

    Computes all metrics and produces a report separating functional
    success from conversational fluency.
    """

    def __init__(self, device: str = "cpu"):
        self.functional = FunctionalMetrics()
        self.fluency = FluencyMetrics(device=device)
        self.redundancy_detector = RedundancyDetector()
        self.conflict_detector = ConflictDetector()

    def evaluate_episode(
        self,
        result: EpisodeResult,
    ) -> EpisodeResult:
        """Enrich an episode result with detected redundancies and conflicts."""
        result.redundant_actions = self.redundancy_detector.detect(result.agent_actions)
        result.conflicting_actions = self.conflict_detector.detect(result.agent_actions)
        return result

    def evaluate(
        self,
        results: List[EpisodeResult],
        compute_fluency: bool = True,
    ) -> EvaluationReport:
        """
        Run full evaluation on a list of episode results.

        Args:
            results: List of EpisodeResult from running the system.
            compute_fluency: Whether to compute (expensive) fluency metrics.
        Returns:
            EvaluationReport with all metrics.
        """
        # Enrich with redundancy/conflict counts
        results = [self.evaluate_episode(r) for r in results]

        report = EvaluationReport(num_episodes=len(results))

        # Functional metrics
        report.task_success_rate = self.functional.task_success_rate(results)
        report.action_efficiency = self.functional.action_efficiency(results)
        report.coordination_score = self.functional.coordination_score(results)
        report.delegation_accuracy = self.functional.delegation_accuracy(results)
        report.avg_turn_count = self.functional.avg_turn_count(results)

        # Fluency metrics
        if compute_fluency:
            all_generated = []
            all_reference = []
            all_utterances_for_coherence = []

            for r in results:
                if r.generated_utterances:
                    all_utterances_for_coherence.append(r.generated_utterances)
                    if r.reference_utterances:
                        all_generated.extend(r.generated_utterances)
                        all_reference.extend(r.reference_utterances)

            if all_generated and all_reference:
                report.avg_bertscore = self.fluency.bertscore(all_generated, all_reference)

            if all_utterances_for_coherence:
                coherence_scores = [
                    self.fluency.coherence_nli(utts)
                    for utts in all_utterances_for_coherence
                ]
                report.avg_coherence = float(np.mean(coherence_scores))

        # The Gap: functional success - fluency (our key metric)
        report.functional_fluency_gap = report.task_success_rate - report.avg_bertscore

        # Per-complexity breakdown
        complexity_groups: Dict[str, List[EpisodeResult]] = {}
        for r in results:
            if r.difficulty not in complexity_groups:
                complexity_groups[r.difficulty] = []
            complexity_groups[r.difficulty].append(r)

        for complexity, group in complexity_groups.items():
            report.per_complexity[complexity] = {
                "task_success_rate": self.functional.task_success_rate(group),
                "action_efficiency": self.functional.action_efficiency(group),
                "coordination_score": self.functional.coordination_score(group),
                "avg_turn_count": self.functional.avg_turn_count(group),
                "num_episodes": len(group),
            }

        return report
