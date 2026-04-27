"""
Core dataclasses for the benchmark framework.
All scoring is binary (pass/fail). No scales, no vibes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Attachment:
    type: str  # image | document | video | audio
    path: str  # local path or URL


@dataclass
class Check:
    """Single binary check definition from YAML."""
    type: str  # validator name (contains, judge, max_tokens, ...)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    id: str
    complexity: str  # tutorial | extended | phd
    modality: list[str]  # text, image, document, video, audio
    prompt: str
    checks: list[Check]
    tags: list[str] = field(default_factory=list)
    attachments: list[Attachment] = field(default_factory=list)
    ground_truth: str | None = None  # for judge calibration

    @property
    def is_multimodal(self) -> bool:
        return any(m != "text" for m in self.modality)


@dataclass
class Suite:
    id: str
    name: str
    description: str = ""
    tasks: list[str] = field(default_factory=list)  # explicit task IDs
    task_pattern: str = ""  # glob pattern e.g. "logic/*"
    tags_filter: list[str] = field(default_factory=list)  # filter by tags


@dataclass
class TaskContext:
    """Everything a Validator can see."""
    task: Task
    prompt: str  # final prompt sent (may include [media:...])
    response: str
    attachments: list[Attachment] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    execution_time: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    sandbox_state: dict | None = None


@dataclass
class CheckResult:
    validator_name: str
    passed: bool
    detail: str = ""


@dataclass
class TaskResult:
    task_id: str
    complexity: str
    tags: list[str]
    checks: list[CheckResult] = field(default_factory=list)
    prompt: str = ""
    response: str = ""
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    judge_cost: float = 0.0
    tool_calls: list[dict] = field(default_factory=list)  # [{"name","args","result"}]
    working_history: list[dict] = field(default_factory=list)  # [{"role","content"}]

    @property
    def score(self) -> float:
        """passed / total. 1.0 = perfect, 0.0 = all failed."""
        if not self.checks:
            return 0.0
        return sum(1 for c in self.checks if c.passed) / len(self.checks)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_checks(self) -> int:
        return len(self.checks)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "complexity": self.complexity,
            "tags": self.tags,
            "score": self.score,
            "passed": self.passed,
            "total_checks": self.total_checks,
            "checks": [
                {"validator": c.validator_name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
            "response": self.response,
            "latency_ms": self.latency_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost": self.cost,
            "tool_calls": self.tool_calls,
        }


@dataclass
class JudgeProfile:
    model: str = ""
    disqualified: bool = False
    batch_sizes: dict[str, int] = field(default_factory=dict)  # complexity → max batch
    accuracy: dict[str, float] = field(default_factory=dict)  # complexity → accuracy

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "disqualified": self.disqualified,
            "batch_sizes": self.batch_sizes,
            "accuracy": self.accuracy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> JudgeProfile:
        return cls(
            model=d.get("model", ""),
            disqualified=d.get("disqualified", False),
            batch_sizes=d.get("batch_sizes", {}),
            accuracy=d.get("accuracy", {}),
        )


@dataclass
class Report:
    model_id: str
    suite_id: str
    mode: str = "standard"
    results: list[TaskResult] = field(default_factory=list)
    total_time_s: float = 0.0
    total_cost: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    judge_profile: JudgeProfile | None = None
    judge_cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    @property
    def total_score(self) -> float:
        """Average of all task scores. Pure fact, no weighting."""
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def total_tokens(self) -> int:
        return self.total_tokens_in + self.total_tokens_out

    def scores_by_tag(self) -> dict[str, float]:
        """Average score per tag (acts like 'dimensions')."""
        tag_scores: dict[str, list[float]] = {}
        for r in self.results:
            for tag in r.tags:
                tag_scores.setdefault(tag, []).append(r.score)
        return {t: sum(s) / len(s) for t, s in tag_scores.items() if s}

    def scores_by_complexity(self) -> dict[str, float]:
        by_c: dict[str, list[float]] = {}
        for r in self.results:
            by_c.setdefault(r.complexity, []).append(r.score)
        return {c: sum(s) / len(s) for c, s in by_c.items() if s}

    def to_dict(self) -> dict:
        """Dashboard-compatible output format."""
        tag_scores = self.scores_by_tag()
        # Convert 0-1 scores to 0-100 for dashboard
        dimensions = {t: round(s * 100, 1) for t, s in tag_scores.items()}

        num_results = len(self.results) or 1

        # Build probe details list — used by both "results" and "probe_details"
        # Dashboard score scale: >=1 positive, 0..1 neutral, <0 negative
        # We map task score 0-1 → -1..+1 so: 100% checks → +1, 50% → 0, 0% → -1
        _probe_list = [
            {
                "probe_id": r.task_id,
                "complexity": r.complexity,
                "prompt": r.prompt[:1000],
                "response": r.response[:500],
                "scores": {t: round(r.score * 2 - 1, 2) for t in r.tags},
                "score": round(r.score * 2 - 1, 2),
                "latency_ms": r.latency_ms,
                "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out,
                "cost": r.cost,
                "flags": [],
                "checks": [
                    {"name": c.validator_name, "passed": c.passed, "detail": c.detail}
                    for c in r.checks
                ],
                "tool_calls": [
                    {
                        "name": tc.get("name", "?"),
                        "args": {k: str(v)[:200] for k, v in (tc.get("args") or {}).items()},
                        "result": str(tc.get("result", ""))[:300],
                    }
                    for tc in r.tool_calls
                ],
                "working_history": [
                    {
                        "role": msg.get("role", "?"),
                        "content": (str(msg.get("content", ""))[:500]
                                    if isinstance(msg.get("content"), str)
                                    else str(msg.get("content", ""))[:500]),
                    }
                    for msg in r.working_history
                ] if r.working_history else [],
            }
            for r in self.results
        ]

        return {
            "model": self.model_id,
            "mode": self.mode,
            "total": round(self.total_score * 100, 1),
            "total_raw": round(self.total_score * 100, 1),
            "flag_penalty": 0,
            "dimensions": dimensions,
            "persona": {},  # no persona in new system
            "flags": [],
            "flag_details": [],
            "probes": len(self.results),
            "cost": {
                "total_cost": self.total_cost,
                "total_tokens": self.total_tokens,
                "tokens_in": self.total_tokens_in,
                "tokens_out": self.total_tokens_out,
                "total_time_s": self.total_time_s,
                "cost_per_probe": self.total_cost / num_results,
                "time_per_probe_s": self.total_time_s / num_results,
                "tokens_per_probe": self.total_tokens / num_results,
            },
            "results": _probe_list,
            # Dashboard reads probe_details for the detail view
            "probe_details": _probe_list,
            "timestamp": self.timestamp.isoformat(),
            "suite": self.suite_id,
            "judge_profile": self.judge_profile.to_dict() if self.judge_profile else None,
            "judge_cost": self.judge_cost,
            "complexity_scores": {
                c: round(s * 100, 1) for c, s in self.scores_by_complexity().items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> Report:
        """Reconstruct from serialized dict (for loading saved reports)."""
        r = cls(
            model_id=d.get("model", ""),
            suite_id=d.get("suite", ""),
            mode=d.get("mode", ""),
            total_time_s=d.get("cost", {}).get("total_time_s", 0),
            total_cost=d.get("cost", {}).get("total_cost", 0),
            total_tokens_in=d.get("cost", {}).get("tokens_in", 0),
            total_tokens_out=d.get("cost", {}).get("tokens_out", 0),
            judge_cost=d.get("judge_cost", 0),
        )
        ts = d.get("timestamp")
        if ts:
            try:
                r.timestamp = datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                pass
        jp = d.get("judge_profile")
        if jp:
            r.judge_profile = JudgeProfile.from_dict(jp)
        return r

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> Report:
        return cls.from_dict(json.loads(Path(path).read_text()))
