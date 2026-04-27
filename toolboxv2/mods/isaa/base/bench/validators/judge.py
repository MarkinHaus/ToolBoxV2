"""
Judge validator — delegates binary yes/no questions to an LLM via isaa.format_class.
"""

from __future__ import annotations

import asyncio
import os

from pydantic import BaseModel

from toolboxv2.mods.isaa.base.bench.core import CheckResult, TaskContext
from toolboxv2.mods.isaa.base.bench.validators import Validator, register


class JudgeAnswer(BaseModel):
    answer: bool


def _extract_answer(result) -> bool | None:
    """Extract bool answer from format_class result, handling all return shapes."""
    if result is None:
        return None

    # format_class returns a wrapper — call .get() first
    if hasattr(result, "get") and callable(result.get) and not isinstance(result, dict):
        result = result.get()

    if result is None:
        return None

    # Direct BaseModel instance
    if isinstance(result, JudgeAnswer):
        return result.answer

    # Dict with "answer" key
    if isinstance(result, dict):
        val = result.get("answer")
        if isinstance(val, bool):
            return val
        if val is not None:
            return str(val).strip().lower() in ("true", "yes", "1", "ja")

    # Raw string fallback
    raw = str(result).strip().lower()
    if raw in ("true", "yes", "1", "ja"):
        return True
    if raw in ("false", "no", "0", "nein"):
        return False

    return None


@register("judge")
class JudgeValidator(Validator):
    """Ask a judge LLM a yes/no question about the response.

    YAML usage:
        - type: judge
          question: "Does the answer correctly identify the capital?"
    """
    name = "judge"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        question = self.params["question"]

        try:
            from toolboxv2 import get_app
            isaa = get_app().get_mod("isaa")
        except Exception as e:
            return CheckResult("judge", False, f"ISAA unavailable: {e}")

        task_prompt = (
            f"You are a strict binary judge. Answer ONLY true or false.\n\n"
            f"## Prompt given to the model:\n{ctx.prompt}\n\n"
            f"## Model response:\n{ctx.response}\n\n"
            f"## Question:\n{question}\n\n"
            f"Answer true or false."
        )
        agent = await isaa.get_agent("BenchJudge")

        agent.amd.complex_llm_model = agent.amd.fast_llm_model
        agent.amd.fast_llm_model = os.getenv("BLITZMODEL", agent.amd.fast_llm_model)

        try:
            result = await asyncio.wait_for(
                isaa.format_class(
                    format_schema=JudgeAnswer,
                    task=task_prompt,
                    agent_name="BenchJudge",
                ),
                timeout=self.params.get("timeout", 30),
            )

            answer = _extract_answer(result)
            if answer is None:
                return CheckResult("judge", False, f"judge returned unparseable: {result}")

            return CheckResult("judge", answer, f"judge={'true' if answer else 'false'}: {question}")

        except asyncio.TimeoutError:
            return CheckResult("judge", False, f"judge timeout ({self.params.get('timeout', 30)}s)")
        except Exception as e:
            return CheckResult("judge", False, f"judge error: {e}")


@register("judge_compare")
class JudgeCompareValidator(Validator):
    """Compare response against a ground truth answer via judge.

    YAML usage:
        - type: judge_compare
          ground_truth: "The capital of France is Paris"
          question: "Does the response convey the same factual answer?"
    """
    name = "judge_compare"

    async def validate(self, ctx: TaskContext) -> CheckResult:
        ground_truth = self.params["ground_truth"]
        question = self.params.get("question", "Does the response match the ground truth?")

        try:
            from toolboxv2 import get_app
            isaa = get_app().get_mod("isaa")
        except Exception as e:
            return CheckResult("judge_compare", False, f"ISAA unavailable: {e}")

        task_prompt = (
            f"You are a strict binary judge. Answer ONLY true or false.\n\n"
            f"## Ground truth:\n{ground_truth}\n\n"
            f"## Model response:\n{ctx.response}\n\n"
            f"## Question:\n{question}\n\n"
            f"Answer true or false."
        ).encode("utf-8").decode("utf-8")

        agent = await isaa.get_agent("BenchJudge")

        agent.amd.complex_llm_model = agent.amd.fast_llm_model
        agent.amd.fast_llm_model = os.getenv("BLITZMODEL", agent.amd.fast_llm_model)


        try:
            result = await asyncio.wait_for(
                isaa.format_class(
                    format_schema=JudgeAnswer,
                    task=task_prompt,
                    agent_name="BenchJudge",
                ),
                timeout=self.params.get("timeout", 30),
            )

            answer = _extract_answer(result)
            if answer is None:
                return CheckResult("judge_compare", False, f"judge returned unparseable: {result}")

            return CheckResult("judge_compare", answer, f"judge={'true' if answer else 'false'}")

        except asyncio.TimeoutError:
            return CheckResult("judge_compare", False, f"judge timeout ({self.params.get('timeout', 30)}s)")
        except Exception as e:
            return CheckResult("judge_compare", False, f"judge error: {e}")
