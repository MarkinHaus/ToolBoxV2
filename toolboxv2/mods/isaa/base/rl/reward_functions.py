"""
Reward Functions for FlowAgent RL Training

Verifiable binary and soft rewards that look into what the agent
actually did - not just the final output. Includes code execution,
tool success, syntax validation, and learned rewards.
"""

import ast
import json
import subprocess
import tempfile
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from abc import ABC, abstractmethod


@dataclass
class RewardResult:
    """Result from a reward function evaluation"""
    score: float  # 0.0 - 1.0
    is_binary: bool  # True if this is a pass/fail reward
    details: dict = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_binary(self, threshold: float = 0.5) -> int:
        """Convert to binary reward (0 or 1)"""
        return 1 if self.score >= threshold else 0


class BaseReward(ABC):
    """Abstract base class for reward functions"""
    
    name: str = "base_reward"
    weight: float = 1.0
    is_binary: bool = True
    
    @abstractmethod
    def compute(self, trace) -> RewardResult:
        """
        Compute reward for an execution trace.
        
        Args:
            trace: ExecutionTrace object with full execution details
        
        Returns:
            RewardResult with score and details
        """
        pass
    
    def __call__(self, trace) -> RewardResult:
        return self.compute(trace)


class CodeExecutionReward(BaseReward):
    """
    Reward for successful code execution.
    
    Actually runs the code and checks if it executes without errors.
    This is a verifiable binary reward.
    """
    
    name = "code_execution"
    weight = 2.0
    is_binary = True
    
    def __init__(self, timeout: int = 30, sandbox: bool = True):
        """
        Args:
            timeout: Max execution time in seconds
            sandbox: Use restricted execution environment
        """
        self.timeout = timeout
        self.sandbox = sandbox
    
    def compute(self, trace) -> RewardResult:
        """Check if code in the response executes successfully"""
        
        # Extract code blocks from response
        code_blocks = self._extract_code_blocks(trace.final_response)
        
        if not code_blocks:
            # No code to execute - neutral score
            return RewardResult(
                score=0.5,
                is_binary=False,
                details={"reason": "no_code_found"}
            )
        
        # Execute each code block
        results = []
        for lang, code in code_blocks:
            if lang in ["python", "py", ""]:
                success, output, error = self._execute_python(code)
                results.append({
                    "language": "python",
                    "success": success,
                    "output": output[:500] if output else "",
                    "error": error[:500] if error else ""
                })
            elif lang in ["bash", "sh", "shell"]:
                success, output, error = self._execute_shell(code)
                results.append({
                    "language": "shell",
                    "success": success,
                    "output": output[:500] if output else "",
                    "error": error[:500] if error else ""
                })
        
        if not results:
            return RewardResult(score=0.5, is_binary=False, details={"reason": "no_executable_code"})
        
        # Score: ratio of successful executions
        successes = sum(1 for r in results if r["success"])
        score = successes / len(results)
        
        return RewardResult(
            score=score,
            is_binary=True,
            details={
                "total_blocks": len(results),
                "successful": successes,
                "results": results
            }
        )
    
    def _extract_code_blocks(self, text: str) -> list[tuple[str, str]]:
        """Extract code blocks from markdown-style text"""
        blocks = []
        
        # Pattern for ```language\ncode\n```
        pattern = r"```(\w*)\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        for lang, code in matches:
            code = code.strip()
            if code:
                blocks.append((lang.lower(), code))
        
        return blocks
    
    def _execute_python(self, code: str) -> tuple[bool, str, str]:
        """Execute Python code safely"""
        try:
            # First check syntax
            ast.parse(code)
        except SyntaxError as e:
            return False, "", f"SyntaxError: {e}"
        
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8"
            ) as f:
                f.write(code)
                temp_path = f.name
            
            # Execute with timeout
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir()
            )
            
            os.unlink(temp_path)
            
            if result.returncode == 0:
                return True, result.stdout, ""
            else:
                return False, result.stdout, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "", "Execution timeout"
        except Exception as e:
            return False, "", str(e)
    
    def _execute_shell(self, code: str) -> tuple[bool, str, str]:
        """Execute shell commands safely"""
        if self.sandbox:
            # Restrict dangerous commands
            dangerous = ["rm -rf", "dd ", "mkfs", ":(){", "fork bomb"]
            for d in dangerous:
                if d in code.lower():
                    return False, "", f"Blocked dangerous command: {d}"
        
        try:
            result = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir()
            )
            
            if result.returncode == 0:
                return True, result.stdout, ""
            else:
                return False, result.stdout, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "", "Execution timeout"
        except Exception as e:
            return False, "", str(e)


class SyntaxValidationReward(BaseReward):
    """
    Reward for syntactically correct code.
    
    Checks if code can be parsed without execution.
    Fast binary reward.
    """
    
    name = "syntax_validation"
    weight = 1.0
    is_binary = True
    
    def compute(self, trace) -> RewardResult:
        """Check syntax of all code blocks"""
        
        code_blocks = self._extract_code_blocks(trace.final_response)
        
        if not code_blocks:
            return RewardResult(score=0.5, is_binary=False, details={"reason": "no_code"})
        
        valid_count = 0
        errors = []
        
        for lang, code in code_blocks:
            if lang in ["python", "py", ""]:
                try:
                    ast.parse(code)
                    valid_count += 1
                except SyntaxError as e:
                    errors.append({"lang": lang, "error": str(e)})
            elif lang in ["json"]:
                try:
                    json.loads(code)
                    valid_count += 1
                except json.JSONDecodeError as e:
                    errors.append({"lang": lang, "error": str(e)})
            else:
                # Assume valid for other languages
                valid_count += 1
        
        score = valid_count / len(code_blocks) if code_blocks else 0.5
        
        return RewardResult(
            score=score,
            is_binary=True,
            details={
                "total": len(code_blocks),
                "valid": valid_count,
                "errors": errors
            }
        )
    
    def _extract_code_blocks(self, text: str) -> list[tuple[str, str]]:
        """Extract code blocks from text"""
        pattern = r"```(\w*)\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [(lang.lower(), code.strip()) for lang, code in matches if code.strip()]


class ToolSuccessReward(BaseReward):
    """
    Reward based on actual tool call success.
    
    Looks at what tools the agent called and whether they succeeded.
    This directly examines agent behavior, not just output.
    """
    
    name = "tool_success"
    weight = 2.0
    is_binary = True
    
    def compute(self, trace) -> RewardResult:
        """Compute reward from tool call success rate"""
        
        tool_calls = trace.tool_calls
        
        if not tool_calls:
            # No tools used - check if task needed tools
            if self._task_likely_needs_tools(trace.user_query):
                return RewardResult(
                    score=0.3,
                    is_binary=False,
                    details={"reason": "no_tools_but_likely_needed"}
                )
            return RewardResult(score=0.5, is_binary=False, details={"reason": "no_tools_needed"})
        
        successful = sum(1 for tc in tool_calls if tc.success)
        total = len(tool_calls)
        
        # Bonus for using appropriate tools
        appropriate_tools = self._count_appropriate_tools(trace.user_query, tool_calls)
        
        base_score = successful / total
        appropriateness_bonus = 0.1 * (appropriate_tools / total) if total > 0 else 0
        
        score = min(1.0, base_score + appropriateness_bonus)
        
        return RewardResult(
            score=score,
            is_binary=True,
            details={
                "total_calls": total,
                "successful": successful,
                "appropriate_tools": appropriate_tools,
                "tool_names": [tc.tool_name for tc in tool_calls]
            }
        )
    
    def _task_likely_needs_tools(self, query: str) -> bool:
        """Heuristic: does this query likely need tools?"""
        tool_indicators = [
            "search", "find", "look up", "execute", "run",
            "create file", "write to", "read from", "calculate",
            "fetch", "download", "check", "analyze"
        ]
        query_lower = query.lower()
        return any(ind in query_lower for ind in tool_indicators)
    
    def _count_appropriate_tools(self, query: str, tool_calls: list) -> int:
        """Count tools that seem appropriate for the query"""
        query_lower = query.lower()
        appropriate = 0
        
        tool_query_mapping = {
            "search": ["search", "find", "look"],
            "file": ["file", "read", "write", "create"],
            "execute": ["run", "execute", "shell"],
            "web": ["fetch", "download", "url", "http"],
        }
        
        for tc in tool_calls:
            tool_lower = tc.tool_name.lower()
            for tool_type, keywords in tool_query_mapping.items():
                if tool_type in tool_lower:
                    if any(kw in query_lower for kw in keywords):
                        appropriate += 1
                        break
        
        return appropriate


class TaskCompletionReward(BaseReward):
    """
    Reward based on task completion status.
    
    Checks if the agent actually completed the tasks it created.
    """
    
    name = "task_completion"
    weight = 1.5
    is_binary = True
    
    def compute(self, trace) -> RewardResult:
        """Check task completion rate"""
        
        created = len(trace.tasks_created)
        completed = len(trace.tasks_completed)
        failed = len(trace.tasks_failed)
        
        if created == 0:
            return RewardResult(score=0.5, is_binary=False, details={"reason": "no_tasks"})
        
        # Completion rate
        completion_rate = completed / created
        
        # Penalty for failures
        failure_penalty = 0.2 * (failed / created) if created > 0 else 0
        
        score = max(0.0, completion_rate - failure_penalty)
        
        return RewardResult(
            score=score,
            is_binary=True,
            details={
                "created": created,
                "completed": completed,
                "failed": failed,
                "completion_rate": completion_rate
            }
        )


class EfficiencyReward(BaseReward):
    """
    Soft reward for efficiency.
    
    Rewards concise, efficient responses that don't waste tokens
    or make unnecessary tool calls.
    """
    
    name = "efficiency"
    weight = 0.5
    is_binary = False
    
    def __init__(
        self,
        max_tokens: int = 2000,
        max_tool_calls: int = 10,
        max_reasoning_steps: int = 15
    ):
        self.max_tokens = max_tokens
        self.max_tool_calls = max_tool_calls
        self.max_reasoning_steps = max_reasoning_steps
    
    def compute(self, trace) -> RewardResult:
        """Compute efficiency score"""
        
        scores = []
        
        # Token efficiency (fewer tokens for same result = better)
        total_tokens = trace.total_tokens_in + trace.total_tokens_out
        token_score = max(0.0, 1.0 - (total_tokens / self.max_tokens))
        scores.append(("tokens", token_score, 0.4))
        
        # Tool call efficiency
        tool_count = len(trace.tool_calls)
        if tool_count > 0:
            # Reward fewer calls, but not zero
            tool_score = max(0.0, 1.0 - (tool_count / self.max_tool_calls))
            scores.append(("tool_calls", tool_score, 0.3))
        
        # Reasoning efficiency
        reasoning_count = len(trace.reasoning_steps)
        if reasoning_count > 0:
            reasoning_score = max(0.0, 1.0 - (reasoning_count / self.max_reasoning_steps))
            scores.append(("reasoning", reasoning_score, 0.3))
        
        # Weighted average
        total_weight = sum(w for _, _, w in scores)
        if total_weight > 0:
            score = sum(s * w for _, s, w in scores) / total_weight
        else:
            score = 0.5
        
        return RewardResult(
            score=score,
            is_binary=False,
            details={
                "total_tokens": total_tokens,
                "tool_calls": tool_count,
                "reasoning_steps": reasoning_count,
                "component_scores": {name: s for name, s, _ in scores}
            }
        )


class FormatComplianceReward(BaseReward):
    """
    Reward for following output format requirements.
    
    Checks if the response follows expected formatting patterns
    (NO XML - plain text focus).
    """
    
    name = "format_compliance"
    weight = 1.0
    is_binary = True
    
    def __init__(self, forbidden_patterns: list[str] = None):
        self.forbidden_patterns = forbidden_patterns or [
            r"<[a-zA-Z][^>]*>",  # XML/HTML tags
            r"</[a-zA-Z]+>",     # Closing tags
        ]
    
    def compute(self, trace) -> RewardResult:
        """Check format compliance"""
        
        response = trace.final_response
        violations = []
        
        # Check forbidden patterns
        for pattern in self.forbidden_patterns:
            matches = re.findall(pattern, response)
            if matches:
                violations.extend(matches[:5])  # Limit to 5 examples
        
        if violations:
            # Penalize based on number of violations
            penalty = min(0.5, len(violations) * 0.1)
            score = max(0.0, 1.0 - penalty)
        else:
            score = 1.0
        
        return RewardResult(
            score=score,
            is_binary=True,
            details={
                "violations": violations,
                "violation_count": len(violations)
            }
        )


class LearnedReward(BaseReward):
    """
    Learned reward from manual labels.
    
    Uses a simple pattern matching model trained on
    manually labeled examples to predict reward.
    """
    
    name = "learned_reward"
    weight = 1.0
    is_binary = False
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.patterns = {
            "positive": [],
            "negative": []
        }
        self._load_patterns()
    
    def _load_patterns(self):
        """Load learned patterns from file"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                with open(self.model_path, "r") as f:
                    self.patterns = json.load(f)
            except:
                pass
    
    def save_patterns(self):
        """Save learned patterns"""
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, "w") as f:
                json.dump(self.patterns, f, indent=2)
    
    def learn_from_traces(self, traces: list, min_examples: int = 10):
        """
        Learn patterns from labeled traces.
        
        Simple approach: extract n-grams and tool patterns
        from positive and negative examples.
        """
        positive_traces = [t for t in traces if t.label == True]
        negative_traces = [t for t in traces if t.label == False]
        
        if len(positive_traces) < min_examples or len(negative_traces) < min_examples:
            print(f"Not enough examples: {len(positive_traces)} positive, {len(negative_traces)} negative")
            return
        
        # Extract patterns
        self.patterns["positive"] = self._extract_patterns(positive_traces)
        self.patterns["negative"] = self._extract_patterns(negative_traces)
        
        self.save_patterns()
        print(f"Learned {len(self.patterns['positive'])} positive and {len(self.patterns['negative'])} negative patterns")
    
    def _extract_patterns(self, traces: list) -> list[dict]:
        """Extract patterns from traces"""
        patterns = []
        
        # Tool usage patterns
        tool_counts = {}
        for trace in traces:
            for tc in trace.tool_calls:
                tool_counts[tc.tool_name] = tool_counts.get(tc.tool_name, 0) + 1
        
        for tool, count in tool_counts.items():
            if count >= 3:  # Minimum frequency
                patterns.append({
                    "type": "tool_usage",
                    "tool": tool,
                    "frequency": count / len(traces)
                })
        
        # Success rate patterns
        success_rates = []
        for trace in traces:
            if trace.tool_calls:
                rate = sum(1 for tc in trace.tool_calls if tc.success) / len(trace.tool_calls)
                success_rates.append(rate)
        
        if success_rates:
            patterns.append({
                "type": "success_rate",
                "avg": sum(success_rates) / len(success_rates)
            })
        
        return patterns
    
    def compute(self, trace) -> RewardResult:
        """Compute reward using learned patterns"""
        
        if not self.patterns["positive"] and not self.patterns["negative"]:
            return RewardResult(score=0.5, is_binary=False, details={"reason": "no_patterns_learned"})
        
        positive_score = self._match_patterns(trace, self.patterns["positive"])
        negative_score = self._match_patterns(trace, self.patterns["negative"])
        
        # Combine scores
        if positive_score + negative_score > 0:
            score = positive_score / (positive_score + negative_score)
        else:
            score = 0.5
        
        return RewardResult(
            score=score,
            is_binary=False,
            details={
                "positive_match": positive_score,
                "negative_match": negative_score
            }
        )
    
    def _match_patterns(self, trace, patterns: list) -> float:
        """Calculate pattern match score"""
        if not patterns:
            return 0.0
        
        matches = 0
        for pattern in patterns:
            if pattern["type"] == "tool_usage":
                for tc in trace.tool_calls:
                    if tc.tool_name == pattern["tool"]:
                        matches += pattern["frequency"]
            elif pattern["type"] == "success_rate":
                if trace.tool_calls:
                    rate = sum(1 for tc in trace.tool_calls if tc.success) / len(trace.tool_calls)
                    # Reward if close to learned average
                    diff = abs(rate - pattern["avg"])
                    if diff < 0.2:
                        matches += 1.0 - diff
        
        return matches


class RewardEngine:
    """
    Combines multiple reward functions for GRPO training.
    
    Provides weighted combination of rewards and normalization
    for group-based advantage computation.
    """
    
    def __init__(self, rewards: list[BaseReward] = None):
        """
        Initialize reward engine with reward functions.
        
        Args:
            rewards: List of reward functions (uses defaults if None)
        """
        if rewards is None:
            rewards = [
                CodeExecutionReward(),
                SyntaxValidationReward(),
                ToolSuccessReward(),
                TaskCompletionReward(),
                EfficiencyReward(),
                FormatComplianceReward(),
            ]
        
        self.rewards = rewards
        self.reward_history = []
    
    def compute_all(self, trace) -> dict[str, RewardResult]:
        """Compute all rewards for a trace"""
        results = {}
        for reward in self.rewards:
            try:
                results[reward.name] = reward.compute(trace)
            except Exception as e:
                results[reward.name] = RewardResult(
                    score=0.0,
                    is_binary=reward.is_binary,
                    error=str(e)
                )
        return results
    
    def compute_combined(self, trace) -> float:
        """Compute weighted combined reward"""
        results = self.compute_all(trace)
        
        total_weight = sum(r.weight for r in self.rewards)
        weighted_sum = 0.0
        
        for reward in self.rewards:
            if reward.name in results:
                weighted_sum += results[reward.name].score * reward.weight
        
        combined = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Track for normalization
        self.reward_history.append(combined)
        
        return combined
    
    def compute_for_group(self, traces: list) -> list[float]:
        """
        Compute rewards for a group of traces (for GRPO).
        
        Returns normalized rewards suitable for advantage computation.
        """
        raw_rewards = [self.compute_combined(trace) for trace in traces]
        
        # Normalize within group
        if len(raw_rewards) > 1:
            mean = sum(raw_rewards) / len(raw_rewards)
            variance = sum((r - mean) ** 2 for r in raw_rewards) / len(raw_rewards)
            std = variance ** 0.5 if variance > 0 else 1.0
            
            normalized = [(r - mean) / std for r in raw_rewards]
        else:
            normalized = raw_rewards
        
        return normalized
    
    def get_binary_label(self, trace, threshold: float = 0.6) -> bool:
        """Get binary label for KTO training"""
        combined = self.compute_combined(trace)
        return combined >= threshold
    
    def summary(self, trace) -> str:
        """Get human-readable reward summary"""
        results = self.compute_all(trace)
        combined = self.compute_combined(trace)
        
        lines = [
            "=" * 40,
            "Reward Summary",
            "=" * 40,
        ]
        
        for name, result in results.items():
            status = "✓" if result.score >= 0.5 else "✗"
            lines.append(f"{status} {name}: {result.score:.3f}")
            if result.error:
                lines.append(f"    Error: {result.error}")
        
        lines.extend([
            "-" * 40,
            f"Combined: {combined:.3f}",
            f"Binary Label: {'GOOD' if combined >= 0.6 else 'BAD'}",
            "=" * 40,
        ])
        
        return "\n".join(lines)
