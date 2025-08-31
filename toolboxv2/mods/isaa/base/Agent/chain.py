import asyncio
import random
from enum import Enum
from typing import Any, Union, List, Dict, Tuple, Optional
from pydantic import BaseModel
import copy


class ChainRunType(Enum):
    auto = "auto"
    a_run = "a_run"
    format_class = "format_class"

class CF:
    """Chain Format - handles formatting between agents"""

    def __init__(self, format_class: type[BaseModel]):
        self.format_class = format_class
        self.extract_key = None
        self.extract_multiple = False
        self.parallel_count = None

    def __sub__(self, key):
        """Implements - operator for key extraction"""
        new_cf = copy.copy(self)
        if isinstance(key, str):
            if key == '*':
                new_cf.extract_key = '*'  # Extract all fields
            elif key.startswith('*'):
                new_cf.extract_key = key[1:]  # Remove * prefix
            elif '[n]' in key:
                new_cf.extract_key = key.replace('[n]', '')
                new_cf.parallel_count = 'n'  # Will be determined at runtime
            else:
                new_cf.extract_key = key
        elif isinstance(key, tuple):
            new_cf.extract_key = key
        return new_cf

    def __rshift__(self, other):
        """Implements >> operator after CF"""
        return Chain._create_chain([self, other])


class IS:
    """Conditional check"""

    def __init__(self, key: str, expected_value: Any):
        self.key = key
        self.expected_value = expected_value

    def __rshift__(self, other):
        """Implements >> operator after IS"""
        return ConditionalChain(self, other)


class ParallelChain:
    """Handles parallel execution of agents"""

    def __init__(self, agents: List[Union['FlowAgent', 'Chain']]):
        self.agents = agents

    async def a_run(self, query):
        """Run all agents in parallel"""
        tasks = []
        for agent in self.agents:
            if hasattr(agent, 'a_run'):
                tasks.append(agent.a_run(query))
            else:
                tasks.append(agent.run(query))  # For other chain types

        results = await asyncio.gather(*tasks)
        return self._combine_results(results)

    def _combine_results(self, results):
        """Intelligently combine parallel results"""
        if len(results) == 1:
            return results[0]

        # Simple combination - could be made more sophisticated
        if all(isinstance(r, str) for r in results):
            return " | ".join(results)
        elif all(isinstance(r, dict) for r in results):
            combined = {}
            for i, result in enumerate(results):
                combined[f"agent_{i}"] = result
            return combined
        else:
            return results

    def __rshift__(self, other):
        """Implements >> operator"""
        return Chain._create_chain([self, other])

    def __or__(self, other):
        """Implements | operator for error handling"""
        return ErrorHandlingChain(self, other)


class ConditionalChain:
    """Handles conditional execution"""

    def __init__(self, condition, true_branch, false_branch=None):
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch

    def __mod__(self, other):
        """Implements % operator for false branch"""
        return ConditionalChain(self.condition, self.true_branch, other)

    async def a_run(self, query_or_data):
        """Execute based on condition"""
        if isinstance(self.condition, IS):
            # Check condition
            if isinstance(query_or_data, dict) and self.condition.key in query_or_data:
                condition_met = query_or_data[self.condition.key] == self.condition.expected_value
            else:
                condition_met = False

            if condition_met:
                return await self._run_branch(self.true_branch, query_or_data)
            elif self.false_branch:
                return await self._run_branch(self.false_branch, query_or_data)
            else:
                return query_or_data
        else:
            # Direct conditional execution
            return await self._run_branch(self.true_branch, query_or_data)

    async def _run_branch(self, branch, data):
        if hasattr(branch, 'a_run'):
            return await branch.a_run(data)
        elif hasattr(branch, 'run'):
            return await branch.run(data)
        else:
            return data


class ErrorHandlingChain:
    """Handles error cases with fallback"""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    async def a_run(self, query):
        try:
            return await self.primary.a_run(query)
        except Exception as e:
            print(f"Primary chain failed with {e}, trying fallback")
            if hasattr(self.fallback, 'a_run'):
                return await self.fallback.a_run(query)
            else:
                return await self.fallback.run(query)


class Chain:
    def __init__(self, agent: 'FlowAgent' = None):
        if agent:
            self.agent = agent
            self.tasks = [agent]
        else:
            self.tasks = []
        self.name = "chain"
        self.description = "A chain of tasks"
        self.return_key = "chain"
        self.use = "chain"

    @classmethod
    def _create_chain(cls, components):
        """Create a chain from a list of components"""
        chain = cls()
        chain.tasks = components
        return chain

    async def a_run(self, query: Union[str, BaseModel], use="auto"):
        """Execute the chain asynchronously"""
        current_data = query

        for i, task in enumerate(self.tasks):
            print(f"Executing step {i + 1}: {type(task).__name__}")

            if isinstance(task, CF):
                # Format the current data using the format class
                if hasattr(current_data, 'model_dump'):
                    current_data = current_data.model_dump()

                # Apply any extraction logic
                if task.extract_key:
                    current_data = self._extract_data(current_data, task)

                # Convert to the desired format
                if isinstance(current_data, dict):
                    current_data = task.format_class(**current_data)
                else:
                    current_data = task.format_class(value=str(current_data))

            elif isinstance(task, IS):
                # This would be handled by ConditionalChain
                pass

            elif isinstance(task, (ParallelChain, ConditionalChain, ErrorHandlingChain)):
                current_data = await task.a_run(current_data)

            elif hasattr(task, 'a_run'):
                # Regular agent
                if isinstance(current_data, BaseModel):
                    current_data = await task.a_run(str(current_data.model_dump()))
                else:
                    current_data = await task.a_run(str(current_data))

        return current_data

    def _extract_data(self, data, cf: CF):
        """Extract data based on CF configuration"""
        if not isinstance(data, dict):
            return data

        if cf.extract_key == '*':
            return data  # Return all
        elif isinstance(cf.extract_key, tuple):
            return {k: data.get(k) for k in cf.extract_key if k in data}
        elif cf.extract_key in data:
            return data[cf.extract_key]
        else:
            return data

    def run(self, query: Union[str, BaseModel], use="auto"):
        """Synchronous wrapper"""
        return asyncio.run(self.a_run(query, use))

    def __rshift__(self, other):
        """Implements >> operator for extending chains"""
        new_tasks = self.tasks + [other]
        return Chain._create_chain(new_tasks)

    def __add__(self, other):
        """Implements + operator for parallel execution"""
        return ParallelChain([self, other])

    def __and__(self, other):
        """Implements & operator for parallel execution"""
        return ParallelChain([self, other])

    def __mod__(self, other):
        """Implements % operator for conditional branching"""
        return ConditionalChain(None, self, other)

    def __or__(self, other):
        """Implements | operator for error handling"""
        return ErrorHandlingChain(self, other)



