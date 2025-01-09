import json
import os
import threading
import time

from toolboxv2.utils.Irings.network import NetworkManager
from toolboxv2.utils.Irings.tk_live import NetworkVisualizer
from toolboxv2.utils.Irings.one import IntelligenceRing, InputProcessor

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from queue import PriorityQueue
import numpy as np
from collections import defaultdict


class ReasoningTaskType(Enum):
    GENERAL = "general"
    PROBLEM_SOLVING = "problem_solving"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    DECISION_MAKING = "decision_making"
    PLANNING = "planning"


class ReasoningType(Enum):
    SCEPTIC = "SCEPTIC"
    CONTEXT = "CONTEXT"


class ReasoningStep(Enum):
    QUERY_GENERATION = "query_generation"
    RETRIEVAL = "knowledge_retrieval"
    SYNTHESIS = "result_synthesis"
    BACKTRACK = "backtracking"
    BRANCH = "branching"
    REASONING = "reasoning"
    INTERN_RETRIEVAL = "cached_retrieval"


@dataclass
class VectorState:
    """Vector representation of a reasoning state"""
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def similarity(self, other: 'VectorState') -> float:
        return float(np.dot(self.embedding, other.embedding))


@dataclass
class ReasoningNode:
    """Enhanced node with vector state and multi-type reasoning"""
    state: str
    vector_state: VectorState
    reasoning_type: str
    reasoning: str
    confidence: float
    parent: Optional['ReasoningNode'] = None
    children: List['ReasoningNode'] = field(default_factory=list)
    depth: int = 0
    retrieved_context: List[str] = field(default_factory=list)
    reasoning_path: List[ReasoningStep] = field(default_factory=list)

    def __lt__(self, other):
        return self.confidence > other.confidence


class RetrievalCache:
    """Cached retrieval results with vector indexing"""

    def __init__(self, vector_dimension: int):
        self.cache: Dict[str, List[Tuple[str, np.ndarray]]] = defaultdict(list)
        self.vector_dimension = vector_dimension

    def add(self, query: str, result: str, vector: np.ndarray):
        self.cache[query].append((result, vector))

    def get_similar(self, query: str, vector: np.ndarray, threshold: float = 0.8) -> List[str]:
        results = []
        if query in self.cache:
            for result, cached_vector in self.cache[query]:
                similarity = float(np.dot(vector, cached_vector))
                if similarity >= threshold:
                    results.append(result)
        return results


class ReasoningSystem:
    """Integrated FLARE, ToT, and ToR system"""

    def __init__(
        self,
        llm: Any,
        retriever: Any,
        vector_encoder: Any,
        similarity_engine: Optional[Any] = None,
        confidence_threshold: float = 0.22,
        retrieval_threshold: float = 0.8,
        max_depth: int = 4,
        max_notes: int = 6,
        vector_dimension: int = 768,
        max_branches_per_node: int = 2,
        max_retrievals_per_branch: int = 2,
        extra_reasoning_steps: Optional[Any] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.vector_encoder = vector_encoder
        self.similarity_engine = similarity_engine
        self.confidence_threshold = confidence_threshold
        self.retrieval_threshold = retrieval_threshold
        self.max_depth = max_depth
        self.max_notes = max_notes
        self.vector_dimension = vector_dimension
        self.max_branches_per_node = max_branches_per_node
        self.max_retrievals_per_branch = max_retrievals_per_branch
        if extra_reasoning_steps is None:
            extra_reasoning_steps = []
        extra_reasoning_steps = [self._apply_flare_reasoning] + extra_reasoning_steps
        self.extra_reasoning_steps = extra_reasoning_steps

        self.retrieval_cache = RetrievalCache(vector_dimension)
        self.history_manager = ReasoningHistoryManager()
        self.reasoning_history: List[Dict] = []
        self.active_nodes: Set[str] = set()

    def _encode_state(self, state: str) -> np.ndarray:
        """Encode state into vector representation"""
        return self.vector_encoder(state)

    def _generate_retrieval_queries(self, node: ReasoningNode, iquery) -> List[str]:
        """Generate diverse retrieval queries based on current reasoning state"""
        queries = []
        base_query = node.state

        # Generate variations using different reasoning types
        for reasoning_type in ReasoningType:
            prompt = f"""
            Current query: {iquery}
            Current state: {node.state}
            Current reasoning: {node.reasoning}
            Reasoning type: {reasoning_type.value}
            Generate a focused query to retrieve relevant information. for a vector db!
            """
            query = self.llm.process(prompt, max_tokens=15)
            queries.append(query)

        return queries[:self.max_retrievals_per_branch]

    def _apply_flare_reasoning(
        self,
        node: ReasoningNode,
        query: str,
        context: str
    ) -> List[ReasoningNode]:
        """Apply FLARE reasoning to generate new nodes"""
        new_nodes = []

        # Generate vector representation
        vector_state = VectorState(
            embedding=self._encode_state(node.state),
        )

        # Retrieve relevant information
        retrieved_info = []
        retrieval_queries = self._generate_retrieval_queries(node, query)

        for ret_query in retrieval_queries:
            # Check cache first
            cached_results = self.retrieval_cache.get_similar(
                ret_query,
                vector_state.embedding,
                self.retrieval_threshold
            )

            if cached_results:
                node.reasoning_path += [ReasoningStep.INTERN_RETRIEVAL]
                retrieved_info.extend(cached_results)
            else:
                # Perform new retrieval
                node.reasoning_path += [ReasoningStep.RETRIEVAL]
                result = self.retriever(ret_query)
                retrieved_info.append(result)
                # Cache the result
                self.retrieval_cache.add(
                    ret_query,
                    result,
                    self._encode_state(result)
                )
        node.reasoning_path += [ReasoningStep.BRANCH]
        # Generate new reasoning steps
        for _ in range(self.max_branches_per_node):
            combined_context = f"{context} {node.state} {' '.join(retrieved_info)}"

            new_state = self.llm.process(
                f"Reasoning about {query} with context: {combined_context}", max_tokens=300
            )

            confidence = self._calculate_confidence(new_state, context, retrieved_info)
            node.reasoning_path.append(ReasoningStep.REASONING)
            new_node = ReasoningNode(
                state=new_state,
                vector_state=VectorState(
                    embedding=self._encode_state(new_state)
                ),
                reasoning_type="FLARE",
                reasoning=f"FLARE reasoning step from {node.state}",
                confidence=confidence,
                parent=node,
                depth=node.depth + 1,
                retrieved_context=retrieved_info,
                reasoning_path=node.reasoning_path
            )

            new_nodes.append(new_node)

        return new_nodes

    def _explore_reasoning_tree(
        self,
        root: ReasoningNode,
        query: str,
        context: str
    ) -> ReasoningNode:
        """Explore reasoning tree using multiple reasoning types"""
        priority_queue = PriorityQueue()
        priority_queue.put((-root.confidence, root))
        best_node = root
        loop = 0
        while not priority_queue.empty() and loop < self.max_notes:
            loop += 1
            print(f"LOOP INDEX {loop} max : {self.max_notes}")
            _, current_node = priority_queue.get()
            print("Note Confidace:", _)
            if _ != 0.0 and _ < 0.1:
                continue

            if (current_node.confidence >= self.confidence_threshold or
                current_node.depth >= self.max_depth):
                if current_node.confidence > best_node.confidence:
                    best_node = current_node
                continue

            # Apply different reasoning types
            new_nodes = []
            for i, ex_fuc in enumerate(self.extra_reasoning_steps):
                #try:
                    # print(f"Process {i} from {len(self.extra_reasoning_steps)} steps")
                    new_nodes.extend(ex_fuc(current_node, query, context))
                #except Exception as e:
                #    print(f"Error applying extra reasoning step: {i} with {str(e)}")
            # new_nodes.extend(self._apply_tor_reasoning(current_node, query, context))

            # Add new nodes to queue
            for node in new_nodes:
                if node.state not in self.active_nodes:
                    current_node.children.append(node)
                    priority_queue.put((node.confidence, node))
                    self.active_nodes.add(node.state)

                    if node.confidence > best_node.confidence:
                        node.reasoning_path += [ReasoningStep.BACKTRACK]
                        best_node = node

            # Record reasoning history
            self.reasoning_history.append({
                "step": "exploration" if len(current_node.reasoning_path) == 0 else '->'.join(
                    [x.name for x in current_node.reasoning_path]),
                "node_state": current_node.state,
                "confidence": current_node.confidence,
                "depth": current_node.depth,
                "reasoning_type": current_node.reasoning_type,
                "retrieved_context": current_node.retrieved_context,
                "children_count": len(current_node.children),
            })

            if best_node.confidence == current_node.confidence and current_node.confidence >= self.confidence_threshold:
                break
        best_node.reasoning_path += [ReasoningStep.SYNTHESIS]

        self.history_manager.update_reasoning_step(self.reasoning_history)
        return best_node

    def generate(
        self,
        query: str,
        return_reasoning: bool = False,
        task_type: ReasoningTaskType = ReasoningTaskType.GENERAL
    ) -> Union[str, Tuple[str, List[Dict]]]:
        """Generate response using integrated reasoning approach"""
        self.reasoning_history = []
        self.active_nodes = set()

        # Start new reasoning task
        task_id = self.history_manager.start_task(query, task_type)

        # Initialize root node
        initial_state = ""
        root = ReasoningNode(
            state=query,
            vector_state=VectorState(
                embedding=self._encode_state(query)
            ),
            reasoning_type="Custom",
            reasoning="Initial state",
            confidence=0.0,
            reasoning_path=[ReasoningStep.QUERY_GENERATION]
        )

        # Explore reasoning tree
        best_node = self._explore_reasoning_tree(root, query, "")

        # Construct final output
        final_output = []
        current = best_node

        while current:
            final_output.append(current.state)
            current = current.parent

        final_result = " ".join(reversed([s for s in final_output if s]))

        self.history_manager.complete_task(
            final_response=final_result,
            confidence=best_node.confidence,
            metadata={"best_node_depth": best_node.depth}
        )

        if return_reasoning:
            return final_result, self.reasoning_history

        return final_result

    def _calculate_confidence(
        self,
        output: str,
        context: str,
        retrieved_info: List[str]
    ) -> float:
        """Calculate confidence using multiple factors"""
        if self.similarity_engine:
            # Combine multiple confidence signals
            context_confidence = self.similarity_engine(
                self._encode_state(output),
                self._encode_state(context)
            )

            retrieval_confidence = np.mean([
                self.similarity_engine(
                    self._encode_state(output),
                    self._encode_state(info)
                )
                for info in retrieved_info
            ]) if retrieved_info else 0.5

            return 0.7 * context_confidence + 0.3 * retrieval_confidence

        return 0.8  # Default confidence


@dataclass
class ReasoningMetrics:
    success_rate: float = 0.0
    average_confidence: float = 0.0
    average_depth: float = 0.0
    common_paths: Dict[str, int] = field(default_factory=dict)
    retrieval_stats: Dict[str, int] = field(default_factory=dict)


@dataclass
class TaskHistory:
    task_id: str
    task_type: ReasoningTaskType
    query: str
    timestamp: float
    final_response: str
    confidence: float
    reasoning_path: List[str]
    retrieved_contexts: List[str]
    metrics: ReasoningMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReasoningHistoryManager:
    def __init__(self, max_history_per_type: int = 1000):
        self.max_history_per_type = max_history_per_type
        self.task_histories: Dict[ReasoningTaskType, List[TaskHistory]] = defaultdict(list)
        self.current_task: Optional[TaskHistory] = None
        self.metrics_cache: Dict[ReasoningTaskType, ReasoningMetrics] = {}

    def start_task(self, query: str, task_type: ReasoningTaskType = ReasoningTaskType.GENERAL) -> str:
        """Initialize a new reasoning task"""
        task_id = f"{task_type.value}-{int(time.time())}"

        self.current_task = TaskHistory(
            task_id=task_id,
            task_type=task_type,
            query=query,
            timestamp=time.time(),
            final_response="",
            confidence=0.0,
            reasoning_path=[],
            retrieved_contexts=[],
            metrics=ReasoningMetrics()
        )

        return task_id

    def update_reasoning_step(self, reasoning_steps: List[Dict]):
        """Update current task with new reasoning steps"""
        if not self.current_task:
            return

        path = []
        contexts = []

        for step in reasoning_steps:
            path.append(step["step"])
            if step.get("retrieved_context"):
                contexts.extend(step["retrieved_context"])
            if step.get("node_state"):
                contexts.extend(step["node_state"])

        self.current_task.reasoning_path = path
        self.current_task.retrieved_contexts = contexts

    def complete_task(self, final_response: str, confidence: float, metadata: Dict[str, Any] = None):
        """Complete current task and store it in history"""
        if not self.current_task:
            return

        self.current_task.final_response = final_response
        self.current_task.confidence = confidence
        if metadata:
            self.current_task.metadata = metadata

        # Calculate task metrics
        self.current_task.metrics = self._calculate_task_metrics(self.current_task)

        # Add to history
        task_type = self.current_task.task_type
        self.task_histories[task_type].append(self.current_task)

        # Trim history if needed
        if len(self.task_histories[task_type]) > self.max_history_per_type:
            self.task_histories[task_type].pop(0)

        # Clear current task
        completed_task = self.current_task
        self.current_task = None

        # Update cached metrics
        self._update_type_metrics(task_type)

        return completed_task

    def _calculate_task_metrics(self, task: TaskHistory) -> ReasoningMetrics:
        """Calculate metrics for a single task"""
        path_str = "->".join(task.reasoning_path)
        retrieval_count = sum(1 for step in task.reasoning_path if "RETRIEVAL" in step)

        return ReasoningMetrics(
            success_rate=1.0 if task.confidence >= 0.8 else 0.0,
            average_confidence=task.confidence,
            average_depth=len(task.reasoning_path),
            common_paths={path_str: 1},
            retrieval_stats={"total_retrievals": retrieval_count}
        )

    def _update_type_metrics(self, task_type: ReasoningTaskType):
        """Update cached metrics for a task type"""
        histories = self.task_histories[task_type]
        if not histories:
            return

        paths = defaultdict(int)
        retrieval_stats = defaultdict(int)
        total_confidence = 0
        total_depth = 0
        success_count = 0

        for task in histories:
            path_str = "->".join(task.reasoning_path)
            paths[path_str] += 1

            retrieval_count = sum(1 for step in task.reasoning_path if "RETRIEVAL" in step)
            retrieval_stats["total_retrievals"] += retrieval_count

            total_confidence += task.confidence
            total_depth += len(task.reasoning_path)
            if task.confidence >= 0.8:
                success_count += 1

        self.metrics_cache[task_type] = ReasoningMetrics(
            success_rate=success_count / len(histories),
            average_confidence=total_confidence / len(histories),
            average_depth=total_depth / len(histories),
            common_paths=dict(paths),
            retrieval_stats=dict(retrieval_stats)
        )

    def get_task_history(self, task_id: str) -> Optional[TaskHistory]:
        """Retrieve specific task history"""
        for histories in self.task_histories.values():
            for task in histories:
                if task.task_id == task_id:
                    return task
        return None

    def get_type_metrics(self, task_type: ReasoningTaskType) -> Optional[ReasoningMetrics]:
        """Get cached metrics for a task type"""
        return self.metrics_cache.get(task_type)

    def search_similar_tasks(self, query: str, task_type: Optional[ReasoningTaskType] = None,
                             limit: int = 5) -> List[TaskHistory]:
        """Find similar historical tasks"""
        relevant_histories = []

        if task_type:
            histories = self.task_histories[task_type]
        else:
            histories = [task for tasks in self.task_histories.values() for task in tasks]

        # Simple keyword matching for now - could be enhanced with vector similarity
        query_words = set(query.lower().split())

        for task in histories:
            task_words = set(task.query.lower().split())
            similarity = len(query_words.intersection(task_words)) / len(query_words.union(task_words))
            if similarity > 0.3:
                relevant_histories.append((similarity, task))

        relevant_histories.sort(reverse=True, key=lambda x: x[0])
        return [task for _, task in relevant_histories[:limit]]

    def export_history(self, filepath: str):
        """Export reasoning history to file"""
        export_data = {
            "task_histories": {
                task_type.value: [
                    {
                        "task_id": task.task_id,
                        "query": task.query,
                        "timestamp": task.timestamp,
                        "final_response": task.final_response,
                        "confidence": task.confidence,
                        "reasoning_path": task.reasoning_path,
                        "retrieved_contexts": task.retrieved_contexts,
                        "metrics": vars(task.metrics),
                        "metadata": task.metadata
                    }
                    for task in histories
                ]
                for task_type, histories in self.task_histories.items()
            },
            "metrics_cache": {
                task_type.value: vars(metrics)
                for task_type, metrics in self.metrics_cache.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

    @classmethod
    def import_history(cls, filepath: str) -> 'ReasoningHistoryManager':
        """Import reasoning history from file"""
        manager = cls()

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Restore task histories
        for task_type_str, histories in data["task_histories"].items():
            task_type = ReasoningTaskType(task_type_str)
            manager.task_histories[task_type] = [
                TaskHistory(
                    task_id=h["task_id"],
                    task_type=task_type,
                    query=h["query"],
                    timestamp=h["timestamp"],
                    final_response=h["final_response"],
                    confidence=h["confidence"],
                    reasoning_path=h["reasoning_path"],
                    retrieved_contexts=h["retrieved_contexts"],
                    metrics=ReasoningMetrics(**h["metrics"]),
                    metadata=h["metadata"]
                )
                for h in histories
            ]

        # Restore metrics cache
        for task_type_str, metrics in data["metrics_cache"].items():
            manager.metrics_cache[ReasoningTaskType(task_type_str)] = ReasoningMetrics(**metrics)

        return manager
