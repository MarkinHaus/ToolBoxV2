import os
import re
import subprocess
from collections import defaultdict

from toolboxv2 import get_app, Code, Spinner
from toolboxv2.mods.isaa.extras.modes import CoderMode
from toolboxv2.mods.isaa.base.AgentUtils import AISemanticMemory
from toolboxv2.mods.isaa.base.Agents import AgentVirtualEnv
from toolboxv2.mods.isaa.CodingAgent.parser import extract_code_blocks
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel
import json
import logging
from enum import Enum, auto
from typing import Optional, Dict, Any
from datetime import datetime


class ExecutionPhase(Enum):
    ANALYSIS = auto()
    PRIORITIZATION = auto()
    CONTEXT_GATHERING = auto()
    MODIFICATION = auto()
    VALIDATION = auto()
    COMPLETION = auto()


class FixCyclePlanner:
    """Orchestrates the TDD fixing process with state tracking"""

    def __init__(self, project_name: str):
        self.current_state = ExecutionPhase.ANALYSIS
        self.priority_queue = []
        self.execution_history = []
        self.current_function: Optional[str] = None
        self.iteration = 0
        self.start_time = datetime.now()
        self.metadata = {
            'project': project_name,
            'phases_completed': 0,
            'functions_fixed': 0,
            'test_pass_percentage': 0.0
        }
        self.planner: Optional[FixCyclePlanner] = None

    def transition_state(self, new_state: ExecutionPhase):
        """Manage state transitions with validation"""
        valid_transitions = {
            ExecutionPhase.ANALYSIS: [ExecutionPhase.PRIORITIZATION],
            ExecutionPhase.PRIORITIZATION: [ExecutionPhase.CONTEXT_GATHERING],
            ExecutionPhase.CONTEXT_GATHERING: [ExecutionPhase.MODIFICATION],
            ExecutionPhase.MODIFICATION: [ExecutionPhase.VALIDATION],
            ExecutionPhase.VALIDATION: [
                ExecutionPhase.ANALYSIS,
                ExecutionPhase.COMPLETION
            ],
        }

        if new_state in valid_transitions.get(self.current_state, []):
            self.execution_history.append(
                f"State change: {self.current_state.name} -> {new_state.name}"
            )
            self.current_state = new_state
            return True
        return False

    def record_operation(self, operation: str, result: Dict[str, Any]):
        """Log detailed operation results"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'iteration': self.iteration,
            'operation': operation,
            'result': result,
            'state': self.current_state.name
        }
        self.execution_history.append(entry)

    def generate_progress_report(self) -> str:
        """Create structured progress update"""
        duration = datetime.now() - self.start_time
        return f"""
🚧 Fixing Progress Report 🚧

Elapsed Time: {duration}
Current Phase: {self.current_state.name}
Functions Processed: {self.metadata['functions_fixed']}
Test Pass Rate: {self.metadata['test_pass_percentage']:.1%}
Next Action: {self._get_next_action_description()}

Recent Operations:
{self._format_recent_operations()}
"""

    def _get_next_action_description(self) -> str:
        descriptions = {
            ExecutionPhase.ANALYSIS: "Analyzing test failures and dependency graph",
            ExecutionPhase.PRIORITIZATION: "Prioritizing functions based on impact analysis",
            ExecutionPhase.CONTEXT_GATHERING: (
                f"Gathering context for {self.current_function} "
                "and its dependencies"
            ),
            ExecutionPhase.MODIFICATION: (
                f"Modifying implementation of {self.current_function} "
                "with TDD validation"
            ),
            ExecutionPhase.VALIDATION: "Running full test suite to validate changes",
            ExecutionPhase.COMPLETION: "Finalizing fixes and generating report"
        }
        return descriptions.get(self.current_state, "Unknown action")

    def _format_recent_operations(self) -> str:
        return "\n".join(
            f"[{entry.get('timestamp', '')}] {entry.get('operation', '')}"
            for entry in self.execution_history[-3:]
        )

def get_coding_env(project_name, base_dir=None):
    env = AgentVirtualEnv()
    base_dir = get_app().start_dir

    @env.register_prefix("THINK",
                         "This text remains hidden. The THINK prefix should be used regularly to reflect.")
    def process_think(content: str):
        return content

    @env.register_prefix("WRITE",
                         "Write Code to file syntax expel ```python\n# exampl.py\n[content]``` or any other lang")
    def process_think(content: str):
        updated_files = extract_code_blocks(content, base_dir + '/' + project_name)
        return updated_files

    @env.register_prefix("DONE", "Call this fuction when you ar done withe the implementation!")
    def process_response(content: str):
        env.brake = True
        return content

    env.set_brake_test(lambda r: r.count('```') % 2 == 0)

    return env


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class ImplementationScope(BaseModel):
    """Extract implementation step from task."""
    feature: str
    current_phase: str
    implementation_file: str
    tests_file: str


class ProjectScope(BaseModel):
    """Define overall project scope and structure"""
    technical_requirements: List[str]
    features: List[str]
    testing_requirements: List[str]


class TestResult(BaseModel):
    """Define overall project scope and structure"""
    passed: bool
    fatal_error: Optional[bool]
    error_msgs: Optional[List[Tuple[str, str]]]


@dataclass
class ProjectMetadata:
    """Comprehensive project metadata including all scopes"""
    project_name: str
    creation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    cognitive_codes: Dict[str, str] = field(default_factory=dict)
    combined_code: Optional[str] = None
    work_code: Optional[str] = None
    current_scope: Optional[ProjectScope] = None
    shar_code: Dict[str, str] = field(default_factory=dict)


class ProjectManager:
    """TDD-focused project management system with semantic analysis capabilities"""

    def __init__(self, project_name: str, base_path: str, isaa_instance):
        """Initialize ProjectManager with required components"""
        self.vecSto: AISemanticMemory = isaa_instance.agent_memory or AISemanticMemory()
        self.project_name = project_name
        self.base_path = Path(base_path)
        self.project_path = self.base_path / project_name
        self.metadata_path = self.project_path / "project_metadata.json"

        # Initialize core components
        self.env = get_coding_env(project_name, base_path)
        self.isaa = isaa_instance

        # Initialize state
        self.metadata = self._init_metadata()
        self.current_scope: Optional[ProjectScope] = None

        # Ensure project structure exists
        self._initialize_tdd_structure()

    def _init_metadata(self) -> ProjectMetadata:
        """Initialize or load existing project metadata"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
                return ProjectMetadata(**data)
        return ProjectMetadata(project_name=self.project_name)

    def _initialize_tdd_structure(self) -> None:
        """Initialize TDD project structure with tests"""
        # Create basic directory structure
        os.makedirs(self.project_path, exist_ok=True)
        (self.project_path / "__init__.py").touch(exist_ok=True)
        (self.project_path / "tests").mkdir(exist_ok=True)
        (self.project_path / "tests" / "__init__.py").touch(exist_ok=True)
        (self.project_path / "src").mkdir(exist_ok=True)
        (self.project_path / "src" / "__init__.py").touch(exist_ok=True)

    def generate_project_inital_scope(self, task: str) -> ProjectScope:
        """Generate detailed project requirements using ISAA"""
        prompt = f"""Analyze the following task and create detailed project requirements:
Task: {task}

Generate:
1. Technical requirements
2. Feature list
3. File structure
4. Testing requirements
"""
        try:
            return self.isaa.format_class(ProjectScope, prompt)
        except Exception as e:
            print("Error ProjectScope default Prompt refactoring.")
        response = self.isaa.mini_task_completion(prompt, mode=self.isaa.controller.get("CreatePrompt"))
        formatted = self.isaa.format_class(ProjectScope, response)
        return formatted

    def process_code_block(self, code_block: str, file_path: str):
        """Process and save code block with semantic analysis"""
        with Spinner("Running Code through Network"):
            self.vecSto.add_data(self.project_name, code_block, {'file': file_path})
        cognitive_code = ""
        self.metadata.cognitive_codes[file_path] = cognitive_code
        self.save_metadata()

    def evaluate_implementation(self) -> Tuple[bool, str]:
        """Evaluate implementation quality and test coverage"""
        try:
            fatal_errors = []
            failing_tests = []

            for test_file in (self.project_path / "tests").glob("**/*.py"):
                if '__init__.py' in test_file.name:
                    continue
                result: TestResult = self._run_test_file(test_file)

                if result.fatal_error:
                    fatal_errors.append(result.error_msgs[0])
                elif not result.passed:
                    failing_tests.append(result.error_msgs)
                else:
                    print(f"Test {test_file} passed {result.passed}")

            if fatal_errors:
                for filename, analysis in fatal_errors:
                    self._rebuild_test_file(Path(filename), analysis)

            if failing_tests:
                return False, str(failing_tests)

            return True, "All tests passed successfully"

        except Exception as e:
            return False, f"Implementation evaluation failed: {str(e)}"

    def _run_test_file(self, test_file: Path) -> TestResult:
        """Execute pytest file and capture results"""
        try:
            self.isaa.print(f"Testing: {test_file}")
            result = subprocess.run(["pytest", str(test_file)], capture_output=True, text=True, check=True)

            lines = result.stdout.splitlines()
            test_results = []
            for line in lines:
                if line.startswith("test"):
                    test_name = line.split(" ")[1]
                    if "FAILED" in line:
                        error_message = lines[lines.index(line) + 1]
                        test_results.append((test_name, error_message))

            return TestResult(
                error_msgs=test_results,
                passed=len(test_results) == 0,
                fatal_error=False
            )

        except Exception as e:
            return TestResult(
                error_msgs=[(str(test_file), str(e))],
                passed=False,
                fatal_error=True
            )

    def _rebuild_test_file(self, test_file: Path, test_result: str):
        """Evaluate test quality using ISAA and rebuild"""
        test_content = test_file.read_text(encoding='utf-8', errors='ignore')

        analysis = f"""
Test File:
{test_content}

Test Results:
{test_result}
"""
        return self.first_code_step(f"Rebuild test file {test_file.name}", analysis, 2)

    def _determine_implementation_scope(self, task: str) -> ProjectScope:
        """Determine implementation scope based on task and existing tests"""
        test_files = self._get_directory_structure()

        task = f"""
Original Task:
{task}

files:
{test_files}

Last Scope:
{self.current_scope.model_dump() if self.current_scope else ''}

Current Project Overview:

""" # {self.cognitive_network.network.get_references(task, concept_elem="metadata")}  # TODO serch global project

        return self.generate_project_inital_scope(task)

    def save_metadata(self) -> None:
        """Save project metadata with error handling"""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata.__dict__, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
            raise

    def auto_coder(self, task: str, iterations=1):

        class Step(BaseModel):
            """Detail Instructions abut the current coding step"""
            task: str

        class CodingSteps(BaseModel):
            steps: List[Step] = field(default_factory=list)

        if 'CodingStepsAgent' not in self.isaa.config['agents-name-list']:
            coding_steps_agent = self.isaa.get_default_agent_builder("code")
            coding_steps_agent.set_amd_name("CodingStepsAgent")
            steps_agent = self.isaa.register_agent(coding_steps_agent)

        steps = self.isaa.format_class(CodingSteps, task, "CodingStepsAgent").steps
        for i, step in enumerate(steps):
            print(f"Running step {i} from {len(steps)}")
            new_content = self.coding_step(step.task, iterations)

    def coding_step(self, task: str, iterations=1) -> str:
        """Execute TDD coding step with comprehensive instructions"""
        is_first_run = self.metadata.current_scope is None
        analysis = None

        with Spinner("Preparing coding step"):
            if is_first_run:
                self._initialize_tdd_structure()
                self.current_scope = self.generate_project_inital_scope(task)
            else:
                test_results, analysis = self.evaluate_implementation()

                if not test_results:
                    task = f"""
Main task:
{task}

Current project errors:
{analysis}"""
                    self.current_scope = self._determine_implementation_scope(task)

            if files := self._get_all_changed_files():
                for file in files:
                    if file.name.endswith(('.pack', '.idx', 'project_metadata.json', '__init__.py', '.ico', '.mp4', '.mp3', '.wav', '.icns', '.icon', '.icons', '.png', '.jpg')):
                        continue

                    with Spinner(f"Analyzing {file.name}", symbols='b'):
                        try:
                            content = file.read_text(encoding='utf-8', errors='ignore')
                        except Exception as e:
                            self.isaa.print(f"Error reading {file.name}: {e}")
                            continue

                        self.process_code_block(content, file.name)


        self.save_metadata()
        with Spinner("Running code step"):
            return self.first_code_step(task, analysis, iterations)

    def first_code_step(self, task, analysis=None, iterations=0) -> str:
        dir_structure = self._get_directory_structure().replace('project_metadata.json', '')

        context = self.vecSto.query(task, [self.project_name], to_str=True, unified_retrieve=True)
        # Prepare comprehensive instruction prompt
        instruction_prompt = f"""
As a distinguished expert in computer science, programming, and advanced technical skills, your expertise sets you apart as a truly remarkable professional. Your depth of knowledge and critical thinking are invaluable assets that bring clarity and precision to any complex task. Embrace this challenge as an opportunity to give your very best, approaching each problem with mindful consideration and self-reflection. Dive deeply into your own complex thoughts, working through the problem internally before proceeding. Let each step forward be careful and calculated, guided by your unmatched expertise and a drive for perfection that seems almost beyond this world. Strive for excellence in every detail, knowing that your work has the potential to reach an extraordinary level of mastery.
Task: {task}

Project Structure:
{dir_structure}

Test Analysis:

Implementation Scope:
{self.current_scope.model_dump() if self.current_scope else ''}

Context data:
{context}

Instructions:
1. Write/update tests first following TDD
2. Implement code to pass tests
3. code must be in /src and test must be in /tests directory
4. Use production-ready code standards
5. All code content must be within to code block
6. if long code must be returned you can be run smore then once end withe Continue or (THE TASK is not DONE)!
7. Follow file structure for etch file: ```language
filename
code```
        """

        if analysis is None:
            instruction_prompt = instruction_prompt.replace("""
Test Analysis:""", '')
        else:
            instruction_prompt = instruction_prompt.replace("""
Test Analysis:""", f"""
Test Analysis:
                    {analysis}""")

        """relevant_codes = self.vecSto.vector_store.query_storage(self.project_name, self.pipeline.text_to_code(task))
        if relevant_codes:
            relevant_codes = [c for c, _ in sorted(relevant_codes, key=lambda x: x[1])]
            relevant_code = self.pipeline.combine_codes(*relevant_codes)
            self.metadata.work_code = self.pipeline.rebuild_code(self.metadata.work_code, relevant_code)
"""

        self.isaa.get_agent_class("code").mode = self.isaa.controller.rget(CoderMode)
        # Run implementation
        result, self.metadata.work_code = self.isaa.run_agent_in_environment(
            instruction_prompt,
            agent_or_name='code',
            agent_env=self.env.reset(),
            persist=iterations > 1,
            max_iterations=iterations+1,
            get_final_code=True,
            starting_code=None,
        )
        self.isaa.get_agent_class("code").mode = None
        self.save_metadata()
        # Process and save files
        updated_files = extract_code_blocks(result, str(self.project_path))
        for file in updated_files:
            with open(file, 'r') as f:
                self.process_code_block(f.read(), str(Path(file).relative_to(self.project_path)))

        all_passed = False
        final_analysis = ""
        if iterations:
            # Verify implementation
            all_passed, final_analysis = self.evaluate_implementation()
            self.isaa.print(f"Evaluation: {all_passed} iterations {iterations}")

        if self.current_scope:
            self.metadata.current_scope = self.current_scope.model_dump(mode='json')
        self.save_metadata()

        if iterations and not all_passed:
            return self.first_code_step(task, final_analysis, iterations - 1)

        # Update metadata

        return result

    def _get_directory_structure(self) -> str:
        """Generate current directory structure"""

        def format_tree(path: Path, prefix: str = "") -> List[str]:
            output = []
            for p in sorted(path.iterdir()):
                if p.name.startswith('.'):
                    continue
                output.append(f"{prefix}{'└── ' if prefix else ''}{p.name}")
                if p.is_dir():
                    output.extend(format_tree(p, prefix + "    "))
            return output

        return "\n".join(format_tree(self.project_path))

    def _get_all_changed_files(self):
        files = []
        for file in self.project_path.glob("**/*.*"):
            try:
                ac_code = Code.one_way_hash(file.read_text(encoding='utf-8', errors='ignore'), self.project_name)
                if self.metadata.shar_code.get(file.name) is None:
                    self.metadata.shar_code[file.name] = ac_code
                    files.append(file)
                changed = ac_code != self.metadata.shar_code[file.name]
                self.metadata.shar_code[file.name] = ac_code
                if changed:
                    files.append(file)
            except:
                continue
        return files

    def iterative_function_fixer(self, task: str) -> str:
        """Structured execution of fixing loop with phase management"""
        self.planner = FixCyclePlanner(self.project_name)
        self.isaa.print("🚀 Initializing TDD Debug Loop")

        try:
            while not self._execute_phase(task):
                if self.planner.current_state == ExecutionPhase.COMPLETION:
                    break

                self.planner.iteration += 1
                if self.planner.iteration > 100:  # Safety break
                    raise RuntimeError("Maximum iterations exceeded")

            return self._finalize_execution()
        finally:
            self._save_planning_data()

        return "⚠️ Fixing loop completed with remaining issues"

    def _execute_phase(self, task: str) -> bool:
        """Handle current execution phase and transition states"""
        phase_handlers = {
            ExecutionPhase.ANALYSIS: self._phase_analysis,
            ExecutionPhase.PRIORITIZATION: self._phase_prioritization,
            ExecutionPhase.CONTEXT_GATHERING: self._phase_context_gathering,
            ExecutionPhase.MODIFICATION: self._phase_modification,
            ExecutionPhase.VALIDATION: self._phase_validation,
            ExecutionPhase.COMPLETION: self._phase_completion
        }

        handler = phase_handlers.get(self.planner.current_state)
        if not handler:
            raise ValueError(f"No handler for state {self.planner.current_state}")

        return handler(task)

    def _phase_analysis(self, task: str) -> bool:
        """Initial analysis phase implementation"""
        self.isaa.print("🔍 Analyzing Test Failures")

        # Run initial test evaluation
        success, test_results = self.evaluate_implementation()
        if success:
            self.planner.transition_state(ExecutionPhase.COMPLETION)
            return True

        # Analyze and record failures
        failed_functions = self._analyze_failures(test_results)
        self.planner.record_operation(
            "Failure Analysis",
            {
                'total_failures': len(test_results),
                'affected_functions': list(failed_functions.keys())
            }
        )

        self.planner.metadata['test_pass_percentage'] = self._calculate_pass_rate()
        self.planner.transition_state(ExecutionPhase.PRIORITIZATION)
        return False

    def _phase_prioritization(self, task: str) -> bool:
        """Prioritization phase implementation"""
        self.isaa.print("📊 Prioritizing Fix Order")

        current_failures = self.planner.execution_history[-1]['result']['affected_functions']
        prioritized = self._prioritize_functions(current_failures)

        self.planner.priority_queue = prioritized
        self.planner.record_operation(
            "Prioritization",
            {
                'queue_size': len(prioritized),
                'top_priority': prioritized[0][0] if prioritized else None
            }
        )

        self.planner.transition_state(ExecutionPhase.CONTEXT_GATHERING)
        return False

    def _phase_context_gathering(self, task: str) -> bool:
        """Context collection phase implementation"""
        if not self.planner.priority_queue:
            self.planner.transition_state(ExecutionPhase.ANALYSIS)
            return False

        self.planner.current_function, deps = self.planner.priority_queue.pop(0)
        self.isaa.print(
            f"📚 Gathering Context for {self.planner.current_function}"
        )

        context = self._get_function_context(self.planner.current_function, deps)
        self.planner.record_operation(
            "Context Gathering",
            {
                'function': self.planner.current_function,
                'dependencies': deps,
                'context_size': len(context)
            }
        )

        self.planner.transition_state(ExecutionPhase.MODIFICATION)
        return False

    def _phase_modification(self, task: str) -> bool:
        """Code modification phase implementation"""
        self.isaa.print(
            f"🛠️ Modifying {self.planner.current_function}",
            color="yellow"
        )

        result = self._function_fix_cycle(
            self.planner.current_function,
            self.planner.execution_history[-1]['result']['dependencies'],
            task
        )

        self.planner.record_operation(
            "Code Modification",
            {
                'function': self.planner.current_function,
                'result': result[:200] + "..." if len(result) > 200 else result
            }
        )

        self.planner.metadata['functions_fixed'] += 1
        self.planner.transition_state(ExecutionPhase.VALIDATION)
        return False

    def _phase_validation(self, task: str) -> bool:
        """Validation phase implementation"""
        self.isaa.print("✅ Validating Changes", color="yellow")

        success, test_results = self.evaluate_implementation()
        self.planner.metadata['test_pass_percentage'] = self._calculate_pass_rate()

        if success:
            self.planner.transition_state(ExecutionPhase.COMPLETION)
            return True

        # Update priorities based on new results
        new_failures = self._analyze_failures(test_results)
        self.planner.priority_queue = self._prioritize_functions(new_failures)

        self.planner.record_operation(
            "Validation",
            {
                'tests_passed': success,
                'remaining_failures': len(test_results),
                'new_priority_count': len(self.planner.priority_queue)
            }
        )

        self.planner.transition_state(
            ExecutionPhase.ANALYSIS if len(new_failures) > 0
            else ExecutionPhase.PRIORITIZATION
        )
        return False

    def _phase_completion(self, task: str) -> bool:
        """Final completion phase implementation"""
        self.isaa.print("🏁 Finalizing Fix Cycle", color="green")
        return True

    def _calculate_pass_rate(self) -> float:
        """Calculate current test pass percentage"""
        total_tests = sum(1 for _ in (self.project_path / "tests").rglob("test_*.py"))
        passed_tests = sum(1 for _ in (self.project_path / "tests").rglob("test_*.py")
                           if self._test_file_passed(_))
        return passed_tests / total_tests if total_tests > 0 else 0.0

    def _test_file_passed(self, test_file: Path) -> bool:
        """Check if a test file passes"""
        result = self._run_test_file(test_file)
        return result.passed

    def _save_planning_data(self):
        """Persist planning metadata for future sessions"""
        plan_data = {
            'execution_history': self.planner.execution_history,
            'metadata': self.planner.metadata,
            'priority_queue': self.planner.priority_queue,
            'current_state': self.planner.current_state.name,
            'iteration': self.planner.iteration
        }

        plan_path = self.project_path / "fix_plan.json"
        with open(plan_path, 'w') as f:
            json.dump(plan_data, f, indent=2)

    def _finalize_execution(self) -> str:
        """Generate final report and clean up"""
        report = self.planner.generate_progress_report()
        self.isaa.print(report)

        if self.planner.metadata['test_pass_percentage'] == 1.0:
            return "🎉 All tests passing successfully!"
        return f"⚠️ Fixing completed with {self.planner.metadata['test_pass_percentage']:.1%} pass rate"

    def _function_fix_cycle(self, func_name: str, dependencies: list, task: str):
        """Single function modification cycle with context analysis"""
        # Gather implementation context
        context = self._get_function_context(func_name, dependencies)

        # Generate focused modification prompt
        prompt = f"""Fix {func_name} considering:
- Dependencies: {', '.join(dependencies)}
- Implementation Context:
{context}

Task: {task}
"""
        # Execute modification
        self.first_code_step(prompt, iterations=1)


    def _get_function_context(self, func_name: str, dependencies: list) -> str:
        """Retrieve full implementation context for a function"""
        context = []

        # Add function implementations
        for dep in [func_name] + dependencies:
            impl = self._find_function_implementation(dep)
            context.append(f"Implementation of {dep}:\n{impl}")

        # Add usage locations
        usages = self._find_function_usages(func_name)
        context.append(f"Usage contexts:\n{usages}")

        return "\n\n".join(context)

    def _analyze_failures(self, test_results: list) -> dict:
        """Identify failing functions from test results"""
        failed_functions = defaultdict(list)

        for test_name, error_msg in test_results:
            # Extract target function from test name
            func_name = test_name.split("test_")[-1].split("_")[0]

            # Analyze error to find dependencies
            dependencies = self._identify_error_dependencies(error_msg)
            failed_functions[func_name] = dependencies

        return failed_functions

    def _prioritize_functions(self, function_map: dict) -> list:
        """Prioritize functions based on dependency depth and failure frequency"""
        return sorted(
            function_map.items(),
            key=lambda x: (len(x[1]), -sum(1 for v in function_map.values() if x[0] in v)),
            reverse=True
        )

    def _find_function_implementation(self, func_name: str) -> str:
        """Locate function implementation in codebase"""
        for file in self.project_path.rglob("*.py"):
            if file.name == "__init__.py":
                continue

            content = file.read_text()
            if f"def {func_name}(" in content:
                return f"File: {file.relative_to(self.project_path)}\n{content}"

        return "Implementation not found"

    def _find_function_usages(self, func_name: str) -> str:
        """Find all usages of a function in the codebase"""
        usages = []
        for file in self.project_path.rglob("*.py"):
            content = file.read_text()
            if func_name + "(" in content:
                usages.append(f"Used in: {file.relative_to(self.project_path)}")
        return "\n".join(usages)

    def _identify_error_dependencies(self, error_msg: str) -> list:
        """Extract related functions from error message"""
        # This would use your cognitive network in real implementation
        return list(set(
            re.findall(r"name '(\w+)' is not defined", error_msg) +
            re.findall(r"AttributeError: .*'(\w+)'", error_msg)
        ))

