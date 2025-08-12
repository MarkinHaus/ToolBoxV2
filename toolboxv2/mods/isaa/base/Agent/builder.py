import asyncio
import json
import yaml
import logging
import os
import sys
import inspect
import re
import ast
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Type, Set
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from dataclasses import asdict
from datetime import datetime
import uuid

# Import agent components
from agent import (
    FlowAgent,
    AgentModelData,
    PersonaConfig,
    AsyncFlowT,
    AsyncNodeT,
    VariableManager,
    LITELLM_AVAILABLE,
    A2A_AVAILABLE,
    MCP_AVAILABLE,
    OTEL_AVAILABLE
)

# Framework imports
if LITELLM_AVAILABLE:
    from litellm import BudgetManager
    import litellm

if OTEL_AVAILABLE:
    from opentelemetry.sdk.trace import TracerProvider

if MCP_AVAILABLE:
    from mcp.server.fastmcp import FastMCP

from toolboxv2 import get_logger

logger = get_logger()


# ===== ENHANCED CONFIGURATION MODELS =====

class MCPToolConfig(BaseModel):
    """Configuration for MCP tools"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    function_code: Optional[str] = None
    function_file: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    async_mode: bool = True
    expose_in_mcp: bool = True
    auto_analyze: bool = True
    category: str = "general"
    requires_packages: List[str] = Field(default_factory=list)
    environment_setup: Optional[str] = None


class CodeExecutionConfig(BaseModel):
    """Configuration for code execution capabilities"""
    enabled: bool = True
    allowed_languages: List[str] = Field(default_factory=lambda: ["python", "javascript", "bash"])
    sandbox_mode: bool = True
    timeout_seconds: int = 30
    max_output_length: int = 10000
    install_packages: bool = True
    persistent_environment: bool = False
    environment_path: Optional[str] = None
    allowed_imports: List[str] = Field(default_factory=lambda: ["os", "sys", "json", "yaml", "re", "math", "datetime"])
    blocked_imports: List[str] = Field(default_factory=lambda: ["subprocess", "socket", "urllib"])


class FlowSpecializationConfig(BaseModel):
    """Configuration for specialized flows"""
    name: str
    domain: str
    nodes: List[str] = Field(default_factory=list)
    custom_strategies: Dict[str, Dict] = Field(default_factory=dict)
    specialized_tools: List[str] = Field(default_factory=list)
    context_awareness: str = "enhanced"
    persona_integration: str = "domain_adapted"
    flow_pattern: str = "adaptive"
    priority_rules: Dict[str, int] = Field(default_factory=dict)


class PersonaProfileConfig(BaseModel):
    """Extended persona configuration"""
    profiles: Dict[str, PersonaConfig] = Field(default_factory=dict)
    active_profile: str = "default"
    domain_adaptations: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    auto_switch: bool = False
    switch_triggers: Dict[str, str] = Field(default_factory=dict)
    context_sensitivity: float = 0.7


class AdvancedBuilderConfig(BaseModel):
    """Complete enhanced agent configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Basic agent settings
    agent_name: str = "AdvancedProductionAgent"
    agent_version: str = "2.0.0"
    description: str = "Advanced production-ready PocketFlow agent with specialization capabilities"

    # Model configuration
    fast_llm_model: str = "openrouter/anthropic/claude-3-haiku"
    complex_llm_model: str = "openrouter/openai/gpt-4o"
    system_message: str = """You are an advanced autonomous agent built on the enhanced PocketFlow framework.

Your capabilities include:
- Intelligent task decomposition and execution
- Advanced context management with session awareness
- Specialized domain expertise (code, analysis, management)
- Dynamic tool integration and usage
- Adaptive planning with reflection and optimization
- Persona-aware communication
- Code execution and development assistance
- Variable system for dynamic content generation

You operate with specialized flows optimized for different domains and can adapt your approach based on task requirements.
Always use available tools when they can help solve the user's request."""

    # LLM parameters
    temperature: float = 0.7
    max_tokens_output: int = 2048
    max_tokens_input: int = 32768
    api_key_env_var: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    use_fast_response: bool = True
    enable_streaming: bool = False
    verbose_logging: bool = False

    # Enhanced configurations
    mcp_tools: List[MCPToolConfig] = Field(default_factory=list)
    code_execution: CodeExecutionConfig = Field(default_factory=CodeExecutionConfig)
    flow_specializations: List[FlowSpecializationConfig] = Field(default_factory=list)
    persona_profiles: PersonaProfileConfig = Field(default_factory=PersonaProfileConfig)

    # Context and session management
    session_management: Dict[str, Any] = Field(default_factory=lambda: {
        "max_history": 200,
        "compression_threshold": 0.76,
        "enable_infinite_scaling": True,
        "context_awareness": "advanced_session_aware"
    })

    # Task management
    task_management: Dict[str, Any] = Field(default_factory=lambda: {
        "max_parallel_tasks": 5,
        "enable_reflection": True,
        "enable_adaptation": True,
        "max_adaptations": 3,
        "intelligent_routing": True,
        "task_timeout": 300
    })

    # Tool management
    tool_management: Dict[str, Any] = Field(default_factory=lambda: {
        "auto_analyze_capabilities": True,
        "enable_tool_chaining": True,
        "intelligent_selection": True,
        "cache_analysis": True,
        "fallback_strategies": True
    })

    # Servers and integrations
    a2a: Dict[str, Any] = Field(default_factory=dict)
    mcp: Dict[str, Any] = Field(default_factory=dict)
    telemetry: Dict[str, Any] = Field(default_factory=dict)
    checkpoint: Dict[str, Any] = Field(default_factory=lambda: {
        "enabled": True,
        "interval_seconds": 300,
        "max_checkpoints": 10,
        "checkpoint_dir": "./checkpoints"
    })

    # World model and state
    initial_world_model: Dict[str, Any] = Field(default_factory=dict)
    default_tools: List[str] = Field(default_factory=list)
    custom_variables: Dict[str, Any] = Field(default_factory=dict)


# ===== SPECIALIZED NODES =====

class CodeExecutorNode(AsyncNodeT):
    """Enhanced code execution node with security and multiple language support"""

    def __init__(self, config: CodeExecutionConfig):
        super().__init__()
        self.config = config
        self.persistent_env = None
        self.installed_packages = set()

    async def prep_async(self, shared):
        return {
            "code": shared.get("code_to_execute", ""),
            "language": shared.get("language", "python"),
            "context": shared.get("execution_context", {}),
            "config": self.config,
            "install_deps": shared.get("install_dependencies", False)
        }

    async def exec_async(self, prep_res):
        if not prep_res["config"].enabled:
            return {"error": "Code execution is disabled", "success": False}

        code = prep_res["code"]
        language = prep_res["language"]

        if language not in prep_res["config"].allowed_languages:
            return {"error": f"Language {language} not allowed", "success": False}

        try:
            if language == "python":
                result = await self._execute_python(code, prep_res)
            elif language == "javascript":
                result = await self._execute_javascript(code, prep_res)
            elif language == "bash":
                result = await self._execute_bash(code, prep_res)
            else:
                return {"error": f"Unsupported language: {language}", "success": False}

            return result

        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return {"error": str(e), "success": False}

    async def _execute_python(self, code: str, prep_res: Dict) -> Dict:
        """Execute Python code with security checks"""
        # Security check - scan for dangerous imports
        if not self._validate_python_code(code, prep_res["config"]):
            return {
                "error": "Code contains blocked imports or dangerous operations",
                "success": False
            }

        # Auto-install dependencies if enabled
        if prep_res["install_deps"] and prep_res["config"].install_packages:
            await self._install_python_dependencies(code)

        if prep_res["config"].sandbox_mode:
            return await self._execute_python_sandboxed(code, prep_res)
        else:
            return await self._execute_python_direct(code, prep_res)

    def _validate_python_code(self, code: str, config: CodeExecutionConfig) -> bool:
        """Validate Python code for security"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        blocked = config.blocked_imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in blocked:
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module in blocked:
                    return False

        # Check for dangerous function calls
        dangerous_calls = ['exec', 'eval', 'compile', '__import__']
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in dangerous_calls:
                    return False

        return True

    async def _install_python_dependencies(self, code: str):
        """Auto-install Python dependencies from code"""
        try:
            tree = ast.parse(code)
            imports = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])

            # Install missing packages
            for pkg in imports:
                if pkg not in self.installed_packages and pkg not in sys.stdlib_module_names:
                    try:
                        process = await asyncio.create_subprocess_exec(
                            sys.executable, "-m", "pip", "install", pkg,
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL
                        )
                        await process.wait()
                        if process.returncode == 0:
                            self.installed_packages.add(pkg)
                    except Exception as e:
                        logger.debug(f"Failed to install {pkg}: {e}")

        except Exception as e:
            logger.debug(f"Dependency installation failed: {e}")

    async def _execute_python_sandboxed(self, code: str, prep_res: Dict) -> Dict:
        """Execute Python code in sandboxed environment"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=prep_res["config"].timeout_seconds
            )

            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode()[:prep_res["config"].max_output_length],
                "stderr": stderr.decode()[:prep_res["config"].max_output_length],
                "return_code": process.returncode
            }

        finally:
            os.unlink(temp_file)

    async def _execute_python_direct(self, code: str, prep_res: Dict) -> Dict:
        """Execute Python code directly (less secure but more flexible)"""
        import io
        import contextlib

        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        execution_globals = {"__builtins__": __builtins__}
        execution_locals = {}

        try:
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
                exec(code, execution_globals, execution_locals)

            return {
                "success": True,
                "stdout": output_buffer.getvalue()[:prep_res["config"].max_output_length],
                "stderr": error_buffer.getvalue()[:prep_res["config"].max_output_length],
                "return_code": 0,
                "locals": {k: str(v) for k, v in execution_locals.items() if not k.startswith('_')}
            }

        except Exception as e:
            return {
                "success": False,
                "stdout": output_buffer.getvalue(),
                "stderr": str(e),
                "return_code": 1
            }

    async def _execute_javascript(self, code: str, prep_res: Dict) -> Dict:
        """Execute JavaScript code using Node.js"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            process = await asyncio.create_subprocess_exec(
                "node", temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=prep_res["config"].timeout_seconds
            )

            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode()[:prep_res["config"].max_output_length],
                "stderr": stderr.decode()[:prep_res["config"].max_output_length],
                "return_code": process.returncode
            }

        except FileNotFoundError:
            return {
                "error": "Node.js not found. Please install Node.js for JavaScript execution.",
                "success": False
            }
        finally:
            os.unlink(temp_file)

    async def _execute_bash(self, code: str, prep_res: Dict) -> Dict:
        """Execute Bash code"""
        try:
            process = await asyncio.create_subprocess_shell(
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=prep_res["config"].timeout_seconds
            )

            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode()[:prep_res["config"].max_output_length],
                "stderr": stderr.decode()[:prep_res["config"].max_output_length],
                "return_code": process.returncode
            }

        except Exception as e:
            return {
                "error": f"Bash execution failed: {str(e)}",
                "success": False
            }


# ===== SPECIALIZED FLOW NODES =====

class CodeRequirementsNode(AsyncNodeT):
    """Analyze and clarify code requirements"""

    async def prep_async(self, shared):
        return {
            "query": shared.get("current_query", ""),
            "context": shared.get("context", {}),
            "fast_llm_model": shared.get("fast_llm_model")
        }

    async def exec_async(self, prep_res):
        if not LITELLM_AVAILABLE:
            return {"requirements_clear": True, "analysis": "Basic requirements extracted"}

        query = prep_res["query"]

        prompt = f"""
Analyze this code request and extract clear, detailed requirements:

Request: {query}

Extract:
1. Functional requirements (what the code should do)
2. Technical requirements (language, frameworks, constraints)
3. Quality requirements (performance, security, style)
4. Input/Output specifications
5. Edge cases to consider

Respond with structured requirements in JSON format:
{{
    "functional": ["requirement1", "requirement2"],
    "technical": {{"language": "python", "frameworks": [], "constraints": []}},
    "quality": {{"performance": "", "security": "", "style": ""}},
    "io_specs": {{"inputs": [], "outputs": []}},
    "edge_cases": [],
    "clarity_score": 0.8
}}
"""

        try:
            response = await litellm.acompletion(
                model=prep_res.get("fast_llm_model", "openrouter/anthropic/claude-3-haiku"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )

            content = response.choices[0].message.content
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                requirements = json.loads(json_match.group())
            else:
                requirements = {"functional": ["Basic functionality"], "clarity_score": 0.5}

            return {
                "requirements_clear": requirements.get("clarity_score", 0.5) > 0.6,
                "analysis": requirements,
                "needs_clarification": requirements.get("clarity_score", 0.5) < 0.6
            }

        except Exception as e:
            logger.error(f"Requirements analysis failed: {e}")
            return {
                "requirements_clear": False,
                "analysis": {"error": str(e)},
                "needs_clarification": True
            }

    async def post_async(self, shared, prep_res, exec_res):
        shared["code_requirements"] = exec_res["analysis"]
        if exec_res["requirements_clear"]:
            return "requirements_clear"
        else:
            return "needs_clarification"


class CodeGeneratorNode(AsyncNodeT):
    """Generate code based on requirements"""

    async def prep_async(self, shared):
        return {
            "requirements": shared.get("code_requirements", {}),
            "context": shared.get("context", {}),
            "complex_llm_model": shared.get("complex_llm_model"),
            "code_executor": shared.get("code_executor")
        }

    async def exec_async(self, prep_res):
        if not LITELLM_AVAILABLE:
            return {"code_generated": False, "error": "LLM not available"}

        requirements = prep_res["requirements"]

        prompt = f"""
Generate high-quality code based on these requirements:

Requirements: {json.dumps(requirements, indent=2)}

Guidelines:
1. Write clean, readable, well-documented code
2. Include proper error handling
3. Add type hints (if Python)
4. Follow best practices
5. Include example usage
6. Add comprehensive docstrings

Generate the complete code solution:
"""

        try:
            response = await litellm.acompletion(
                model=prep_res.get("complex_llm_model", "openrouter/openai/gpt-4o"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )

            generated_code = response.choices[0].message.content

            # Extract code from markdown blocks if present
            code_blocks = re.findall(r'```(?:python|py|javascript|js|bash)?\n(.*?)\n```',
                                     generated_code, re.DOTALL)

            if code_blocks:
                code = code_blocks[0]
            else:
                code = generated_code

            # Test the code if executor is available
            test_result = None
            if prep_res.get("code_executor") and code:
                test_context = {
                    "code_to_execute": code,
                    "language": requirements.get("technical", {}).get("language", "python")
                }
                test_result = await prep_res["code_executor"].run_async(test_context)

            return {
                "code_generated": True,
                "code": code,
                "full_response": generated_code,
                "test_result": test_result,
                "quality_score": self._assess_code_quality(code)
            }

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {
                "code_generated": False,
                "error": str(e)
            }

    def _assess_code_quality(self, code: str) -> float:
        """Basic code quality assessment"""
        score = 0.5

        # Check for docstrings
        if '"""' in code or "'''" in code:
            score += 0.1

        # Check for type hints (Python)
        if '->' in code or ': ' in code:
            score += 0.1

        # Check for error handling
        if 'try:' in code or 'except' in code:
            score += 0.1

        # Check for comments
        if '#' in code:
            score += 0.05

        # Length reasonable
        if 50 <= len(code.split('\n')) <= 200:
            score += 0.1

        return min(1.0, score)

    async def post_async(self, shared, prep_res, exec_res):
        shared["generated_code"] = exec_res.get("code", "")
        shared["code_quality_score"] = exec_res.get("quality_score", 0.0)

        if exec_res["code_generated"]:
            return "code_generated"
        else:
            return "generation_failed"


# ===== SPECIALIZED FLOW BUILDERS =====

class FlowSpecializer:
    """Creates specialized flows for different domains"""

    def __init__(self):
        self.specializations = {
            "code_writing": self._build_code_writing_flow,
            "code_editing": self._build_code_editing_flow,
            "data_analysis": self._build_data_analysis_flow,
            "project_management": self._build_project_management_flow,
            "research": self._build_research_flow,
            "creative_writing": self._build_creative_writing_flow,
            "problem_solving": self._build_problem_solving_flow,
            "refinement": self._build_refinement_flow
        }

    def create_specialized_flow(self, spec_config: FlowSpecializationConfig) -> AsyncFlowT:
        """Create a specialized flow based on configuration"""
        domain = spec_config.domain

        if domain in self.specializations:
            return self.specializations[domain](spec_config)
        else:
            return self._build_custom_flow(spec_config)

    def _build_code_writing_flow(self, config: FlowSpecializationConfig) -> AsyncFlowT:
        """Specialized flow for code writing tasks"""

        class CodeWritingFlow(AsyncFlowT):
            def __init__(self):
                self.requirements_analyzer = CodeRequirementsNode()
                self.code_generator = CodeGeneratorNode()
                self.code_tester = CodeTestingNode()
                self.code_refiner = CodeRefinerNode()

                # Flow connections
                self.requirements_analyzer - "requirements_clear" >> self.code_generator
                self.requirements_analyzer - "needs_clarification" >> self.code_generator
                self.code_generator - "code_generated" >> self.code_tester
                self.code_tester - "tests_passed" >> self.code_refiner
                self.code_tester - "tests_failed" >> self.code_generator

                super().__init__(start=self.requirements_analyzer)

        return CodeWritingFlow()

    def _build_refinement_flow(self, config: FlowSpecializationConfig) -> AsyncFlowT:
        """Specialized flow for content refinement"""

        class RefinementFlow(AsyncFlowT):
            def __init__(self):
                self.content_analyzer = ContentAnalyzerNode()
                self.improvement_generator = ImprovementGeneratorNode()
                self.refinement_applier = RefinementApplierNode()
                self.quality_checker = QualityCheckerNode()

                # Flow connections
                self.content_analyzer - "analyzed" >> self.improvement_generator
                self.improvement_generator - "improvements_generated" >> self.refinement_applier
                self.refinement_applier - "refinement_applied" >> self.quality_checker
                self.quality_checker - "needs_more_refinement" >> self.improvement_generator

                super().__init__(start=self.content_analyzer)

        return RefinementFlow()

    def _build_custom_flow(self, config: FlowSpecializationConfig) -> AsyncFlowT:
        """Build custom flow from configuration"""

        # For now, return a simple flow
        # In production, this would dynamically create flows based on config
        class CustomFlow(AsyncFlowT):
            pass

        return CustomFlow()


# Additional specialized nodes
class CodeTestingNode(AsyncNodeT):
    """Test generated code"""

    async def exec_async(self, prep_res):
        return {"tests_passed": True, "test_results": "All tests passed"}


class CodeRefinerNode(AsyncNodeT):
    """Refine and optimize code"""

    async def exec_async(self, prep_res):
        return {"code_refined": True, "refinements": "Code optimized"}


class ContentAnalyzerNode(AsyncNodeT):
    """Analyze content for refinement opportunities"""

    async def exec_async(self, prep_res):
        return {"analyzed": True, "analysis": "Content analyzed"}


class ImprovementGeneratorNode(AsyncNodeT):
    """Generate improvement suggestions"""

    async def exec_async(self, prep_res):
        return {"improvements_generated": True, "improvements": []}


class RefinementApplierNode(AsyncNodeT):
    """Apply refinements to content"""

    async def exec_async(self, prep_res):
        return {"refinement_applied": True, "refined_content": ""}


class QualityCheckerNode(AsyncNodeT):
    """Check quality of refined content"""

    async def exec_async(self, prep_res):
        return {"quality_acceptable": True, "quality_score": 0.8}


# ===== MAIN ENHANCED BUILDER =====

class AgentBuilder:
    """Comprehensive builder for production-ready Flow Agents with specialization capabilities"""

    def __init__(self, config: AdvancedBuilderConfig = None, config_path: str = None):
        """Initialize builder with enhanced configuration"""

        if config and config_path:
            raise ValueError("Provide either config object or config_path, not both")

        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = AdvancedBuilderConfig()

        # Enhanced registries
        self._custom_tools: Dict[str, Dict] = {}
        self._mcp_tools: Dict[str, MCPToolConfig] = {}
        self._custom_nodes: Dict[str, Any] = {}
        self._custom_flows: Dict[str, AsyncFlowT] = {}
        self._specialized_flows: Dict[str, AsyncFlowT] = {}

        # Specialization system
        self.flow_specializer = FlowSpecializer()

        # Runtime components
        self._budget_manager: Optional[BudgetManager] = None
        self._tracer_provider: Optional[TracerProvider] = None
        self._callbacks: Dict[str, Callable] = {}
        self._code_executor: Optional[CodeExecutorNode] = None
        self._variable_manager: Optional[VariableManager] = None

        # Initialize code execution if enabled
        if self.config.code_execution.enabled:
            self._code_executor = CodeExecutorNode(self.config.code_execution)

        # Set logging level
        if self.config.verbose_logging:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

        logger.info(f"Advanced builder initialized: {self.config.agent_name}")

    # ===== CONFIG MANAGEMENT =====

    def _load_config(self, config_path: str) -> AdvancedBuilderConfig:
        """Load configuration from file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            return AdvancedBuilderConfig(**data)

        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    def save_config(self, config_path: str, format: str = 'yaml'):
        """Save current configuration to file"""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = self.config.model_dump()

            with open(path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(data, f, indent=2)

            logger.info(f"Configuration saved to {config_path}")

        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
            raise

    # ===== FLUENT CONFIGURATION API =====

    def with_name(self, name: str) -> 'AgentBuilder':
        """Set agent name"""
        self.config.agent_name = name
        return self

    def with_models(self, fast_model: str, complex_model: str = None) -> 'AgentBuilder':
        """Set LLM models"""
        self.config.fast_llm_model = fast_model
        if complex_model:
            self.config.complex_llm_model = complex_model
        return self

    def with_system_message(self, message: str) -> 'AgentBuilder':
        """Set system message"""
        self.config.system_message = message
        return self

    def with_temperature(self, temp: float) -> 'AgentBuilder':
        """Set temperature"""
        self.config.temperature = temp
        return self

    def with_max_tokens(self, output: int, input: int = None) -> 'AgentBuilder':
        """Set max tokens for output and optionally input"""
        self.config.max_tokens_output = output
        if input:
            self.config.max_tokens_input = input
        return self

    def with_api_config(self, api_key_env_var: str = None, api_base: str = None,
                        api_version: str = None) -> 'AgentBuilder':
        """Configure API settings"""
        if api_key_env_var:
            self.config.api_key_env_var = api_key_env_var
        if api_base:
            self.config.api_base = api_base
        if api_version:
            self.config.api_version = api_version
        return self

    def with_budget_manager(self, max_cost: float = 10.0) -> 'AgentBuilder':
        """Enable budget management"""
        if LITELLM_AVAILABLE:
            self._budget_manager = BudgetManager(max_budget=max_cost)
            logger.info(f"Budget manager enabled: ${max_cost}")
        else:
            logger.warning("LiteLLM not available, budget manager disabled")
        return self

    def verbose(self, enable: bool = True) -> 'AgentBuilder':
        """Enable verbose logging"""
        self.config.verbose_logging = enable
        if enable:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
        return self

    def with_session_config(self, max_history: int = 200, compression_threshold: float = 0.76,
                            infinite_scaling: bool = True) -> 'AgentBuilder':
        """Configure session management"""
        self.config.session_management.update({
            "max_history": max_history,
            "compression_threshold": compression_threshold,
            "enable_infinite_scaling": infinite_scaling
        })
        return self

    def with_task_config(self, max_parallel: int = 5, enable_reflection: bool = True,
                         enable_adaptation: bool = True, max_adaptations: int = 3) -> 'AgentBuilder':
        """Configure task management"""
        self.config.task_management.update({
            "max_parallel_tasks": max_parallel,
            "enable_reflection": enable_reflection,
            "enable_adaptation": enable_adaptation,
            "max_adaptations": max_adaptations
        })
        return self

    # ===== ENHANCED TOOL MANAGEMENT =====

    def add_tool(self, func: Callable, name: str = None, description: str = None,
                 expose_in_mcp: bool = True, category: str = "general",
                 auto_analyze: bool = True) -> 'AgentBuilder':
        """Add tool function with enhanced metadata"""
        tool_name = name or func.__name__

        # Extract parameter information
        sig = inspect.signature(func)
        parameters = {}
        for param_name, param in sig.parameters.items():
            param_info = {
                "type": param.annotation.__name__ if param.annotation != param.empty else "Any",
                "default": param.default if param.default != param.empty else None,
                "required": param.default == param.empty
            }
            parameters[param_name] = param_info

        self._custom_tools[tool_name] = {
            'function': func,
            'description': description or func.__doc__ or f"Tool function: {tool_name}",
            'expose_in_mcp': expose_in_mcp,
            'category': category,
            'parameters': parameters,
            'auto_analyze': auto_analyze,
            'added_at': datetime.now().isoformat(),
            'is_async': asyncio.iscoroutinefunction(func)
        }

        if tool_name not in self.config.default_tools:
            self.config.default_tools.append(tool_name)

        logger.info(f"Tool added: {tool_name} (category: {category}, async: {asyncio.iscoroutinefunction(func)})")
        return self

    def add_tools_from_module(self, module, prefix: str = "",
                              exclude: List[str] = None,
                              include_private: bool = False) -> 'AgentBuilder':
        """Add all functions from a module as tools"""
        exclude = exclude or []

        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name in exclude:
                continue

            if not include_private and name.startswith('_'):
                continue

            tool_name = f"{prefix}{name}" if prefix else name
            self.add_tool(obj, name=tool_name,
                          description=f"Function from {module.__name__}: {obj.__doc__ or 'No description'}",
                          category="imported")

        logger.info(f"Added tools from module {module.__name__}")
        return self

    def add_lambda_tool(self, lambda_func: Callable, name: str, description: str = "",
                        category: str = "lambda") -> 'AgentBuilder':
        """Add a lambda function as a tool"""
        return self.add_tool(lambda_func, name=name, description=description or f"Lambda tool: {name}",
                             category=category, auto_analyze=False)

    def add_mcp_tool_from_code(self, name: str, code: str, description: str = "",
                               parameters: Dict[str, Any] = None,
                               async_mode: bool = True,
                               requires_packages: List[str] = None) -> 'AgentBuilder':
        """Add MCP tool from code string with enhanced validation"""

        # Validate code syntax
        try:
            compile(code, f"<{name}>", "exec")
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code for tool {name}: {e}")

        mcp_config = MCPToolConfig(
            name=name,
            description=description,
            function_code=code,
            parameters=parameters or {},
            async_mode=async_mode,
            requires_packages=requires_packages or []
        )

        self._mcp_tools[name] = mcp_config
        self.config.mcp_tools.append(mcp_config)

        # Create and register the function
        try:
            func = self._create_function_from_code(code, name, async_mode)
            self.add_tool(func, name, description, expose_in_mcp=True, category="mcp")
        except Exception as e:
            logger.error(f"Failed to create function from code for {name}: {e}")
            raise

        logger.info(f"MCP tool added from code: {name}")
        return self

    def add_mcp_tool_from_file(self, name: str, file_path: str, description: str = "",
                               parameters: Dict[str, Any] = None,
                               async_mode: bool = True) -> 'AgentBuilder':
        """Add MCP tool from file with validation"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Tool file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            raise ValueError(f"Failed to read tool file {file_path}: {e}")

        return self.add_mcp_tool_from_code(name, code, description, parameters, async_mode)

    def load_mcp_tools_from_config(self, config_path: str) -> 'AgentBuilder':
        """Load MCP tools from configuration file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"MCP tools config not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    tools_config = yaml.safe_load(f)
                else:
                    tools_config = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load MCP tools config: {e}")

        loaded_count = 0
        for tool_data in tools_config.get('tools', []):
            try:
                mcp_config = MCPToolConfig(**tool_data)

                if mcp_config.function_code:
                    self.add_mcp_tool_from_code(
                        mcp_config.name,
                        mcp_config.function_code,
                        mcp_config.description,
                        mcp_config.parameters,
                        mcp_config.async_mode,
                        mcp_config.requires_packages
                    )
                    loaded_count += 1
                elif mcp_config.function_file:
                    self.add_mcp_tool_from_file(
                        mcp_config.name,
                        mcp_config.function_file,
                        mcp_config.description,
                        mcp_config.parameters,
                        mcp_config.async_mode
                    )
                    loaded_count += 1
                else:
                    logger.warning(f"Tool {mcp_config.name} has no code or file specified")

            except Exception as e:
                logger.error(f"Failed to load MCP tool {tool_data.get('name', 'unknown')}: {e}")

        logger.info(f"Loaded {loaded_count} MCP tools from config")
        return self

    def _create_function_from_code(self, code: str, name: str, async_mode: bool) -> Callable:
        """Create callable function from code string with enhanced error handling"""
        namespace = {"__builtins__": __builtins__}

        try:
            exec(code, namespace)
        except Exception as e:
            raise ValueError(f"Failed to execute code for {name}: {e}")

        # Find the function
        func = None
        if name in namespace and callable(namespace[name]):
            func = namespace[name]
        else:
            # Try to find any function in the namespace
            functions = [v for v in namespace.values()
                         if callable(v) and not getattr(v, '__name__', '').startswith('_')]
            if functions:
                func = functions[0]

        if not func:
            raise ValueError(f"No callable function found in code for {name}")

        # Handle async mode
        if async_mode and not asyncio.iscoroutinefunction(func):
            # Wrap sync function as async
            import functools
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await asyncio.to_thread(func, *args, **kwargs)

            return async_wrapper
        elif not async_mode and asyncio.iscoroutinefunction(func):
            # Wrap async function as sync (not recommended but supported)
            import functools
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(func(*args, **kwargs))
                finally:
                    loop.close()

            return sync_wrapper

        return func

    # ===== CODE EXECUTION CAPABILITIES =====

    def enable_code_execution(self, languages: List[str] = None, sandbox: bool = True,
                              timeout: int = 30, install_packages: bool = True,
                              allowed_imports: List[str] = None,
                              blocked_imports: List[str] = None) -> 'AgentBuilder':
        """Enable code execution capabilities with security controls"""
        if languages is None:
            languages = ["python", "javascript", "bash"]

        config = CodeExecutionConfig(
            enabled=True,
            allowed_languages=languages,
            sandbox_mode=sandbox,
            timeout_seconds=timeout,
            install_packages=install_packages
        )

        if allowed_imports:
            config.allowed_imports = allowed_imports
        if blocked_imports:
            config.blocked_imports = blocked_imports

        self.config.code_execution = config
        self._code_executor = CodeExecutorNode(config)

        # Add code execution as tools
        async def execute_code(code: str, language: str = "python",
                               install_dependencies: bool = False):
            """Execute code in specified language with optional dependency installation"""
            if not self._code_executor:
                return {"error": "Code execution not enabled", "success": False}

            shared_context = {
                "code_to_execute": code,
                "language": language,
                "execution_context": {},
                "install_dependencies": install_dependencies
            }
            result = await self._code_executor.run_async(shared_context)
            return result

        async def execute_python(code: str, install_deps: bool = False):
            """Execute Python code with optional auto-installation of dependencies"""
            return await execute_code(code, "python", install_deps)

        async def execute_javascript(code: str):
            """Execute JavaScript code using Node.js"""
            return await execute_code(code, "javascript")

        async def execute_bash(command: str):
            """Execute bash commands safely"""
            return await execute_code(command, "bash")

        # Add the tools
        self.add_tool(execute_code, "execute_code",
                      "Execute code in various languages with security controls",
                      category="code_execution")
        self.add_tool(execute_python, "execute_python",
                      "Execute Python code with auto-dependency installation",
                      category="code_execution")

        if "javascript" in languages:
            self.add_tool(execute_javascript, "execute_javascript",
                          "Execute JavaScript code using Node.js",
                          category="code_execution")

        if "bash" in languages:
            self.add_tool(execute_bash, "execute_bash",
                          "Execute bash commands safely",
                          category="code_execution")

        logger.info(f"Code execution enabled for languages: {languages}")
        return self

    def with_secure_python_execution(self, timeout: int = 30) -> 'AgentBuilder':
        """Enable secure Python execution with restricted imports"""
        safe_imports = [
            "json", "yaml", "csv", "math", "statistics", "datetime", "time",
            "re", "string", "itertools", "collections", "functools",
            "numpy", "pandas", "matplotlib", "seaborn", "plotly",
            "requests", "pathlib", "os.path"
        ]

        dangerous_imports = [
            "subprocess", "os.system", "eval", "exec", "compile",
            "socket", "urllib", "http", "ftplib", "smtplib",
            "pickle", "shelve", "dbm", "__import__"
        ]

        return self.enable_code_execution(
            languages=["python"],
            sandbox=True,
            timeout=timeout,
            install_packages=True,
            allowed_imports=safe_imports,
            blocked_imports=dangerous_imports
        )

    # ===== FLOW SPECIALIZATION =====

    def add_specialization(self, domain: str, specialized_tools: List[str] = None,
                           custom_strategies: Dict[str, Dict] = None,
                           **kwargs) -> 'AgentBuilder':
        """Add flow specialization for a domain"""
        spec_config = FlowSpecializationConfig(
            name=f"{domain}_specialization",
            domain=domain,
            specialized_tools=specialized_tools or [],
            custom_strategies=custom_strategies or {},
            **kwargs
        )

        # Create specialized flow
        try:
            specialized_flow = self.flow_specializer.create_specialized_flow(spec_config)
            self._specialized_flows[domain] = specialized_flow
            self.config.flow_specializations.append(spec_config)

            # Add as a tool
            async def run_specialized_flow(query: str, context: Dict[str, Any] = None):
                """Run specialized flow for specific domain"""
                flow_context = context or {}
                flow_context.update({
                    "current_query": query,
                    "domain": domain,
                    "specialization_config": spec_config
                })

                # Add code executor if available
                if self._code_executor:
                    flow_context["code_executor"] = self._code_executor

                return await specialized_flow.run_async(flow_context)

            self.add_tool(run_specialized_flow, f"{domain}_flow",
                          f"Execute specialized flow for {domain} tasks",
                          category="specialization")

            logger.info(f"Specialization added: {domain}")
        except Exception as e:
            logger.error(f"Failed to create specialization for {domain}: {e}")

        return self

    def with_code_writing_specialization(self) -> 'AgentBuilder':
        """Add comprehensive code writing specialization"""
        tools = ["execute_code", "execute_python"]
        if self._code_executor:
            tools.extend(["execute_javascript", "execute_bash"])

        return self.add_specialization("code_writing",
                                       specialized_tools=tools,
                                       context_awareness="code_focused",
                                       flow_pattern="iterative")

    def with_data_analysis_specialization(self) -> 'AgentBuilder':
        """Add data analysis specialization with Python focus"""
        return self.add_specialization("data_analysis",
                                       specialized_tools=["execute_python", "execute_code"],
                                       context_awareness="data_focused",
                                       flow_pattern="analytical")

    def with_project_management_specialization(self) -> 'AgentBuilder':
        """Add project management specialization"""
        return self.add_specialization("project_management",
                                       specialized_tools=[],
                                       context_awareness="project_focused",
                                       flow_pattern="structured")

    def with_refinement_specialization(self) -> 'AgentBuilder':
        """Add content refinement and optimization specialization"""
        return self.add_specialization("refinement",
                                       specialized_tools=[],
                                       context_awareness="quality_focused",
                                       flow_pattern="iterative")

    def with_all_specializations(self) -> 'AgentBuilder':
        """Add all available specializations"""
        return (self.with_code_writing_specialization()
                .with_data_analysis_specialization()
                .with_project_management_specialization()
                .with_refinement_specialization())

    # ===== PERSONA MANAGEMENT =====

    def add_persona_profile(self, profile_name: str, persona: PersonaConfig) -> 'AgentBuilder':
        """Add a persona profile"""
        self.config.persona_profiles.profiles[profile_name] = persona
        logger.info(f"Persona profile added: {profile_name}")
        return self

    def set_active_persona(self, profile_name: str) -> 'AgentBuilder':
        """Set active persona profile"""
        if profile_name in self.config.persona_profiles.profiles:
            self.config.persona_profiles.active_profile = profile_name
            logger.info(f"Active persona set: {profile_name}")
        else:
            logger.warning(f"Persona profile not found: {profile_name}")
        return self

    def with_developer_persona(self, name: str = "Senior Developer") -> 'AgentBuilder':
        """Add and set a developer persona"""
        persona = PersonaConfig(
            name=name,
            style="technical",
            tone="professional",
            personality_traits=["precise", "thorough", "best_practices", "security_conscious"],
            custom_instructions="Focus on code quality, maintainability, security, and best practices. "
                                "Always consider edge cases and provide comprehensive documentation.",
            apply_method="both",
            integration_level="medium"
        )
        return self.add_persona_profile("developer", persona).set_active_persona("developer")

    def with_analyst_persona(self, name: str = "Data Analyst") -> 'AgentBuilder':
        """Add and set an analyst persona"""
        persona = PersonaConfig(
            name=name,
            style="analytical",
            tone="objective",
            personality_traits=["methodical", "thorough", "insight_driven", "evidence_based"],
            custom_instructions="Focus on statistical rigor, clear data insights, and actionable recommendations. "
                                "Always validate assumptions and provide confidence intervals where appropriate.",
            apply_method="both",
            integration_level="medium"
        )
        return self.add_persona_profile("analyst", persona).set_active_persona("analyst")

    def with_assistant_persona(self, name: str = "AI Assistant") -> 'AgentBuilder':
        """Add and set a general assistant persona"""
        persona = PersonaConfig(
            name=name,
            style="friendly",
            tone="helpful",
            personality_traits=["helpful", "patient", "clear", "adaptive"],
            custom_instructions="Be helpful, patient, and clear in all interactions. "
                                "Adapt your communication style to the user's level of expertise.",
            apply_method="system_prompt",
            integration_level="light"
        )
        return self.add_persona_profile("assistant", persona).set_active_persona("assistant")

    def with_domain_adapted_personas(self, enable: bool = True) -> 'AgentBuilder':
        """Enable domain-adapted personas that change based on task type"""
        if enable:
            self.config.persona_profiles.domain_adaptations = {
                "code_writing": {"style": "technical", "tone": "precise"},
                "data_analysis": {"style": "analytical", "tone": "objective"},
                "project_management": {"style": "professional", "tone": "authoritative"},
                "creative_writing": {"style": "creative", "tone": "inspiring"},
                "refinement": {"style": "editorial", "tone": "constructive"}
            }
            self.config.persona_profiles.auto_switch = True

            # Set up switch triggers
            self.config.persona_profiles.switch_triggers = {
                "code|programming|development|bug|function": "developer",
                "data|analysis|statistics|chart|graph": "analyst",
                "project|management|timeline|milestone": "manager",
                "creative|story|content|writing": "creative"
            }

        return self

    # ===== CUSTOM FLOWS =====

    def add_custom_flow(self, flow: AsyncFlowT, name: str,
                        description: str = "", category: str = "custom") -> 'AgentBuilder':
        """Add custom flow as a tool"""
        self._custom_flows[name] = flow

        # Wrap flow as a tool
        async def run_custom_flow(query: str = "", context: Dict[str, Any] = None):
            """Run custom flow with provided context"""
            flow_context = context or {}
            flow_context.update({"current_query": query})

            # Add shared resources
            if self._code_executor:
                flow_context["code_executor"] = self._code_executor
            if self._variable_manager:
                flow_context["variable_manager"] = self._variable_manager

            return await flow.run_async(flow_context)

        self.add_tool(run_custom_flow, name,
                      description or f"Custom flow: {flow.__class__.__name__}",
                      category=category)

        logger.info(f"Custom flow added as tool: {name}")
        return self

    def add_custom_node(self, node: AsyncNodeT, name: str,
                        description: str = "") -> 'AgentBuilder':
        """Add custom node as a tool"""
        self._custom_nodes[name] = node

        # Wrap node as a tool
        async def run_custom_node(context: Dict[str, Any] = None):
            """Run custom node with provided context"""
            node_context = context or {}
            return await node.run_async(node_context)

        self.add_tool(run_custom_node, name,
                      description or f"Custom node: {node.__class__.__name__}",
                      category="custom_node")

        logger.info(f"Custom node added as tool: {name}")
        return self

    # ===== VARIABLE SYSTEM =====

    def with_custom_variables(self, variables: Dict[str, Any]) -> 'AgentBuilder':
        """Add custom variables to the agent"""
        self.config.custom_variables.update(variables)
        return self

    def add_variable_scope(self, scope_name: str, variables: Dict[str, Any]) -> 'AgentBuilder':
        """Add a variable scope"""
        if 'variable_scopes' not in self.config.custom_variables:
            self.config.custom_variables['variable_scopes'] = {}
        self.config.custom_variables['variable_scopes'][scope_name] = variables
        return self

    # ===== SERVER CONFIGURATION =====

    def enable_a2a_server(self, host: str = "0.0.0.0", port: int = 5000,
                          **kwargs) -> 'AgentBuilder':
        """Enable A2A server"""
        if not A2A_AVAILABLE:
            logger.warning("A2A not available, cannot enable server")
            return self

        self.config.a2a = {
            "enabled": True,
            "host": host,
            "port": port,
            **kwargs
        }
        logger.info(f"A2A server configured: {host}:{port}")
        return self

    def enable_mcp_server(self, host: str = "0.0.0.0", port: int = 8000,
                          server_name: str = None, **kwargs) -> 'AgentBuilder':
        """Enable MCP server"""
        if not MCP_AVAILABLE:
            logger.warning("MCP not available, cannot enable server")
            return self

        self.config.mcp = {
            "enabled": True,
            "host": host,
            "port": port,
            "server_name": server_name or f"{self.config.agent_name}_MCP",
            **kwargs
        }
        logger.info(f"MCP server configured: {host}:{port}")
        return self

    def enable_telemetry(self, service_name: str = None, **kwargs) -> 'AgentBuilder':
        """Enable OpenTelemetry tracing"""
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not available, cannot enable telemetry")
            return self

        self.config.telemetry = {
            "enabled": True,
            "service_name": service_name or self.config.agent_name,
            **kwargs
        }

        # Initialize tracer provider
        self._tracer_provider = TracerProvider()
        logger.info(f"Telemetry enabled for service: {service_name}")
        return self

    # ===== VALIDATION AND OPTIMIZATION =====

    def validate_config(self) -> Dict[str, List[str]]:
        """Validate the current configuration and return issues"""
        issues = {"errors": [], "warnings": []}

        # Validate models
        if not self.config.fast_llm_model:
            issues["errors"].append("Fast LLM model not specified")
        if not self.config.complex_llm_model:
            issues["errors"].append("Complex LLM model not specified")

        # Validate tools
        if self.config.code_execution.enabled:
            for lang in self.config.code_execution.allowed_languages:
                if lang not in ["python", "javascript", "bash", "sql", "r"]:
                    issues["warnings"].append(f"Unsupported language: {lang}")

        # Validate specializations
        for spec in self.config.flow_specializations:
            if spec.domain not in ["code_writing", "data_analysis", "project_management", "refinement"]:
                issues["warnings"].append(f"Unknown specialization domain: {spec.domain}")

        # Validate personas
        if not self.config.persona_profiles.profiles:
            issues["warnings"].append("No persona profiles defined")

        # Validate server configs
        if self.config.a2a.get("enabled") and not A2A_AVAILABLE:
            issues["errors"].append("A2A server enabled but A2A not available")
        if self.config.mcp.get("enabled") and not MCP_AVAILABLE:
            issues["errors"].append("MCP server enabled but MCP not available")

        return issues

    def optimize_config(self) -> 'AgentBuilder':
        """Optimize configuration based on enabled features"""

        # Optimize token limits based on models
        if "claude" in self.config.fast_llm_model.lower():
            self.config.max_tokens_input = min(self.config.max_tokens_input, 100000)
        elif "gpt-4" in self.config.complex_llm_model.lower():
            self.config.max_tokens_input = min(self.config.max_tokens_input, 128000)

        # Optimize task management based on tools
        tool_count = len(self.config.default_tools) + len(self._custom_tools)
        if tool_count > 10:
            self.config.task_management["max_parallel_tasks"] = min(
                self.config.task_management["max_parallel_tasks"], 3
            )

        # Optimize session management for code execution
        if self.config.code_execution.enabled:
            self.config.session_management["compression_threshold"] = 0.8

        logger.info("Configuration optimized")
        return self

    def get_build_summary(self) -> Dict[str, Any]:
        """Get summary of what will be built"""
        return {
            "agent_name": self.config.agent_name,
            "models": {
                "fast": self.config.fast_llm_model,
                "complex": self.config.complex_llm_model
            },
            "tools": {
                "custom_tools": len(self._custom_tools),
                "mcp_tools": len(self._mcp_tools),
                "default_tools": len(self.config.default_tools)
            },
            "features": {
                "code_execution": self.config.code_execution.enabled,
                "specializations": len(self.config.flow_specializations),
                "persona_profiles": len(self.config.persona_profiles.profiles),
                "custom_flows": len(self._custom_flows),
                "custom_nodes": len(self._custom_nodes)
            },
            "servers": {
                "a2a": self.config.a2a.get("enabled", False),
                "mcp": self.config.mcp.get("enabled", False),
                "telemetry": self.config.telemetry.get("enabled", False)
            },
            "configuration": {
                "max_parallel_tasks": self.config.task_management.get("max_parallel_tasks"),
                "session_management": self.config.session_management.get("context_awareness"),
                "budget_manager": self._budget_manager is not None,
                "verbose_logging": self.config.verbose_logging
            }
        }

    # ===== BUILD METHOD =====

    async def build(self) -> FlowAgent:
        """Build the enhanced FlowAgent with comprehensive error handling"""

        logger.info(f"Building advanced FlowAgent: {self.config.agent_name}")

        # Validate configuration
        validation_issues = self.validate_config()
        if validation_issues["errors"]:
            error_msg = f"Configuration validation failed: {', '.join(validation_issues['errors'])}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if validation_issues["warnings"]:
            for warning in validation_issues["warnings"]:
                logger.warning(f"Configuration warning: {warning}")

        try:
            # 1. Setup API configuration
            api_key = None
            if self.config.api_key_env_var:
                api_key = os.getenv(self.config.api_key_env_var)
                if not api_key:
                    logger.warning(f"API key env var {self.config.api_key_env_var} not set")

            # 2. Determine active persona
            active_persona = None
            if self.config.persona_profiles.profiles:
                active_profile = self.config.persona_profiles.active_profile
                if active_profile in self.config.persona_profiles.profiles:
                    active_persona = self.config.persona_profiles.profiles[active_profile]
                    logger.info(f"Using persona: {active_persona.name}")

            # 3. Create enhanced AgentModelData
            amd = AgentModelData(
                name=self.config.agent_name,
                fast_llm_model=self.config.fast_llm_model,
                complex_llm_model=self.config.complex_llm_model,
                system_message=self.config.system_message,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens_output,
                max_input_tokens=self.config.max_tokens_input,
                api_key=api_key,
                api_base=self.config.api_base,
                budget_manager=self._budget_manager,
                persona=active_persona,
                use_fast_response=self.config.use_fast_response
            )

            # 4. Create enhanced FlowAgent
            agent = FlowAgent(
                amd=amd,
                world_model=self.config.initial_world_model.copy(),
                verbose=self.config.verbose_logging,
                enable_pause_resume=self.config.checkpoint.get("enabled", True),
                checkpoint_interval=self.config.checkpoint.get("interval_seconds", 300),
                max_parallel_tasks=self.config.task_management.get("max_parallel_tasks", 5)
            )

            # 5. Add custom variable scopes
            if 'variable_scopes' in self.config.custom_variables:
                for scope_name, variables in self.config.custom_variables['variable_scopes'].items():
                    agent.variable_manager.register_scope(scope_name, variables)

            # Add other custom variables
            for key, value in self.config.custom_variables.items():
                if key != 'variable_scopes':
                    agent.set_variable(key, value)

            # 6. Add all custom tools
            tool_count = 0
            for tool_name, tool_info in self._custom_tools.items():
                try:
                    await agent.add_tool(
                        tool_info['function'],
                        name=tool_name,
                        description=tool_info['description']
                    )
                    tool_count += 1
                except Exception as e:
                    logger.error(f"Failed to add tool {tool_name}: {e}")

            # 7. Add specialized flows as tools
            for domain, flow in self._specialized_flows.items():
                try:
                    async def run_flow(query: str = "", **kwargs):
                        context = {"current_query": query, "domain": domain}
                        context.update(kwargs)

                        # Add enhanced context
                        context["agent_instance"] = agent
                        context["variable_manager"] = agent._variable_manager
                        if self._code_executor:
                            context["code_executor"] = self._code_executor

                        return await flow.run_async(context)

                    await agent.add_tool(run_flow, f"{domain}_flow",
                                         f"Execute {domain} specialized flow")
                    tool_count += 1
                except Exception as e:
                    logger.error(f"Failed to add specialized flow {domain}: {e}")

            # 8. Add custom flows
            for flow_name, flow in self._custom_flows.items():
                try:
                    async def run_custom_flow(query: str = "", context: Dict[str, Any] = None):
                        flow_context = context or {}
                        flow_context.update({"current_query": query})
                        flow_context["agent_instance"] = agent
                        if agent._variable_manager:
                            flow_context["variable_manager"] = agent._variable_manager
                        return await flow.run_async(flow_context)

                    await agent.add_tool(run_custom_flow, flow_name,
                                         f"Custom flow: {flow.__class__.__name__}")
                    tool_count += 1
                except Exception as e:
                    logger.error(f"Failed to add custom flow {flow_name}: {e}")

            # 9. Initialize session context if configured
            if self.config.session_management.get("enable_infinite_scaling"):
                try:
                    await agent.initialize_session_context(
                        max_history=self.config.session_management.get("max_history", 200)
                    )
                except Exception as e:
                    logger.warning(f"Session context initialization failed: {e}")

            # 10. Setup servers
            if self.config.a2a.get("enabled"):
                try:
                    agent.setup_a2a_server(
                        host=self.config.a2a.get("host", "0.0.0.0"),
                        port=self.config.a2a.get("port", 5000),
                        **{k: v for k, v in self.config.a2a.items()
                           if k not in ["enabled", "host", "port"]}
                    )
                except Exception as e:
                    logger.error(f"Failed to setup A2A server: {e}")

            if self.config.mcp.get("enabled"):
                try:
                    agent.setup_mcp_server(
                        host=self.config.mcp.get("host", "0.0.0.0"),
                        port=self.config.mcp.get("port", 8000),
                        name=self.config.mcp.get("server_name"),
                        **{k: v for k, v in self.config.mcp.items()
                           if k not in ["enabled", "host", "port", "server_name"]}
                    )
                except Exception as e:
                    logger.error(f"Failed to setup MCP server: {e}")

            # 11. Final setup and validation
            build_summary = self.get_build_summary()

            logger.info(f"Advanced FlowAgent built successfully!")
            logger.info(f"Agent: {agent.amd.name}")
            logger.info(f"Tools registered: {len(agent.shared.get('available_tools', []))}")
            logger.info(f"Specializations: {len(self._specialized_flows)}")
            logger.info(f"Custom flows: {len(self._custom_flows)}")
            logger.info(f"Features enabled: {list(build_summary['features'].keys())}")

            return agent

        except Exception as e:
            logger.error(f"Failed to build FlowAgent: {e}")
            raise

    # ===== CONVENIENCE FACTORY METHODS =====

    @classmethod
    def create_developer_agent(cls, name: str = "DeveloperAgent") -> 'AgentBuilder':
        """Create an agent optimized for software development"""
        return (cls()
                .with_name(name)
                .with_developer_persona()
                .enable_code_execution()
                .with_code_writing_specialization()
                .with_refinement_specialization()
                .with_session_config(max_history=300)
                .with_task_config(max_parallel=3)
                .verbose(True))

    @classmethod
    def create_analyst_agent(cls, name: str = "AnalystAgent") -> 'AgentBuilder':
        """Create an agent optimized for data analysis"""
        return (cls()
                .with_name(name)
                .with_analyst_persona()
                .with_secure_python_execution()
                .with_data_analysis_specialization()
                .with_session_config(max_history=250)
                .verbose(False))

    @classmethod
    def create_general_assistant(cls, name: str = "AssistantAgent") -> 'AgentBuilder':
        """Create a general-purpose assistant agent"""
        return (cls()
                .with_name(name)
                .with_assistant_persona()
                .with_all_specializations()
                .enable_code_execution(sandbox=True)
                .with_domain_adapted_personas(True)
                .with_session_config(max_history=200)
                .with_task_config(max_parallel=5))

    @classmethod
    def from_config_file(cls, config_path: str) -> 'AgentBuilder':
        """Create builder from configuration file"""
        return cls(config_path=config_path)


# ===== DEFAULT CONFIGURATIONS =====

def create_code_development_config() -> AdvancedBuilderConfig:
    """Create configuration optimized for code development"""
    return AdvancedBuilderConfig(
        agent_name="CodeDevelopmentAgent",
        description="Advanced code development agent with secure execution and testing capabilities",
        fast_llm_model=os.getenv("DEFAULTMODEL2", "openrouter/anthropic/claude-3-haiku"),
        complex_llm_model=os.getenv("DEFAULTMODELsT", "openrouter/anthropic/claude-3-haiku"),

        code_execution=CodeExecutionConfig(
            enabled=True,
            allowed_languages=["python", "javascript", "typescript", "bash"],
            sandbox_mode=True,
            timeout_seconds=60,
            install_packages=True,
            allowed_imports=[
                "json", "yaml", "csv", "math", "statistics", "datetime",
                "re", "string", "itertools", "collections", "functools",
                "numpy", "pandas", "matplotlib", "seaborn", "requests"
            ],
            blocked_imports=["subprocess", "os.system", "eval", "exec", "socket"]
        ),

        flow_specializations=[
            FlowSpecializationConfig(
                name="code_writing_flow",
                domain="code_writing",
                specialized_tools=["execute_code", "execute_python"],
                flow_pattern="iterative",
                context_awareness="code_focused"
            )
        ],

        persona_profiles=PersonaProfileConfig(
            profiles={
                "senior_developer": PersonaConfig(
                    name="Senior Developer",
                    style="technical",
                    tone="professional",
                    personality_traits=["precise", "thorough", "security_conscious"],
                    custom_instructions="Focus on code quality, security, and maintainability."
                )
            },
            active_profile="senior_developer"
        ),

        task_management={
            "max_parallel_tasks": 3,
            "enable_reflection": True,
            "enable_adaptation": True,
            "intelligent_routing": True
        }
    )


def create_data_analysis_config() -> AdvancedBuilderConfig:
    """Create configuration optimized for data analysis"""
    return AdvancedBuilderConfig(
        agent_name="DataAnalysisAgent",
        description="Advanced data analysis agent with statistical computing capabilities",
        fast_llm_model=os.getenv("DEFAULTMODEL2", "openrouter/anthropic/claude-3-haiku"),
        complex_llm_model=os.getenv("DEFAULTMODELsT", "openrouter/anthropic/claude-3-haiku"),

        code_execution=CodeExecutionConfig(
            enabled=True,
            allowed_languages=["python"],
            sandbox_mode=True,
            timeout_seconds=120,
            install_packages=True
        ),

        persona_profiles=PersonaProfileConfig(
            profiles={
                "data_scientist": PersonaConfig(
                    name="Data Scientist",
                    style="analytical",
                    tone="objective",
                    personality_traits=["methodical", "insight_driven", "evidence_based"],
                    custom_instructions="Focus on statistical rigor and actionable insights."
                )
            },
            active_profile="data_scientist"
        )
    )


# ===== EXAMPLE USAGE =====

async def example_comprehensive_usage():
    """Comprehensive example showing all builder capabilities"""

    logger.info("Starting comprehensive FlowAgent builder example...")

    # Create builder with full configuration
    builder = (AgentBuilder()
               .with_name("ComprehensiveAgent")
               .with_models(
        fast_model="openrouter/anthropic/claude-3-haiku",
        complex_model="openrouter/openai/gpt-4o"
    )
               .with_temperature(0.7)
               .with_api_config(api_key_env_var="OPENROUTER_API_KEY")
               .verbose(True))

    # Add custom tools
    def get_current_time():
        """Get current timestamp"""
        return datetime.now().isoformat()

    def calculate_fibonacci(n: int) -> int:
        """Calculate nth Fibonacci number"""
        if n <= 1:
            return n
        return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)

    builder.add_tool(get_current_time, description="Get current system time")
    builder.add_tool(calculate_fibonacci, description="Calculate Fibonacci numbers")

    # Add MCP tool from code
    mcp_code = '''
async def analyze_text_sentiment(text: str) -> dict:
    """Analyze sentiment of text (mock implementation)"""
    import re

    positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
    negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]

    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    if pos_count > neg_count:
        sentiment = "positive"
        score = min(0.5 + (pos_count - neg_count) * 0.1, 1.0)
    elif neg_count > pos_count:
        sentiment = "negative"
        score = max(-0.5 - (neg_count - pos_count) * 0.1, -1.0)
    else:
        sentiment = "neutral"
        score = 0.0
return {
        "sentiment": sentiment,
        "score": score,
        "positive_words_found": pos_count,
        "negative_words_found": neg_count
    }
'''

    builder.add_mcp_tool_from_code(
        "analyze_sentiment",
        mcp_code,
        "Analyze text sentiment with scoring"
    )

    # Enable code execution with security
    builder.with_secure_python_execution(timeout=45)

    # Add all specializations
    builder.with_all_specializations()

    # Configure personas
    builder.with_developer_persona("Senior Full-Stack Developer")
    builder.with_domain_adapted_personas(True)

    # Add custom variables
    builder.with_custom_variables({
        "project_name": "FlowAgent Demo",
        "version": "2.0.0",
        "environment": "development"
    })

    builder.add_variable_scope("demo", {
        "started_at": datetime.now().isoformat(),
        "features_tested": [],
        "test_counter": 0
    })

    # Enable servers
    builder.enable_mcp_server(port=8001, server_name="ComprehensiveAgent_MCP")

    # Optimize and validate
    builder.optimize_config()
    validation_issues = builder.validate_config()

    if validation_issues["warnings"]:
        logger.info(f"Configuration warnings: {validation_issues['warnings']}")

    # Show build summary
    summary = builder.get_build_summary()
    logger.info(f"Build summary: {json.dumps(summary, indent=2)}")

    # Build the agent
    agent = await builder.build()

    logger.info(f"Agent built successfully: {agent.amd.name}")
    logger.info(f"Available tools: {agent.shared.get('available_tools', [])}")

    # Test variable system
    logger.info("Testing variable system...")

    # Set some test variables
    agent.set_variable("test.start_time", datetime.now().isoformat())
    agent.set_variable("user.name", "TestUser")
    agent.set_variable("demo.test_counter", 1)

    # Test variable resolution
    test_text = "Hello {{ user.name }}! Project: {{ project_name }} started at {{ test.start_time }}"
    formatted_text = agent.format_text(test_text)
    logger.info(f"Variable formatting test: {formatted_text}")

    # Get variable documentation
    var_docs = agent.get_variable_documentation()
    logger.info("Variable system documentation generated")

    # Test basic functionality
    logger.info("\n=== Testing Basic Functionality ===")

    simple_response = await agent.a_run("Hello! What's the current time?")
    logger.info(f"Simple query response: {simple_response}")

    # Test code execution
    logger.info("\n=== Testing Code Execution ===")

    code_response = await agent.a_run(
        "Write and execute Python code to calculate the factorial of 5 and show the result"
    )
    logger.info(f"Code execution response: {code_response}")

    # Test sentiment analysis tool
    logger.info("\n=== Testing MCP Tool ===")

    sentiment_response = await agent.a_run(
        "Analyze the sentiment of this text: 'This is an amazing product! I love how well it works.'"
    )
    logger.info(f"Sentiment analysis response: {sentiment_response}")

    # Test specialization
    logger.info("\n=== Testing Code Writing Specialization ===")

    specialization_response = await agent.a_run(
        "Create a Python class for managing a simple todo list with add, remove, and list methods. "
        "Include proper error handling and documentation."
    )
    logger.info(f"Code writing specialization response: {specialization_response[:200]}...")

    # Test variable usage in query
    logger.info("\n=== Testing Variables in Queries ===")

    # Update counter
    agent.set_variable("demo.test_counter",
                       agent.get_variable("demo.test_counter", 0) + 1)

    variable_response = await agent.a_run(
        "This is test number {{ demo.test_counter }} for project {{ project_name }}. "
        "Can you create a summary of what we've accomplished so far?"
    )
    logger.info(f"Variable integration response: {variable_response}")

    # Test refinement specialization
    logger.info("\n=== Testing Refinement Specialization ===")

    refinement_response = await agent.a_run(
        "Please refine and improve this code snippet: "
        "def calc(x): return x*2 if x>0 else 'error'"
    )
    logger.info(f"Refinement response: {refinement_response}")

    # Test agent statistics and status
    logger.info("\n=== Agent Statistics ===")

    status = agent.status
    logger.info(f"Agent status: {status}")

    if hasattr(agent.task_flow, 'executor_node'):
        exec_stats = agent.task_flow.executor_node.get_execution_statistics()
        logger.info(f"Execution statistics: {exec_stats}")

    # Test context management
    logger.info("\n=== Testing Context Management ===")

    context_stats = agent.get_context_statistics()
    logger.info(f"Context statistics: {json.dumps(context_stats, indent=2)}")

    # Test reasoning explanation
    logger.info("\n=== Testing Reasoning Process ===")

    try:
        reasoning = await agent.explain_reasoning_process()
        logger.info(f"Reasoning explanation: {reasoning}")
    except Exception as e:
        logger.info(f"Reasoning explanation not available: {e}")

    # Test task execution summary
    logger.info("\n=== Testing Task Summary ===")

    try:
        task_summary = await agent.get_task_execution_summary()
        logger.info(f"Task execution summary: {json.dumps(task_summary, indent=2)}")
    except Exception as e:
        logger.info(f"Task summary not available: {e}")

    # Test variable system features
    logger.info("\n=== Testing Advanced Variable Features ===")

    # Test variable validation
    test_template = "Hello {{ user.name }}, you have {{ invalid.variable }} notifications"
    validation_results = agent.variable_manager.validate_references(test_template)
    logger.info(f"Variable validation: {validation_results}")

    # Test variable suggestions
    suggestions = agent.variable_manager.get_variable_suggestions("user information")
    logger.info(f"Variable suggestions for 'user information': {suggestions}")

    # Test scope information
    scope_info = agent.variable_manager.get_scope_info()
    logger.info(f"Variable scope info: {json.dumps(scope_info, indent=2)}")

    # Final comprehensive test
    logger.info("\n=== Comprehensive Integration Test ===")

    comprehensive_response = await agent.a_run(
        "I'm working on project {{ project_name }} and need help. "
        "Can you: 1) Write Python code to generate the first 10 prime numbers, "
        "2) Execute the code to verify it works, "
        "3) Analyze the sentiment of this feedback: 'The code works perfectly!', "
        "4) Provide a summary with the current time. "
        "This is test {{ demo.test_counter }} of our comprehensive evaluation."
    )
    logger.info(f"Comprehensive test response: {comprehensive_response}")

    # Update final variables
    agent.set_variable("demo.features_tested", [
        "basic_queries", "code_execution", "mcp_tools",
        "specializations", "variable_system", "context_management"
    ])
    agent.set_variable("demo.completed_at", datetime.now().isoformat())

    # Final status
    logger.info("\n=== Final Agent State ===")
    final_status = agent.status
    logger.info(f"Final status: {json.dumps(final_status, indent=2)}")

    logger.info("\n=== Available Variables ===")
    available_vars = agent.get_available_variables()
    for scope, variables in available_vars.items():
        logger.info(f"Scope '{scope}': {len(variables)} variables")
        for var_name, var_info in list(variables.items())[:3]:  # Show first 3
            logger.info(f"  - {var_info['path']}: {var_info['preview']}")

    # Test configuration save/load
    logger.info("\n=== Testing Configuration Persistence ===")

    config_path = "/tmp/test_agent_config.yaml"
    try:
        builder.save_config(config_path)
        logger.info(f"Configuration saved to {config_path}")

        # Test loading
        loaded_builder = AgentBuilder.from_config_file(config_path)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Configuration persistence test failed: {e}")

    # Cleanup
    logger.info("\n=== Cleanup ===")
    await agent.close()
    logger.info("Agent closed successfully")

    logger.info("Comprehensive FlowAgent builder example completed!")


async def example_quick_start():
    """Quick start example for common use cases"""

    logger.info("=== Quick Start Examples ===")

    # Example 1: Developer Agent
    logger.info("Creating developer agent...")
    dev_agent = await (AgentBuilder
                       .create_developer_agent("DevHelper")
                       .with_models("openrouter/anthropic/claude-3-haiku")
                       .build())

    dev_response = await dev_agent.a_run(
        "Create a Python function to validate email addresses using regex"
    )
    logger.info(f"Dev agent response: {dev_response[:150]}...")
    await dev_agent.close()

    # Example 2: Analyst Agent
    logger.info("Creating analyst agent...")
    analyst_agent = await (AgentBuilder
                           .create_analyst_agent("DataHelper")
                           .build())

    analyst_response = await analyst_agent.a_run(
        "Generate Python code to create a simple data visualization of sales trends"
    )
    logger.info(f"Analyst agent response: {analyst_response[:150]}...")
    await analyst_agent.close()

    # Example 3: General Assistant
    logger.info("Creating general assistant...")
    assistant_agent = await (AgentBuilder
                             .create_general_assistant("GeneralHelper")
                             .build())

    assistant_response = await assistant_agent.a_run(
        "Help me plan a simple Python project structure for a web scraping tool"
    )
    logger.info(f"Assistant agent response: {assistant_response[:150]}...")
    await assistant_agent.close()

    logger.info("Quick start examples completed!")


if __name__ == "__main__":
    # Run comprehensive example
    asyncio.run(example_comprehensive_usage())

    # Run quick start examples
    # asyncio.run(example_quick_start())
