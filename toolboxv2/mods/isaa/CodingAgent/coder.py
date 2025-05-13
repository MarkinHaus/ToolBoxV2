import asyncio
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from toolboxv2 import get_logger
from toolboxv2.mods.isaa.base.Agents import Agent
from toolboxv2.mods.isaa.extras.modes import get_free_agent

# Configure logging
logger = get_logger()


class TestCase(BaseModel):
    """Individual test case specification"""
    name: str = Field(..., description="Unique test case identifier")
    description: str = Field(..., description="What the test verifies")
    expected_result: str = Field(..., description="Expected outcome")
    test_code: str = Field(..., description="Actual test implementation")
    dependencies: list[str] = Field(default_factory=list, description="Required dependencies")


class CodeFile(BaseModel):
    """Source or test file specification"""
    path: str = Field(..., description="Relative file path")
    content: str = Field(..., description="File contents")
    language: Literal["python", "javascript", "html", "css", "c", "rust", "go"]
    is_test: bool = Field(default=False, description="Whether this is a test file")


class ProjectStructure(BaseModel):
    """Project layout specification"""
    root_dir: str
    directories_files: dict[str, list[str]] = Field(
        description="Map of directory types to paths"
    )
    test_framework: str = Field(
        default="pytest",
        description="Testing framework to use"
    )


class TestResult(BaseModel):
    """Individual test execution result"""
    passed: bool
    message: str
    details: dict[str, str]


@dataclass
class MVPPipeline:
    agent: Agent
    message: list

    async def generate_project(self, requirements: str) -> list[CodeFile]:
        """Generate complete project from requirements"""
        try:
            logger.info("Starting MVP project generation")

            # Generate project structure
            structure_dict = self.agent.format_class(
                ProjectStructure,
                f"""Create a project structure for these requirements:
                {requirements}

                Include:
                - Source files by logic structure
                - Test files by logic structure
                - Configuration files if needed
                - Documentation placement
                """
            )
            print(structure_dict)
            structure = ProjectStructure(**structure_dict)
            logger.info(f"Generated project structure with {len(structure.directories_files)} directories_files")
            # Generate test cases
            test_cases_dict = []
            for file_name in structure.directories_files.get('test',
                            structure.directories_files.get('tests',
                         structure.directories_files.get('test_files',
                         structure.directories_files.get('tests_files',
                         structure.directories_files.get('test_file',
                         structure.directories_files.get('tests_file', [])))))):
                test_cases_dict.append(self.agent.format_class(
                TestCase,
                f"""Generate test cases following TDD for:
                Requirements: {requirements}
                Only for the file : {file_name}
                Structure: {structure.model_dump_json()}

                Create comprehensive tests that:
                1. Cover all requirements
                2. Include edge cases
                3. Test cross-language integration
                4. Follow {structure.test_framework} best practices""", message=self.message
                ))
                self.message.append({'content': self.agent.last_result, 'role': 'assistant'})
            test_cases = [TestCase(**tc) for tc in test_cases_dict]
            logger.info(f"Generated {len(test_cases)} test cases")

            # Generate test files
            test_files_dict = []
            for tc in test_cases:
                test_files_dict.append(self.agent.format_class(
                CodeFile,
                f"""Create test files implementing these cases:
                {tc.model_dump_json()}

                Requirements:
                - Use {structure.test_framework}
                - One file per major feature
                - Include setup/teardown
                - Mock external services""", message=self.message
                ))
                self.message.append({'content': self.agent.last_result, 'role': 'assistant'})
            test_files = [CodeFile(**tf) for tf in test_files_dict]
            logger.info(f"Generated {len(test_files)} test files")

            # Generate implementation files
            impl_files_dict = []
            for tf in test_cases:
                impl_files_dict.append(self.agent.format_class(
                CodeFile,
                f"""Create implementation files to pass tests:
                Test Files: {tf}
                Structure: {structure.model_dump_json()}

                Requirements:
                - Follow language best practices
                - Include documentation
                - Handle errors properly
                - Use proper typing"""
                ))
            impl_files = [CodeFile(**impl) for impl in impl_files_dict]
            logger.info(f"Generated {len(impl_files)} implementation files")

            # Save files
            await self._save_files(structure, test_files + impl_files)

            # Run tests
            test_results = await run_tests(test_files, impl_files)

            if not all(tr.passed for tr in test_results):
                failed = [tr for tr in test_results if not tr.passed]
                raise Exception(f"Tests failed: {[f.message for f in failed]}")

            logger.info("Project generation completed successfully")
            return test_files + impl_files

        except Exception as e:
            logger.error(f"Project generation failed: {str(e)}")
            raise

    @staticmethod
    async def _save_files(structure: ProjectStructure,
                          files: list[CodeFile]) -> None:
        """Save generated files to disk"""
        try:
            # Create directories_files
            for _dir_type, paths in structure.directories_files.items():
                for path in paths:
                    full_path = os.path.join('data', structure.root_dir, path)
                    os.makedirs(full_path, exist_ok=True)
                    logger.debug(f"Created directory: {full_path}")

            # Save files
            for file in files:
                full_path = os.path.join('data',structure.root_dir, file.path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(file.content)
                logger.debug(f"Saved file: {full_path}")

        except OSError as e:
            logger.error(f"Failed to save files: {str(e)}")
            raise

async def run_tests(test_files: list[CodeFile], impl_files: list[CodeFile]) -> list[TestResult]:
    """Execute tests using pytest and return results"""
    try:
        if len(test_files + impl_files) == 0:
            return []
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save all files preserving their structure
            for file in test_files + impl_files:
                path = os.path.join(tmpdir, file.path)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w') as f:
                    f.write(file.content)

            # Run pytest with JSON report
            report_path = os.path.join(tmpdir, 'report.json')
            cmd = [
                sys.executable, '-m'
                'pytest',
                '--json-report',
                '--json-report-file', report_path,
                '-v',
                tmpdir
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode not in (0, 1):  # 1 is test failures, other codes are pytest errors
                raise Exception(f"Pytest execution failed: {stderr.decode()}")

            # Parse JSON report
            if not os.path.exists(report_path):
                raise Exception("Test report not generated")

            with open(report_path) as f:
                report = json.load(f)

            results = []
            for test in report.get('tests', []):
                passed = test['outcome'] == 'passed'
                error_data = test.get('call', {}).get('crash', {})

                result = TestResult(
                    passed=passed,
                    message=error_data.get('message', 'Test passed' if passed else 'Test failed'),
                    details={
                        'test_id': test['nodeid'],
                        'stdout': test.get('stdout', ''),
                        'stderr': test.get('stderr', ''),
                        'error': error_data.get('traceback', '') if not passed else ''
                    },
                    duration_ms=float(test['duration']) * 1000
                )
                results.append(result)

            return results

    except FileNotFoundError as e:
        raise FileNotFoundError(f"pytest not found. Please install pytest and pytest-json-report : {str(e)}")
    except Exception as e:
        raise Exception(f"Error running tests: {str(e)}")


async def main():
    try:
        agent = get_free_agent("demo", "anthropic/claude-3-haiku-20240307") # qwen-2.5-32b (6/10)
        pipeline = MVPPipeline(agent, [])

        requirements = """
Create a Snake Game with animation in real time with Python with:
use numba for efficient, jit,speedup
use pygame for User Interaction, UI, live iterations config
        """

        files = await pipeline.generate_project(requirements)
        print(f"Successfully generated {len(files)} files")

        # Print summary
        for file in files:
            print(f"{file.language}: {file.path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

