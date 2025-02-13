import shutil
from datetime import datetime

from .coder import *


class ChangeImpact(BaseModel):
    """Analysis of change impact on existing codebase"""
    affected_files: List[str] = Field(..., description="Files that need modification")
    new_files: List[str] = Field(default_factory=list, description="New files to be created")
    risk_level: Literal["low", "medium", "high"]
    required_changes: Dict[str, List[str]] = Field(
        ...,
        description="Map of files to required changes"
    )
    dependencies: Dict[str, str] = Field(
        default_factory=dict,
        description="New or updated dependencies"
    )
    test_impact: Dict[str, List[str]] = Field(
        ...,
        description="Map of test files to affected test cases"
    )


class ChangeSet(BaseModel):
    """Collection of file changes"""
    modified_files: List[CodeFile]
    new_files: List[CodeFile]
    deleted_files: List[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class IncrementalPipeline:
    agent: Agent
    backup_dir: str = "backups"

    async def apply_change(self,
                           change_request: str,
                           existing_files: List[CodeFile],
                           dry_run: bool = False) -> ChangeSet:
        """Apply changes to existing codebase following TDD"""
        try:
            logger.info(f"Starting incremental change: {change_request[:100]}...")

            # Analyze impact
            impact = await self._analyze_change_impact(change_request, existing_files)
            logger.info(f"Change impact analysis: {impact.risk_level} risk, "
                        f"{len(impact.affected_files)} files affected")

            if impact.risk_level == "high" and not dry_run:
                await self._create_backup(existing_files)
                logger.info("Created backup due to high-risk change")

            # Update tests first (TDD)
            updated_tests = await self._update_test_files(
                change_request, existing_files, impact
            )
            logger.info(f"Updated {len(updated_tests)} test files")

            # Validate updated tests compile
            await self._validate_python_syntax(updated_tests)

            if dry_run:
                logger.info("Dry run completed successfully")
                return ChangeSet(
                    modified_files=updated_tests,
                    new_files=[],
                    timestamp=datetime.now().isoformat()
                )

            # Update implementation
            updated_code = await self._update_code_files(
                change_request, existing_files, updated_tests, impact
            )
            logger.info(f"Updated {len(updated_code)} implementation files")

            # Create changeset
            changeset = ChangeSet(
                modified_files=updated_tests + updated_code,
                new_files=[f for f in updated_code if f.path not in
                           [ef.path for ef in existing_files]],
                deleted_files=[]
            )

            # Run tests
            test_results = await run_tests(updated_tests, updated_code)

            if not all(tr.passed for tr in test_results):
                failed = [tr for tr in test_results if not tr.passed]
                if impact.risk_level == "high":
                    await self._restore_backup()
                    logger.error("Tests failed, restored from backup")
                raise Exception(f"Tests failed: {[f.message for f in failed]}")

            logger.info("Change applied successfully")
            return changeset

        except Exception as e:
            logger.error(f"Failed to apply change: {str(e)}")
            raise

    async def _analyze_change_impact(self,
                                     change_request: str,
                                     existing_files: List[CodeFile]) -> ChangeImpact:
        """Analyze impact of requested changes"""
        impact_dict = self.agent.format_class(
            ChangeImpact,
            f"""Analyze the impact of this change request:
            {change_request}

            On existing files:
            {[f.model_dump_json() for f in existing_files]}

            Consider:
            1. Which files need modification
            2. Risk level based on:
               - Number of affected files
               - Complexity of changes
               - Cross-language dependencies
            3. Required changes in each file
            4. Impact on tests
            5. New dependencies needed"""
        )
        return ChangeImpact(**impact_dict)

    async def _update_test_files(self,
                                 change_request: str,
                                 existing_files: List[CodeFile],
                                 impact: ChangeImpact) -> List[CodeFile]:
        """Update test files following TDD"""
        test_updates_dict = self.agent.format_class(
            CodeFile,
            f"""Update or create test files for this change:
            Change Request: {change_request}

            Impact Analysis: {impact.model_dump_json()}

            Existing Tests:
            {[f.model_dump_json() for f in existing_files if f.is_test]}

            Requirements:
            1. Update affected tests
            2. Add new test cases
            3. Maintain existing test coverage
            4. Use pytest features appropriately
            5. Include docstrings and comments"""
        )

        updated_tests = [CodeFile(**tf) for tf in test_updates_dict]
        return updated_tests

    async def _update_code_files(self,
                                 change_request: str,
                                 existing_files: List[CodeFile],
                                 updated_tests: List[CodeFile],
                                 impact: ChangeImpact) -> List[CodeFile]:
        """Update implementation files to pass tests"""
        code_updates_dict = self.agent.format_class(
            CodeFile,
            f"""Update implementation files to pass tests:
            Change Request: {change_request}

            Impact Analysis: {impact.model_dump_json()}

            Updated Tests: {[t.model_dump_json() for t in updated_tests]}

            Requirements:
            1. Implement changes to pass tests
            2. Maintain existing functionality
            3. Follow language-specific best practices
            4. Update documentation
            5. Handle errors appropriately"""
        )

        updated_code = [CodeFile(**cf) for cf in code_updates_dict]
        return updated_code

    async def _validate_python_syntax(self, files: List[CodeFile]) -> None:
        """Validate Python files compile"""
        for file in files:
            if file.language == "python":
                try:
                    compile(file.content, file.path, 'exec')
                except SyntaxError as e:
                    raise Exception(f"Syntax error in {file.path}: {str(e)}")

    async def _create_backup(self, files: List[CodeFile]) -> None:
        """Create backup of existing files"""
        backup_path = os.path.join(
            self.backup_dir,
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(backup_path, exist_ok=True)

        for file in files:
            file_path = os.path.join(backup_path, file.path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(file.content)

    async def _restore_backup(self) -> None:
        """Restore from most recent backup"""
        backups = sorted(os.listdir(self.backup_dir))
        if not backups:
            raise Exception("No backup available")

        latest_backup = os.path.join(self.backup_dir, backups[-1])
        for root, _, files in os.walk(latest_backup):
            for file in files:
                src = os.path.join(root, file)
                dst = os.path.join(
                    os.path.relpath(root, latest_backup),
                    file
                )
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)


async def main():
    try:
        agent = Agent()
        pipeline = IncrementalPipeline(agent)

        # Example existing files
        existing_files = [
            CodeFile(
                path="src/api/auth.py",
                content="...",
                language="python",
                is_test=False
            ),
            CodeFile(
                path="tests/test_auth.py",
                content="...",
                language="python",
                is_test=True
            )
        ]

        # Example change request
        change_request = """
        Add rate limiting to authentication endpoints:
        - Maximum 5 attempts per minute
        - Lockout after 3 failed attempts
        - Email notification on lockout
        """

        # Apply changes
        changeset = await pipeline.apply_change(
            change_request,
            existing_files,
            dry_run=False
        )

        print("Changes applied successfully:")
        print(f"Modified files: {len(changeset.modified_files)}")
        print(f"New files: {len(changeset.new_files)}")
        print(f"Deleted files: {len(changeset.deleted_files)}")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
