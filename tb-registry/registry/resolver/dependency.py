"""Dependency resolution for packages."""

import logging
from dataclasses import dataclass, field
from typing import Optional

from packaging.version import Version

from registry.db.repositories.package_repo import PackageRepository
from registry.resolver.semver import find_best_match, parse_specifier, parse_version

logger = logging.getLogger(__name__)


@dataclass
class Requirement:
    """Package requirement specification.

    Attributes:
        name: Package name.
        specifier: Version specifier string.
        optional: Whether this requirement is optional.
        features: Required features.
    """

    name: str
    specifier: str
    optional: bool = False
    features: list[str] = field(default_factory=list)


@dataclass
class ResolvedPackage:
    """Resolved package with version and download info.

    Attributes:
        name: Package name.
        version: Resolved version string.
        dependencies: List of resolved dependencies.
        download_url: URL to download the package.
        checksum: SHA256 checksum.
    """

    name: str
    version: str
    dependencies: list["ResolvedPackage"] = field(default_factory=list)
    download_url: Optional[str] = None
    checksum: Optional[str] = None


@dataclass
class ResolutionResult:
    """Result of dependency resolution.

    Attributes:
        success: Whether resolution succeeded.
        resolved: Dict of resolved packages by name.
        conflicts: List of conflict descriptions.
        warnings: List of warning messages.
    """

    success: bool
    resolved: dict[str, ResolvedPackage] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DependencyResolver:
    """Resolver for package dependencies.

    Uses a greedy algorithm for MVP, can be upgraded to PubGrub later.

    Attributes:
        package_repo: Package repository for fetching package info.
    """

    def __init__(self, package_repo: PackageRepository) -> None:
        """Initialize the resolver.

        Args:
            package_repo: Package repository instance.
        """
        self.package_repo = package_repo

    async def resolve(
        self,
        requirements: list[Requirement],
        toolbox_version: Optional[str] = None,
    ) -> ResolutionResult:
        """Resolve dependencies for a list of requirements.

        Args:
            requirements: List of requirements to resolve.
            toolbox_version: Optional ToolBox version constraint.

        Returns:
            ResolutionResult with resolved packages or conflicts.
        """
        result = ResolutionResult(success=True)
        visited: set[str] = set()

        for req in requirements:
            if req.optional:
                continue

            try:
                await self._resolve_requirement(
                    req,
                    result,
                    visited,
                    toolbox_version,
                )
            except Exception as e:
                result.success = False
                result.conflicts.append(f"Failed to resolve {req.name}: {e}")
                logger.error(f"Resolution failed for {req.name}: {e}")

        return result

    async def _resolve_requirement(
        self,
        req: Requirement,
        result: ResolutionResult,
        visited: set[str],
        toolbox_version: Optional[str],
    ) -> Optional[ResolvedPackage]:
        """Resolve a single requirement recursively.

        Args:
            req: Requirement to resolve.
            result: Result object to update.
            visited: Set of visited package names.
            toolbox_version: Optional ToolBox version constraint.

        Returns:
            ResolvedPackage or None if already resolved.
        """
        # Check if already resolved
        if req.name in result.resolved:
            existing = result.resolved[req.name]
            # Check compatibility
            spec = parse_specifier(req.specifier)
            if parse_version(existing.version) not in spec:
                result.conflicts.append(
                    f"Conflict: {req.name} requires {req.specifier} "
                    f"but {existing.version} is already resolved"
                )
                result.success = False
            return None

        # Prevent cycles
        if req.name in visited:
            result.warnings.append(f"Circular dependency detected: {req.name}")
            return None

        visited.add(req.name)

        # Fetch package
        package = await self.package_repo.get_by_name(req.name)
        if not package:
            result.conflicts.append(f"Package not found: {req.name}")
            result.success = False
            return None

        # Get available versions
        versions = await self.package_repo.get_versions(req.name)
        if not versions:
            result.conflicts.append(f"No versions available for: {req.name}")
            result.success = False
            return None

        # Parse versions and find best match
        version_objs = [parse_version(v.version) for v in versions if not v.yanked]
        spec = parse_specifier(req.specifier)
        best_version = find_best_match(version_objs, spec)

        if not best_version:
            result.conflicts.append(
                f"No version of {req.name} matches {req.specifier}"
            )
            result.success = False
            return None

        # Get the version details
        version_str = str(best_version)
        version_data = await self.package_repo.get_version(req.name, version_str)

        if not version_data:
            result.conflicts.append(f"Version {version_str} not found for {req.name}")
            result.success = False
            return None

        # Check ToolBox version compatibility
        if toolbox_version and version_data.toolbox_version:
            tb_spec = parse_specifier(version_data.toolbox_version)
            if parse_version(toolbox_version) not in tb_spec:
                result.warnings.append(
                    f"{req.name}@{version_str} requires ToolBox {version_data.toolbox_version}, "
                    f"but {toolbox_version} is being used"
                )

        # Get download info
        download_url = None
        checksum = None
        if version_data.storage_locations:
            loc = version_data.storage_locations[0]
            checksum = loc.checksum_sha256

        # Create resolved package
        resolved = ResolvedPackage(
            name=req.name,
            version=version_str,
            download_url=download_url,
            checksum=checksum,
        )

        # Resolve dependencies recursively
        for dep in version_data.dependencies:
            dep_req = Requirement(
                name=dep.name,
                specifier=dep.version_spec,
                optional=dep.optional,
                features=dep.features,
            )

            if not dep_req.optional:
                dep_resolved = await self._resolve_requirement(
                    dep_req,
                    result,
                    visited,
                    toolbox_version,
                )
                if dep_resolved:
                    resolved.dependencies.append(dep_resolved)

        result.resolved[req.name] = resolved
        return resolved

    @staticmethod
    def parse_requirement_string(req_str: str) -> Requirement:
        """Parse a requirement string into a Requirement object.

        Supports formats:
        - "package_name"
        - "package_name>=1.0.0"
        - "package_name^1.0.0"
        - "package_name~1.0.0"

        Args:
            req_str: Requirement string.

        Returns:
            Parsed Requirement object.
        """
        import re

        # Match package name and optional version specifier
        match = re.match(r"^([a-z0-9_-]+)(.*)$", req_str.strip(), re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid requirement string: {req_str}")

        name = match.group(1)
        specifier = match.group(2).strip() or "*"

        return Requirement(name=name, specifier=specifier)

