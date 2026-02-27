"""Diff routes for package incremental updates."""

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse

from registry.api.deps import get_diff_generator, get_package_repo
from registry.diff import DiffGenerator, DiffInfo
from registry.db.repositories.package_repo import PackageRepository

router = APIRouter()


@router.get("/packages/{name}/diff/{from_version}/{to_version}")
async def get_package_diff(
    name: str,
    from_version: str,
    to_version: str,
    diff_generator: DiffGenerator = Depends(get_diff_generator),
) -> DiffInfo:
    """Get diff information between two package versions.

    Args:
        name: Package name
        from_version: Source version
        to_version: Target version
        diff_generator: Diff generator dependency

    Returns:
        DiffInfo with diff metadata

    Raises:
        HTTPException: If diff cannot be created
    """
    try:
        # Try to get existing diff
        diff_info = await diff_generator.get_diff_info(name, from_version, to_version)

        if diff_info:
            return diff_info

        # Create new diff
        diff_info = await diff_generator.create_diff(name, from_version, to_version)
        return diff_info

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create diff: {e}")


@router.get("/packages/{name}/diff/{from_version}/{to_version}/download")
async def download_diff(
    name: str,
    from_version: str,
    to_version: str,
    diff_generator: DiffGenerator = Depends(get_diff_generator),
    package_repo: PackageRepository = Depends(get_package_repo),
):
    """Download patch file for applying diff.

    Args:
        name: Package name
        from_version: Source version
        to_version: Target version
        diff_generator: Diff generator dependency
        package_repo: Package repository dependency

    Returns:
        FileResponse with patch data

    Raises:
        HTTPException: If diff not found or download fails
    """
    import tempfile
    from pathlib import Path

    try:
        # Get diff info
        diff_info = await diff_generator.get_diff_info(name, from_version, to_version)

        if not diff_info or not diff_info.patch_storage_path:
            # Try to create it
            diff_info = await diff_generator.create_diff(name, from_version, to_version)

        if not diff_info.patch_storage_path:
            raise HTTPException(status_code=404, detail="Patch file not available")

        # Get patch file path from storage
        # For now, return a placeholder response
        # In production, this would stream from MinIO/S3

        return Response(
            content={
                "message": "Patch download",
                "patch_path": diff_info.patch_storage_path,
                "checksum": diff_info.patch_checksum,
                "size": diff_info.patch_size,
            },
            media_type="application/json",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download patch: {e}")


@router.post("/packages/{name}/diff/create")
async def create_diff_endpoint(
    name: str,
    from_version: str,
    to_version: str,
    force: bool = False,
    diff_generator: DiffGenerator = Depends(get_diff_generator),
) -> DiffInfo:
    """Manually trigger diff creation between two versions.

    This is useful for pre-generating diffs for common update paths.

    Args:
        name: Package name
        from_version: Source version
        to_version: Target version
        force: Force recreation even if diff exists
        diff_generator: Diff generator dependency

    Returns:
        DiffInfo with diff metadata
    """
    try:
        diff_info = await diff_generator.create_diff(name, from_version, to_version, force=force)
        return diff_info

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create diff: {e}")
