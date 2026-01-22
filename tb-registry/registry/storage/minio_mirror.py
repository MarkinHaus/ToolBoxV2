"""MinIO mirror storage backend."""

from registry.storage.minio_primary import MinioPrimaryBackend


class MinioMirrorBackend(MinioPrimaryBackend):
    """MinIO mirror storage backend.

    Inherits from MinioPrimaryBackend with a different name.
    Can be extended for mirror-specific logic.

    Attributes:
        name: Backend name identifier.
    """

    name: str = "minio-mirror"

