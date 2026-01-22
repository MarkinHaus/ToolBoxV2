"""Custom exception classes for TB Registry."""


class RegistryException(Exception):
    """Base exception for all registry errors.

    Attributes:
        message: Human-readable error message.
        status_code: HTTP status code for API responses.
    """

    def __init__(self, message: str, status_code: int = 500) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            status_code: HTTP status code for API responses.
        """
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class PackageNotFoundError(RegistryException):
    """Raised when a requested package does not exist."""

    def __init__(self, package_name: str) -> None:
        """Initialize the exception.

        Args:
            package_name: Name of the package that was not found.
        """
        super().__init__(
            message=f"Package '{package_name}' not found",
            status_code=404,
        )
        self.package_name = package_name


class VersionNotFoundError(RegistryException):
    """Raised when a requested version does not exist."""

    def __init__(self, package_name: str, version: str) -> None:
        """Initialize the exception.

        Args:
            package_name: Name of the package.
            version: Version that was not found.
        """
        super().__init__(
            message=f"Version '{version}' of package '{package_name}' not found",
            status_code=404,
        )
        self.package_name = package_name
        self.version = version


class PublisherNotVerifiedError(RegistryException):
    """Raised when a publisher is not verified for the requested action."""

    def __init__(self, publisher_id: str) -> None:
        """Initialize the exception.

        Args:
            publisher_id: ID of the unverified publisher.
        """
        super().__init__(
            message=f"Publisher '{publisher_id}' is not verified",
            status_code=403,
        )
        self.publisher_id = publisher_id


class StorageError(RegistryException):
    """Raised when a storage operation fails."""

    def __init__(self, message: str, operation: str = "unknown") -> None:
        """Initialize the exception.

        Args:
            message: Error message describing the failure.
            operation: The storage operation that failed.
        """
        super().__init__(
            message=f"Storage error during '{operation}': {message}",
            status_code=500,
        )
        self.operation = operation


class DependencyResolutionError(RegistryException):
    """Raised when dependency resolution fails."""

    def __init__(self, message: str, conflicts: list[str] | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Error message describing the resolution failure.
            conflicts: List of conflicting dependencies.
        """
        super().__init__(
            message=f"Dependency resolution failed: {message}",
            status_code=400,
        )
        self.conflicts = conflicts or []


class AuthenticationError(RegistryException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication required") -> None:
        """Initialize the exception.

        Args:
            message: Error message describing the authentication failure.
        """
        super().__init__(message=message, status_code=401)


class PermissionDeniedError(RegistryException):
    """Raised when a user lacks permission for an action."""

    def __init__(self, message: str = "Permission denied") -> None:
        """Initialize the exception.

        Args:
            message: Error message describing the permission issue.
        """
        super().__init__(message=message, status_code=403)


class ValidationError(RegistryException):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Error message describing the validation failure.
            field: The field that failed validation.
        """
        super().__init__(message=message, status_code=400)
        self.field = field


class DuplicatePackageError(RegistryException):
    """Raised when attempting to create a package that already exists."""

    def __init__(self, package_name: str) -> None:
        """Initialize the exception.

        Args:
            package_name: Name of the duplicate package.
        """
        super().__init__(
            message=f"Package '{package_name}' already exists",
            status_code=409,
        )
        self.package_name = package_name


class DuplicateVersionError(RegistryException):
    """Raised when attempting to create a version that already exists."""

    def __init__(self, package_name: str, version: str) -> None:
        """Initialize the exception.

        Args:
            package_name: Name of the package.
            version: Version that already exists.
        """
        super().__init__(
            message=f"Version '{version}' of package '{package_name}' already exists",
            status_code=409,
        )
        self.package_name = package_name
        self.version = version

