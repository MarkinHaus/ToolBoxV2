"""Configuration module using Pydantic Settings."""

from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    Attributes:
        server_host: Host address for the server.
        server_port: Port number for the server.
        database_url: SQLite database URL.
        minio_primary_endpoint: Primary MinIO server endpoint.
        minio_primary_access_key: Primary MinIO access key.
        minio_primary_secret_key: Primary MinIO secret key.
        minio_primary_bucket: Primary MinIO bucket name.
        minio_primary_secure: Use HTTPS for primary MinIO.
        minio_mirror_endpoint: Optional mirror MinIO endpoint.
        minio_mirror_access_key: Optional mirror MinIO access key.
        minio_mirror_secret_key: Optional mirror MinIO secret key.
        minio_mirror_bucket: Optional mirror MinIO bucket name.
        minio_mirror_secure: Use HTTPS for mirror MinIO.
        clerk_secret_key: Clerk authentication secret key.
        clerk_publishable_key: Clerk publishable key.
        cors_origins: Allowed CORS origins.
        debug: Enable debug mode.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Server Configuration
    host: str = Field(default="0.0.0.0", alias="server_host", description="Server host address")
    port: int = Field(default=4025, alias="server_port", description="Server port")

    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./data/registry.db",
        description="Database connection URL",
    )

    # Primary MinIO Configuration
    minio_primary_endpoint: str = Field(
        default="localhost:9000",
        description="Primary MinIO endpoint",
    )
    minio_primary_access_key: str = Field(
        default="minioadmin",
        description="Primary MinIO access key",
    )
    minio_primary_secret_key: str = Field(
        default="minioadmin",
        description="Primary MinIO secret key",
    )
    minio_primary_bucket: str = Field(
        default="tb-registry",
        description="Primary MinIO bucket name",
    )
    minio_primary_secure: bool = Field(
        default=False,
        description="Use HTTPS for primary MinIO",
    )

    # Mirror MinIO Configuration (Optional)
    minio_mirror_endpoint: Optional[str] = Field(
        default=None,
        description="Mirror MinIO endpoint",
    )
    minio_mirror_access_key: Optional[str] = Field(
        default=None,
        description="Mirror MinIO access key",
    )
    minio_mirror_secret_key: Optional[str] = Field(
        default=None,
        description="Mirror MinIO secret key",
    )
    minio_mirror_bucket: Optional[str] = Field(
        default=None,
        description="Mirror MinIO bucket name",
    )
    minio_mirror_secure: bool = Field(
        default=False,
        description="Use HTTPS for mirror MinIO",
    )

    # Clerk Authentication (DEPRECATED - will be removed after migration)
    clerk_secret_key: str = Field(
        default="",
        description="Clerk secret key for authentication (DEPRECATED)",
    )
    clerk_publishable_key: str = Field(
        default="",
        description="Clerk publishable key (DEPRECATED)",
    )

    # CloudM.Auth Configuration (NEW)
    cloudm_jwt_secret: str = Field(
        default="",
        description="Shared secret for CloudM.Auth JWT validation",
    )
    cloudm_auth_url: Optional[str] = Field(
        default=None,
        description="URL to CloudM.Auth service (fallback validation)",
    )

    # CORS Configuration
    cors_origins: list[str] = Field(
        default=["*"],
        description="List of allowed CORS origins",
    )

    # Debug Mode
    debug: bool = Field(default=False, description="Enable debug mode")

    @property
    def has_mirror(self) -> bool:
        """Check if mirror storage is configured."""
        return bool(
            self.minio_mirror_endpoint
            and self.minio_mirror_access_key
            and self.minio_mirror_secret_key
            and self.minio_mirror_bucket
        )

        # Füge diesen Validator am Ende der Klasse (vor den Properties) ein:

    @field_validator("minio_primary_endpoint", "minio_mirror_endpoint", mode="before")
    @classmethod
    def clean_minio_endpoint(cls, v: str | None) -> str | None:
        if not v or not isinstance(v, str):
            return v

        # Entferne Protokolle (http:// oder https://)
        clean = v.replace("http://", "").replace("https://", "")

        # Entferne alles nach einem / (Pfad) oder # (Fragment)
        clean = clean.split("/")[0].split("#")[0]

        # Entferne Leerzeichen (wichtig bei Kommentaren in .env)
        return clean.strip()

# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance.

    Returns:
        Settings instance.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

