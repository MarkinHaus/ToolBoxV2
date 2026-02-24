import os
from typing import Optional
from minio import Minio


def create_minio_client(
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    secure: Optional[bool] = None,
) -> Minio:
    """
    Erstellt einen MinIO-Client für die Hauptapp.

    Prio: Explizite Argumente > Environment Variables > Defaults (lokaler Dev-Server).

    Env-Variablen:
        MINIO_ENDPOINT    z.B. "ryzen.local:9000"
        MINIO_ACCESS_KEY
        MINIO_SECRET_KEY
        MINIO_SECURE      "true" / "false"
    """
    endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
    access_key = access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")

    if secure is None:
        secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )
