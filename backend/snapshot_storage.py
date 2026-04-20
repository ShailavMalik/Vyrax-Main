"""Azure Blob snapshot storage helpers.

This module provides:
- cleanupOldBlobs(containerClient): frees 1GB when total usage exceeds 4GB.
- async cleanup scheduling so uploads are not blocked.
- lightweight SQLite metadata sync for blob lifecycle operations.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
    from azure.storage.blob import BlobServiceClient, ContainerClient
except Exception:  # pragma: no cover - handled at runtime if SDK missing
    BlobServiceClient = None
    ContainerClient = Any
    ResourceExistsError = Exception
    ResourceNotFoundError = Exception


DEFAULT_STORAGE_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_FREE_TARGET_BYTES = 1 * 1024 * 1024 * 1024


def _ensure_azure_sdk() -> None:
    if BlobServiceClient is None:
        raise RuntimeError(
            "azure-storage-blob is not installed. "
            "Install backend dependencies with: pip install -r backend/requirements.txt"
        )


@dataclass
class SnapshotRepository:
    db_path: Path

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS snapshots (
                        blob_name TEXT PRIMARY KEY,
                        blob_url TEXT NOT NULL,
                        content_length INTEGER NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.commit()

    def upsert_snapshot(self, blob_name: str, blob_url: str, content_length: int) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO snapshots (blob_name, blob_url, content_length, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(blob_name)
                    DO UPDATE SET
                        blob_url = excluded.blob_url,
                        content_length = excluded.content_length,
                        created_at = excluded.created_at
                    """,
                    (blob_name, blob_url, int(content_length), created_at),
                )
                conn.commit()

    def delete_snapshot_by_blob_name(self, blob_name: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM snapshots WHERE blob_name = ?", (blob_name,))
                conn.commit()


def build_container_client(connection_string: str, container_name: str) -> ContainerClient:
    _ensure_azure_sdk()
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container_name)
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass
    return container_client


def cleanupOldBlobs(
    containerClient: ContainerClient,
    snapshot_repo: Optional[SnapshotRepository] = None,
    storage_limit_bytes: int = DEFAULT_STORAGE_LIMIT_BYTES,
    free_target_bytes: int = DEFAULT_FREE_TARGET_BYTES,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, int]:
    """Delete only oldest blobs until at least 1GB is freed when usage exceeds 4GB."""
    log = logger or logging.getLogger(__name__)

    blob_items = []
    total_size = 0
    for blob in containerClient.list_blobs():
        content_length = int(
            getattr(getattr(blob, "properties", None), "content_length", None)
            or getattr(blob, "size", 0)
            or 0
        )
        last_modified = getattr(blob, "last_modified", None)
        if last_modified is None:
            last_modified = datetime.min.replace(tzinfo=timezone.utc)

        blob_items.append(
            {
                "name": blob.name,
                "content_length": content_length,
                "last_modified": last_modified,
            }
        )
        total_size += content_length

    if total_size <= storage_limit_bytes:
        return {
            "total_size": total_size,
            "deleted_size": 0,
            "deleted_count": 0,
            "cleanup_triggered": 0,
        }

    blob_items.sort(key=lambda item: item["last_modified"])
    deleted_size = 0
    deleted_count = 0

    for blob in blob_items:
        blob_name = str(blob["name"])
        blob_size = int(blob["content_length"])

        try:
            containerClient.delete_blob(blob_name)
            deleted_size += blob_size
            deleted_count += 1

            if snapshot_repo is not None:
                try:
                    snapshot_repo.delete_snapshot_by_blob_name(blob_name)
                except Exception:
                    log.exception("Failed deleting snapshot DB record for blob=%s", blob_name)
        except ResourceNotFoundError:
            log.warning("Blob already missing during cleanup: %s", blob_name)
        except Exception:
            log.exception("Failed deleting blob during cleanup: %s", blob_name)

        if deleted_size >= free_target_bytes:
            break

    return {
        "total_size": total_size,
        "deleted_size": deleted_size,
        "deleted_count": deleted_count,
        "cleanup_triggered": 1,
    }


class AsyncCleanupScheduler:
    """Runs cleanup in a background thread and supports every-N-upload cadence."""

    def __init__(self, run_every_uploads: int = 1) -> None:
        self._run_every_uploads = max(1, int(run_every_uploads))
        self._upload_counter = 0
        self._lock = threading.Lock()
        self._running = False

    def schedule(
        self,
        container_client: ContainerClient,
        snapshot_repo: Optional[SnapshotRepository] = None,
        logger: Optional[logging.Logger] = None,
        force: bool = False,
    ) -> bool:
        log = logger or logging.getLogger(__name__)

        with self._lock:
            self._upload_counter += 1
            should_run = force or (self._upload_counter % self._run_every_uploads == 0)
            if self._running or not should_run:
                return False
            self._running = True

        def _runner() -> None:
            try:
                cleanupOldBlobs(
                    containerClient=container_client,
                    snapshot_repo=snapshot_repo,
                    logger=log,
                )
            except Exception:
                log.exception("Async blob cleanup failed")
            finally:
                with self._lock:
                    self._running = False

        thread = threading.Thread(target=_runner, name="azure-blob-cleanup", daemon=True)
        thread.start()
        return True
