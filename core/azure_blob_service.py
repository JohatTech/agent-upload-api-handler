"""
Azure Blob Storage Service Adapter.
Handles blob uploads, downloads, container management, and pre-signed SAS generation.
Follows AI/ML Software Engineering Standard (Rule 44: Adapters & Rule 45: API Clients).
"""

from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
from urllib.parse import quote
from azure.storage.blob import (
    BlobServiceClient,
    ContainerClient,
    BlobSasPermissions,
    generate_blob_sas,
)

import config
from core.exceptions import StorageError, StorageAuthenticationError
from core.formatting import sanitize_collection_name
from core.schemas import (
    FilePresignRequestItem,
    FilePresignResponseItem,
    PresignUploadResponse,
)

logger = logging.getLogger("azure_blob_service")


class AzureBlobService:
    def __init__(
        self,
        connection_string: str | None = None,
        container_name: str | None = None
    ):
        self.connection_string = connection_string or config.AZURE_STORAGE_CONNECTION_STRING
        self.container_name = container_name or config.AZURE_STORAGE_CONTAINER_NAME or "licitaciones"

        if not self.connection_string:
            raise StorageAuthenticationError("AZURE_STORAGE_CONNECTION_STRING must be set in config / .env")

        try:
            self.service_client = BlobServiceClient.from_connection_string(self.connection_string)
            self.account_name = self.service_client.account_name
            self.account_key = getattr(self.service_client.credential, "account_key", None)
            self.container_client = self.service_client.get_container_client(self.container_name)
        except Exception as exc:
            logger.error("Failed to initialize Azure Blob Service Client: %s", exc)
            raise StorageAuthenticationError(f"Azure Blob initialization failed: {str(exc)}") from exc

    def _ensure_container_exists(self) -> None:
        """Create container if it does not exist."""
        try:
            if not self.container_client.exists():
                logger.info("Container '%s' not found. Creating...", self.container_name)
                self.container_client.create_container(public_access="blob")
        except Exception as exc:
            logger.warning("Could not auto-create container '%s': %s", self.container_name, exc)

    def get_blob_url(self, blob_name: str) -> str:
        """Returns the canonical public HTTPS URL for a blob in Azure Blob Storage."""
        quoted_blob_name = quote(blob_name, safe="/")
        return f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{quoted_blob_name}"

    def generate_upload_sas(self, blob_name: str, expiry_minutes: int = 60) -> str:
        """
        Generates a write-capable SAS token for a specific blob.
        Allows frontend to perform direct HTTP PUT uploads without routing binary data through web server.
        """
        if not self.account_key:
            raise StorageAuthenticationError("Account key is required to generate Azure Blob SAS tokens.")

        try:
            sas_token = generate_blob_sas(
                account_name=self.account_name,
                container_name=self.container_name,
                blob_name=blob_name,
                account_key=self.account_key,
                permission=BlobSasPermissions(read=True, write=True, create=True),
                expiry=datetime.now(timezone.utc) + timedelta(minutes=expiry_minutes),
            )
            return sas_token
        except Exception as exc:
            logger.error("Failed to generate SAS token for blob '%s': %s", blob_name, exc)
            raise StorageError(f"SAS token generation failed for '{blob_name}': {str(exc)}") from exc

    def generate_upload_sas_url(self, blob_name: str, expiry_minutes: int = 60) -> str:
        """Returns the full URL including SAS query parameters for direct client upload."""
        base_url = self.get_blob_url(blob_name)
        sas_token = self.generate_upload_sas(blob_name, expiry_minutes=expiry_minutes)
        return f"{base_url}?{sas_token}"

    def generate_project_presigned_urls(
        self,
        project_name: str,
        files: list[FilePresignRequestItem],
        expiry_minutes: int = 60
    ) -> PresignUploadResponse:
        """
        Generates a batch of SAS upload URLs for all files in a project.
        """
        self._ensure_container_exists()
        vector_store_id = sanitize_collection_name(project_name)
        response_items: list[FilePresignResponseItem] = []

        for item in files:
            # Sanitize file name to avoid path traversal
            safe_filename = Path(item.file_name).name
            blob_name = f"{project_name}/{safe_filename}"
            upload_sas_url = self.generate_upload_sas_url(blob_name, expiry_minutes=expiry_minutes)
            blob_url = self.get_blob_url(blob_name)

            response_items.append(
                FilePresignResponseItem(
                    file_name=item.file_name,
                    blob_name=blob_name,
                    upload_sas_url=upload_sas_url,
                    blob_url=blob_url,
                )
            )

        logger.info(
            "Generated %d SAS upload URLs for project '%s' (vector_store_id: %s)",
            len(response_items), project_name, vector_store_id
        )

        return PresignUploadResponse(
            project_name=project_name,
            vector_store_id=vector_store_id,
            files=response_items,
        )

    def upload_file(self, file_path: str | Path, blob_name: str) -> str:
        """
        Uploads a local file to Azure Blob Storage under `blob_name` (e.g. 'project_name/filename.pdf').
        Returns the public Azure Blob URL.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found: {file_path}")

        self._ensure_container_exists()

        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            logger.info("Uploading '%s' to Azure Blob Storage container '%s' as '%s'...", path.name, self.container_name, blob_name)

            with open(path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            blob_url = self.get_blob_url(blob_name)
            logger.info("Successfully uploaded '%s' to Azure Blob Storage: %s", path.name, blob_url)
            return blob_url
        except Exception as exc:
            logger.error("Failed to upload '%s' to Azure Blob Storage: %s", path.name, exc)
            raise StorageError(f"Azure upload failed for '{path.name}': {str(exc)}") from exc

    def list_blobs(self, name_starts_with: str | None = None) -> list[str]:
        """Lists blob names in the container with an optional prefix filter."""
        try:
            blobs = self.container_client.list_blobs(name_starts_with=name_starts_with)
            return [b.name for b in blobs]
        except Exception as exc:
            logger.error("Failed to list blobs: %s", exc)
            return []

    def delete_blob(self, blob_name: str) -> bool:
        """Deletes a blob from Azure Blob Storage."""
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            logger.info("Deleted blob '%s' from container '%s'", blob_name, self.container_name)
            return True
        except Exception as exc:
            logger.error("Failed to delete blob '%s': %s", blob_name, exc)
            return False

    def list_project_folders(self) -> dict[str, list[str]]:
        """
        Lists all blobs grouped by top-level project folder name.
        Returns a dict mapping project_name -> list of blob_names.
        """
        all_blobs = self.list_blobs()
        folders: dict[str, list[str]] = {}
        for blob_name in all_blobs:
            parts = blob_name.split("/", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                proj_name = parts[0].strip()
                folders.setdefault(proj_name, []).append(blob_name)
        return folders

    def download_blob(self, blob_name: str, dest_path: str | Path) -> bool:
        """Downloads a blob from Azure Blob Storage to a local file path."""
        try:
            dest = Path(dest_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob_client = self.container_client.get_blob_client(blob_name)
            with open(dest, "wb") as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())
            logger.info("Downloaded blob '%s' to '%s'", blob_name, dest)
            return True
        except Exception as exc:
            logger.error("Failed to download blob '%s': %s", blob_name, exc)
            raise StorageError(f"Failed to download blob '{blob_name}': {str(exc)}") from exc
