import logging
import os
from pathlib import Path
from urllib.parse import quote
from azure.storage.blob import BlobServiceClient, ContainerClient
import config

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
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING must be set in config / .env")

        self.service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.account_name = self.service_client.account_name
        self.container_client = self.service_client.get_container_client(self.container_name)

    def _ensure_container_exists(self) -> None:
        """Create container if it does not exist."""
        try:
            if not self.container_client.exists():
                logger.info("Container '%s' not found. Creating...", self.container_name)
                self.container_client.create_container(public_access="blob")
        except Exception as exc:
            logger.warning("Could not auto-create container '%s': %s", self.container_name, exc)

    def get_blob_url(self, blob_name: str) -> str:
        """Returns the public HTTPS URL for a blob in Azure Blob Storage."""
        quoted_blob_name = quote(blob_name, safe="/")
        return f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{quoted_blob_name}"

    def upload_file(self, file_path: str | Path, blob_name: str) -> str:
        """
        Uploads a local file to Azure Blob Storage under `blob_name` (e.g. 'project_name/filename.pdf').
        Returns the public Azure Blob URL.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found: {file_path}")

        self._ensure_container_exists()

        blob_client = self.container_client.get_blob_client(blob_name)
        logger.info("Uploading '%s' to Azure Blob Storage container '%s' as '%s'...", path.name, self.container_name, blob_name)

        with open(path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        blob_url = self.get_blob_url(blob_name)
        logger.info("Successfully uploaded '%s' to Azure Blob Storage: %s", path.name, blob_url)
        return blob_url

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
