#!/usr/bin/env python3
"""
Migration Script: Supabase Storage -> Azure Blob Storage

This script:
1. Scans all files in Supabase Storage bucket ('project_files').
2. Downloads each file binary.
3. Uploads the file binary to Azure Blob Storage under 'licitaciones' container.
4. Updates the 'file_url' in Supabase DB 'notebook_files' table.
5. Does NOT delete files from Supabase Storage unless explicitly specified via --purge.

Usage:
    python scripts/migrate_supabase_to_azure_blob.py [--dry-run] [--purge]
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from supabase_module.supabase_client import SupabaseModule
from core.azure_blob_service import AzureBlobService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  │  %(name)-22s  │  %(levelname)-7s  │  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("migration.supabase_to_azure")


def run_migration(dry_run: bool = False, purge: bool = False):
    logger.info("=" * 70)
    logger.info("STARTING MIGRATION: Supabase Storage  →  Azure Blob Storage")
    if dry_run:
        logger.info("MODE: DRY RUN (no files will be uploaded, updated, or deleted)")
    if purge:
        logger.info("PURGE ENABLED: Original files will be deleted from Supabase Storage after upload")
    else:
        logger.info("PURGE DISABLED: Original files will remain in Supabase Storage (Safe Copy Mode)")
    logger.info("=" * 70)

    supabase_module = SupabaseModule()
    azure_service = AzureBlobService()
    bucket_name = "project_files"

    # Step 1: List all files in Supabase Storage bucket 'project_files'
    try:
        files_in_bucket = supabase_module.client.storage.from_(bucket_name).list()
    except Exception as exc:
        logger.error("Failed to list files in Supabase Storage bucket '%s': %s", bucket_name, exc)
        return

    if not files_in_bucket:
        logger.info("No files found in Supabase Storage bucket '%s'. Migration complete.", bucket_name)
        return

    logger.info("Found %d top-level items in Supabase bucket '%s'", len(files_in_bucket), bucket_name)

    migrated_count = 0
    updated_db_count = 0
    error_count = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Recursively discover items in bucket (folders / files)
        items_to_process = []

        def scan_folder(prefix=""):
            try:
                res = supabase_module.client.storage.from_(bucket_name).list(prefix)
                for item in res:
                    item_name = item.get("name")
                    if not item_name:
                        continue
                    full_path = f"{prefix}/{item_name}" if prefix else item_name
                    # If metadata is empty or contains id null, it might be a folder
                    if item.get("id") is None and item.get("metadata") is None:
                        scan_folder(full_path)
                    else:
                        items_to_process.append(full_path)
            except Exception as e:
                logger.error("Failed to scan folder '%s': %s", prefix, e)

        scan_folder()

        logger.info("Discovered %d file objects in Supabase bucket '%s'", len(items_to_process), bucket_name)

        for object_path in items_to_process:
            logger.info("Processing object: %s", object_path)
            if dry_run:
                azure_url = azure_service.get_blob_url(object_path)
                logger.info("[DRY RUN] Would download '%s' from Supabase, upload to Azure at '%s', and update DB", object_path, azure_url)
                migrated_count += 1
                continue

            try:
                # 1. Download file from Supabase Storage
                file_bytes = supabase_module.client.storage.from_(bucket_name).download(object_path)
                local_file = temp_dir_path / Path(object_path).name
                with open(local_file, "wb") as f:
                    f.write(file_bytes)

                # 2. Upload to Azure Blob Storage
                azure_url = azure_service.upload_file(local_file, object_path)
                migrated_count += 1
                logger.info("Uploaded to Azure Blob Storage: %s", azure_url)

                # 3. Update file_url in Supabase DB notebook_files table
                file_name = Path(object_path).name
                try:
                    # Search notebook_files matching file_name
                    db_res = supabase_module.client.table("notebook_files").select("id, file_name, file_url").eq("file_name", file_name).execute()
                    if db_res.data:
                        for row in db_res.data:
                            supabase_module.client.table("notebook_files").update({"file_url": azure_url}).eq("id", row["id"]).execute()
                            updated_db_count += 1
                            logger.info("Updated DB row id=%s file_name='%s' with new file_url=%s", row["id"], file_name, azure_url)
                    else:
                        logger.warning("No matching notebook_files row found in DB for file_name='%s'", file_name)
                except Exception as db_exc:
                    logger.error("Failed to update DB for file_name='%s': %s", file_name, db_exc)

                # 4. Optional Purge from Supabase Storage
                if purge:
                    try:
                        supabase_module.client.storage.from_(bucket_name).remove([object_path])
                        logger.info("Purged object '%s' from Supabase Storage bucket '%s'", object_path, bucket_name)
                    except Exception as purge_exc:
                        logger.error("Failed to purge '%s' from Supabase Storage: %s", object_path, purge_exc)

            except Exception as item_exc:
                logger.error("Error migrating object '%s': %s", object_path, item_exc)
                error_count += 1

    logger.info("=" * 70)
    logger.info("MIGRATION SUMMARY")
    logger.info("Total Files Processed: %d", len(items_to_process))
    logger.info("Migrated to Azure Blob: %d", migrated_count)
    logger.info("DB Records Updated:     %d", updated_db_count)
    logger.info("Errors:                 %d", error_count)
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate files from Supabase Storage to Azure Blob Storage.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate migration without modifying files or DB.")
    parser.add_argument("--purge", action="store_true", help="Delete files from Supabase Storage after successful upload.")

    args = parser.parse_args()
    run_migration(dry_run=args.dry_run, purge=args.purge)
