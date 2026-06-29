"""
Unified interface for pushing document chunks into Supabase vector store.
"""

import logging
from langchain_core.documents import Document

import config
from core.embeddings import get_embeddings
from supabase_module import SupabaseModule
from core.utils import format_bytes, set_collection_name

logger = logging.getLogger("vectorstore_manager")


def _push_to_supabase(chunks: list[Document], project_name: str) -> None:
    """
    Push chunks to Supabase. Uses SupabaseModule.
    """
    collection = set_collection_name(project_name)
    content_size = sum(len(chunk.page_content.encode('utf-8')) for chunk in chunks)

    logger.info(
        "Supabase  │  collection='%s'  │  chunks=%d  │  size=%s",
        collection,
        len(chunks),
        format_bytes(content_size) if chunks else "0 B"
    )

    embeddings = get_embeddings()
    supabase_module = SupabaseModule()
    supabase_module.create_collection(collection_name=collection)
    supabase_module.upsert_documents(collection_name=collection, chunks=chunks, embeddings=embeddings)


_TARGET_REGISTRY: dict[str, callable] = {
    "supabase": _push_to_supabase,
}


def push_to_all_targets(chunks: list[Document], project_name: str) -> None:
    """
    Push chunks to every vector store listed in config.VECTORSTORE_TARGETS.
    """
    if not chunks:
        logger.warning("No chunks to push – skipping vectorstore writes.")
        return

    for target_name in config.VECTORSTORE_TARGETS:
        push_fn = _TARGET_REGISTRY.get(target_name)
        if push_fn is None:
            logger.error("Unknown vectorstore target '%s' – skipping.", target_name)
            continue

        try:
            push_fn(chunks, project_name)
        except Exception as exc:
            logger.error(
                "Failed to push to '%s' for project '%s': %s",
                target_name,
                project_name,
                exc,
                exc_info=True,
            )
