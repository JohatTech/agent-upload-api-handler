import logging
from typing import Any, List, Optional, Dict, Tuple
from supabase import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore
import config

logger = logging.getLogger("supabase.client")


class CustomSupabaseVectorStore(SupabaseVectorStore):
    def similarity_search_by_vector_with_relevance_scores(
        self,
        query: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
        postgrest_filter: Optional[str] = None,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Any, float]]:
        """Override to support newer versions of postgrest-py where .params is moved to .request.params."""
        import warnings
        from langchain_core.documents import Document

        # Convert MongoDB-style filter to PostgreSQL syntax if needed
        if filter:
            for key, value in filter.items():
                if isinstance(value, dict) and "$in" in value:
                    # Extract the list of values for the $in operator
                    in_values = value["$in"]
                    # Create a PostgreSQL IN clause
                    values_str = ",".join(f"'{str(v)}'" for v in in_values)
                    new_filter = f"metadata->>{key} IN ({values_str})"

                    # Combine with existing postgrest_filter if present
                    if postgrest_filter:
                        postgrest_filter = f"({postgrest_filter}) and ({new_filter})"
                    else:
                        postgrest_filter = new_filter

        match_documents_params = self.match_args(query, filter)
        query_builder = self._client.rpc(self.query_name, match_documents_params)

        if postgrest_filter:
            if hasattr(query_builder, "params"):
                query_builder.params = query_builder.params.set(
                    "and", f"({postgrest_filter})"
                )
            else:
                query_builder.request.params = query_builder.request.params.set(
                    "and", f"({postgrest_filter})"
                )

        if hasattr(query_builder, "params"):
            query_builder.params = query_builder.params.set("limit", k)
        else:
            query_builder.request.params = query_builder.request.params.set("limit", str(k))

        res = query_builder.execute()

        match_result = [
            (
                Document(
                    metadata=search.get("metadata", {}),
                    page_content=search.get("content", ""),
                ),
                search.get("similarity", 0.0),
            )
            for search in res.data
            if search.get("content")
        ]

        if score_threshold is not None:
            match_result = [
                (doc, similarity)
                for doc, similarity in match_result
                if similarity >= score_threshold
            ]
            if len(match_result) == 0:
                warnings.warn(
                    "No relevant docs were retrieved using the relevance score"
                    f" threshold {score_threshold}"
                )

        return match_result

    @staticmethod
    def _add_vectors(
        client: Client,
        table_name: str,
        vectors: List[List[float]],
        documents: List[Any],
        ids: List[str],
        chunk_size: int,
        **kwargs: Any,
    ) -> List[str]:
        """Add vectors to Supabase table using insert (letting DB generate auto-incrementing bigint id)."""
        rows: List[dict] = [
            {
                # We intentionally omit the 'id' field so that Supabase/PostgreSQL
                # automatically generates the bigint identity/serial primary key.
                "content": documents[idx].page_content,
                "embedding": embedding,
                "metadata": documents[idx].metadata,
                **kwargs,
            }
            for idx, embedding in enumerate(vectors)
        ]
        id_list: List[str] = []
        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]

            # Use insert instead of upsert to let PostgreSQL generate auto-incrementing bigint primary key
            result = client.from_(table_name).insert(chunk).execute()

            if len(result.data) == 0:
                raise Exception("Error inserting: No rows added")

            # VectorStore.add_vectors returns ids as strings
            ret_ids = [str(item.get("id")) for item in result.data if item.get("id")]
            id_list.extend(ret_ids)

        return id_list


class SupabaseModule:
    def __init__(self):
        url = config.SUPABASE_URL
        key = config.SUPABASE_SERVICE_KEY
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        logger.info("Connecting to Supabase  │  url=%s", url)
        self.client: Client = create_client(url, key)

    def create_collection(self, collection_name):
        logger.info("Using Supabase single table structure. 'create_collection' is a no-op for %s", collection_name)
        pass

    def get_vectorstore(self, collection_name, embeddings):
        store = CustomSupabaseVectorStore(
            client=self.client,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        return store

    def upsert_documents(self, collection_name, chunks, embeddings):
        logger.info("Adding %d documents to Supabase for collection '%s'...", len(chunks), collection_name)
        
        # Inject collection name into metadata of each chunk
        for chunk in chunks:
            chunk.metadata["collection"] = collection_name
            
        store = CustomSupabaseVectorStore(
            client=self.client,
            embedding=embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        
        result = store.add_documents(chunks)
        logger.info("Successfully added %d documents to Supabase.", len(chunks))
        return result

    def create_pipeline_job(self, project_name: str, status: str = "triggered", **kwargs) -> str | None:
        """Create a new job tracking record and return its UUID."""
        try:
            data = {"project_name": project_name, "status": status, **kwargs}
            result = self.client.table("pipeline_jobs").insert(data).execute()
            if result.data and len(result.data) > 0:
                return result.data[0].get("id")
        except Exception as exc:
            logger.error("Failed to create pipeline job tracking record: %s", exc)
        return None

    def update_pipeline_job(self, job_id: str, status: str, **kwargs) -> None:
        """Update an existing job tracking record."""
        try:
            data = {"status": status, **kwargs}
            self.client.table("pipeline_jobs").update(data).eq("id", job_id).execute()
        except Exception as exc:
            logger.error("Failed to update pipeline job %s: %s", job_id, exc)

    def upload_file_to_storage(self, bucket_name: str, file_path: str, object_name: str) -> bool:
        """Upload a file to Supabase Storage. Creates the bucket if it doesn't exist."""
        try:
            # Check if bucket exists, if not create it
            try:
                self.client.storage.get_bucket(bucket_name)
            except Exception:
                logger.info("Bucket '%s' not found. Creating it...", bucket_name)
                self.client.storage.create_bucket(bucket_name, {"name": bucket_name, "public": True})

            with open(file_path, "rb") as f:
                res = self.client.storage.from_(bucket_name).upload(
                    path=object_name,
                    file=f,
                    file_options={"upsert": "true"}
                )
            logger.info("Successfully uploaded %s to bucket %s", object_name, bucket_name)
            return True
        except Exception as exc:
            logger.error("Failed to upload file to storage: %s", exc)
            return False

    def get_public_url(self, bucket_name: str, object_name: str) -> str:
        """Get the public URL for an object in Supabase Storage."""
        try:
            res = self.client.storage.from_(bucket_name).get_public_url(object_name)
            return res
        except Exception as exc:
            logger.error("Failed to get public URL: %s", exc)
            return ""

