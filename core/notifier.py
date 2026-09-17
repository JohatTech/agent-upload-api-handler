"""
Sends HTTP callbacks to external systems (N8N, Power Automate, Frontend) after a project has been fully ingested.
"""

import logging
from datetime import datetime, timezone
from typing import Any
import requests
import base64
from pathlib import Path
import config

logger = logging.getLogger("notifier")


def build_payload(
    project_name: str,
    collection_name: str,
    total_chunks: int,
    vectorstore_targets: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "project_name": project_name,
        "collection_name": collection_name,
        "total_chunks": total_chunks,
        "vectorstore_targets": vectorstore_targets,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    return payload


def notify_n8n(payload: dict[str, Any]) -> bool:
    webhook_url = getattr(config, "N8N_WEBHOOK_URL", None)
    if not webhook_url:
        logger.warning("N8N_WEBHOOK_URL is not set – skipping notification.")
        return False

    logger.info("Notifier  │  Sending callback to N8N  →  project='%s'", payload.get("project_name", "unknown"))

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        response.raise_for_status()
        logger.info("Notifier  │  ✓  N8N responded %d", response.status_code)
        return True
    except Exception as exc:
        logger.error("Notifier  │  ✗  N8N callback failed: %s", exc)
        return False


def notify_power_automate(
    project_name: str,
    email: str,
    pdf_path: str | Path,
) -> bool:
    webhook_url = config.POWER_AUTOMATE_WEBHOOK_URL
    if not webhook_url:
        logger.warning("POWER_AUTOMATE_WEBHOOK_URL is not set – skipping notification.")
        return False

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        logger.error("Report file not found at %s", pdf_path)
        return False

    logger.info("Notifier  │  Sending report to Power Automate  →  project='%s', email='%s'", project_name, email)

    try:
        with open(pdf_file, 'rb') as f:
            file_content_b64 = base64.b64encode(f.read()).decode('utf-8')

        payload = {
            "fileName": pdf_file.name,
            "fileContent": file_content_b64,
            "Asunto": f"Reporte de Licitación - {project_name}",
            "Message": f"Adjunto se encuentra el reporte generado para el proyecto {project_name}.",
            "correo": email
        }

        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        logger.info("Notifier  │  ✓  Power Automate responded %d", response.status_code)
        return True
    except Exception as exc:
        logger.error("Notifier  │  ✗  Power Automate callback failed: %s", exc)
        return False


def notify_frontend(
    project_name: str,
    vector_store_id: str,
    total_chunks: int = 0,
    status: str = "ready",
    pdf_path: str | Path | None = None,
    report_summary: str | None = None,
    file_count: int = 0,
    tag: str | None = None,
    uploaded_files: list[tuple[str, str]] | None = None,
    notebook_id: str | None = None,
) -> bool:
    pdf_base64 = None
    if pdf_path:
        pdf_file = Path(pdf_path)
        if pdf_file.exists():
            try:
                with open(pdf_file, "rb") as f:
                    pdf_base64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                logger.error("Failed to read PDF for base64 encoding: %s", e)

    try:
        from supabase_module.supabase_client import SupabaseModule
        import re
        import uuid
        supabase_module = SupabaseModule()
        
        real_notebook_id = notebook_id
        existing_nb = None

        # 1. Primary lookup: by vector_store_id (authoritative for collection identity)
        if vector_store_id:
            vs_res = (
                supabase_module.client.table("notebooks")
                .select("*")
                .eq("vector_store_id", vector_store_id)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            if vs_res.data and len(vs_res.data) > 0:
                existing_nb = vs_res.data[0]
                real_notebook_id = existing_nb["id"]

        # 2. Secondary lookup: by explicit notebook_id if not found via vector_store_id
        if not existing_nb and real_notebook_id:
            nb_res = (
                supabase_module.client.table("notebooks")
                .select("*")
                .eq("id", real_notebook_id)
                .limit(1)
                .execute()
            )
            if nb_res.data and len(nb_res.data) > 0:
                existing_nb = nb_res.data[0]

        clean_name = re.sub(r'^\[[^\]]+\]\s*', '', project_name) if project_name else ""

        # 3. Update existing notebook or insert if completely missing
        if existing_nb:
            update_payload: dict[str, Any] = {
                "status": status,
                "last_activity": datetime.now(timezone.utc).isoformat(),
            }
            if clean_name and clean_name != "Nuevo proyecto":
                update_payload["name"] = clean_name
            if file_count > 0:
                update_payload["file_count"] = file_count
            if tag:
                update_payload["tag"] = tag
            if vector_store_id and not existing_nb.get("vector_store_id"):
                update_payload["vector_store_id"] = vector_store_id

            try:
                supabase_module.client.table("notebooks").update(update_payload).eq("id", real_notebook_id).execute()
                logger.info("Notifier  │  ✓  Updated existing notebook '%s' (%s)", clean_name or real_notebook_id, real_notebook_id)
            except Exception as upd_err:
                logger.error("Failed to update notebook %s: %s", real_notebook_id, upd_err)
        else:
            # Deterministic notebook ID
            if not real_notebook_id:
                real_notebook_id = f"nb-{vector_store_id}" if vector_store_id else str(uuid.uuid4())

            # Resolve owner_id from profiles
            resolved_owner_id = None
            try:
                profiles_res = supabase_module.client.table("profiles").select("id, email, tag").execute()
                if profiles_res.data:
                    vs_prefix = vector_store_id.lower().split("_")[0] if vector_store_id else ""
                    for p in profiles_res.data:
                        p_email = (p.get("email") or "").lower()
                        sanitized_email = re.sub(r'[^a-z0-9]', '', p_email)
                        if sanitized_email and vs_prefix and sanitized_email == vs_prefix:
                            resolved_owner_id = p.get("id")
                            break
                        if tag and p_email == tag.lower():
                            resolved_owner_id = p.get("id")
                            break
            except Exception as p_err:
                logger.warning("Failed to resolve owner_id for notebook insert: %s", p_err)

            nb_payload = {
                "id": real_notebook_id,
                "name": clean_name or project_name or "Nuevo proyecto",
                "project_source": f"blob/{project_name.lower()}" if project_name else "",
                "file_count": file_count,
                "status": status,
                "vector_store_id": vector_store_id,
                "tag": tag,
            }
            if resolved_owner_id:
                nb_payload["owner_id"] = resolved_owner_id

            try:
                supabase_module.client.table("notebooks").insert(nb_payload).execute()
                logger.info("Notifier  │  ✓  Created new notebook '%s' (%s)", clean_name, real_notebook_id)
            except Exception as nb_err:
                # If unique constraint or conflict occurred, fallback to update
                logger.warning("Failed to insert notebook (%s), attempting update on vector_store_id: %s", nb_err, vector_store_id)
                if vector_store_id:
                    try:
                        upd_res = supabase_module.client.table("notebooks").select("id").eq("vector_store_id", vector_store_id).limit(1).execute()
                        if upd_res.data:
                            real_notebook_id = upd_res.data[0]["id"]
                            supabase_module.client.table("notebooks").update({
                                "status": status,
                                "last_activity": datetime.now(timezone.utc).isoformat(),
                                "file_count": file_count if file_count > 0 else 1,
                            }).eq("id", real_notebook_id).execute()
                    except Exception as fall_err:
                        logger.error("Fallback update failed: %s", fall_err)

        # 3. Sync uploaded files
        if real_notebook_id and uploaded_files:
            for file_name, file_url in uploaded_files:
                try:
                    supabase_module.client.table("notebook_files").upsert(
                        {
                            "notebook_id": real_notebook_id,
                            "file_name": file_name,
                            "file_url": file_url,
                        },
                        on_conflict="notebook_id,file_name"
                    ).execute()
                    logger.info("Notifier  │  ✓  Synced notebook_file '%s' with URL %s", file_name, file_url)
                except Exception as nf_err:
                    logger.error("Failed to sync notebook_file '%s': %s", file_name, nf_err)

        # 4. Sync report
        if pdf_base64 and real_notebook_id:
            report_payload = {
                "notebook_id": real_notebook_id,
                "title": f"Reporte — {project_name}",
                "summary": report_summary or f"Análisis de licitación completado para {project_name}.",
                "pdf_base64": pdf_base64,
            }
            existing_report = supabase_module.client.table("reports").select("id").eq("notebook_id", real_notebook_id).limit(1).execute()
            if existing_report.data and len(existing_report.data) > 0:
                supabase_module.client.table("reports").update(report_payload).eq("notebook_id", real_notebook_id).execute()
                logger.info("Notifier  │  ✓  Report directly updated in Supabase for notebook %s", real_notebook_id)
            else:
                supabase_module.client.table("reports").insert(report_payload).execute()
                logger.info("Notifier  │  ✓  Report directly inserted to Supabase for notebook %s", real_notebook_id)
    except Exception as exc:
        logger.error("Notifier  │  ✗  Direct Supabase notebook/report sync failed: %s", exc)

    return True


def notify_chat_responder(
    project_name: str,
    vector_store_id: str,
    natural_project_name: str,
) -> bool:
    """
    HTTP POST callback to the chat-responder service.
    Registers the newly vectorized project's mapping and vector ID so the responder can answer queries.
    """
    responder_url = getattr(config, "CHAT_RESPONDER_URL", None)
    if not responder_url:
        logger.warning("CHAT_RESPONDER_URL is not set – skipping chat responder registration.")
        return False

    endpoint = f"{responder_url.rstrip('/')}/api/register-project"
    logger.info(
        "Notifier  │  Registering project on chat responder  →  project='%s'  │  vectorStoreId='%s'",
        project_name,
        vector_store_id,
    )

    try:
        response = requests.post(
            endpoint,
            json={
                "project_name": project_name,
                "vector_store_id": vector_store_id,
                "natural_project_name": natural_project_name,
            },
            headers={"Content-Type": "application/json"},
            timeout=10,
            verify=False,
        )
        response.raise_for_status()
        logger.info("Notifier  │  ✓  Chat responder registered project successfully.")
        return True
    except Exception as exc:
        logger.error("Notifier  │  ✗  Chat responder registration failed: %s", exc)
        return False

def get_project_title_from_responder(project_name: str) -> str:
    """
    Asks the chat responder server to identify the natural language project name.
    """
    responder_url = getattr(config, "CHAT_RESPONDER_URL", None)
    if not responder_url:
        logger.warning("CHAT_RESPONDER_URL is not set – skipping getting title from chat responder.")
        return project_name

    endpoint = f"{responder_url.rstrip('/')}/api/project-title"
    logger.info("Notifier  │  Requesting project title from chat responder  →  project='%s'", project_name)

    try:
        response = requests.post(
            endpoint,
            json={"project_name": project_name},
            headers={"Content-Type": "application/json"},
            timeout=30,
            verify=False,
        )
        response.raise_for_status()
        data = response.json()
        title = data.get("title", project_name)
        logger.info("Notifier  │  ✓  Chat responder returned title: '%s'", title)
        return title
    except Exception as exc:
        logger.error("Notifier  │  ✗  Failed to get project title from chat responder: %s", exc)
        return project_name
