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

    if pdf_base64:
        try:
            from supabase_module.supabase_client import SupabaseModule
            supabase_module = SupabaseModule()
            
            # Lookup the real notebook_id from Supabase using vector_store_id
            res = supabase_module.client.table("notebooks").select("id").eq("vector_store_id", vector_store_id).order("created_at", desc=True).limit(1).execute()
            
            if res.data and len(res.data) > 0:
                real_notebook_id = res.data[0]["id"]
                report_payload = {
                    "notebook_id": real_notebook_id,
                    "title": f"Reporte — {project_name}",
                    "summary": report_summary or f"Análisis de licitación completado para {project_name}.",
                    "pdf_base64": pdf_base64,
                }
                supabase_module.client.table("reports").insert(report_payload).execute()
                logger.info("Notifier  │  ✓  Report directly inserted to Supabase for notebook %s", real_notebook_id)
            else:
                logger.error("Notifier  │  ✗  Could not find notebook with vector_store_id '%s' in database. Cannot insert report.", vector_store_id)
        except Exception as exc:
            logger.error("Notifier  │  ✗  Direct Supabase report insertion failed: %s", exc)

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
