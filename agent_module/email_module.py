import requests
import logging
import config

logger = logging.getLogger("email_module")

def send_email_notification(subject: str, message: str):
    """
    Sends an email notification via a POST request to a configured webhook (e.g., Power Automate).
    """
    webhook_url = config.POWER_AUTOMATE_WEBHOOK_URL
    
    if not webhook_url:
        logger.warning("Email notification skipped: POWER_AUTOMATE_WEBHOOK_URL is not configured.")
        return False

    payload = {
        "asunto": subject,
        "message": message
    }

    try:
        logger.info("Sending email notification...")
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Email notification sent successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")
        return False
