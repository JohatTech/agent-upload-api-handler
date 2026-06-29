import logging
from pathlib import Path
from xhtml2pdf import pisa
import markdown

logger = logging.getLogger("report_generator")

# ── Styling from USER ────────────────────────────────────────────────────────
STYLING = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: Arial, sans-serif;
    background-color: #fafafa;
    padding: 20px;
    color: #333;
    line-height: 1.6;
  }
  .wrapper {
    max-width: 700px;
    margin: auto;
    background: #ffffff;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #eee;
  }
  .header {
    background-color: #f57c00;
    padding: 18px;
  }
  .header h1 {
    color: #ffffff;
    font-size: 20px;
    font-weight: bold;
  }
  .header p {
    color: #ffe0b2;
    margin-top: 5px;
    font-size: 14px;
  }
  .body {
    padding: 20px;
    color: #333;
  }
  .body h2 {
    margin-top: 0;
    color: #f57c00;
    font-size: 17px;
    margin-bottom: 12px;
  }
  .body h3 {
    color: #f57c00;
    margin-top: 20px;
    margin-bottom: 8px;
  }
  .body p { margin-bottom: 10px; }
  .body ul, .body ol { padding-left: 20px; margin-bottom: 10px; }
  .body li { margin-bottom: 4px; }
  .body code {
    background: #fff3e0;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    color: #e65100;
  }
  .body pre {
    background: #fff3e0;
    padding: 15px;
    overflow-x: auto;
    border-left: 5px solid #f57c00;
    border-radius: 4px;
    margin-bottom: 10px;
  }
  .body table {
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0;
    font-size: 14px;
  }
  .body th, .body td {
    border: 1px solid #ddd;
    padding: 10px 12px;
    text-align: left;
  }
  .body th {
    background-color: #fff3e0;
    color: #e65100;
    font-weight: bold;
  }
  .footer {
    background-color: #eeeeee;
    padding: 15px 20px;
    font-size: 12px;
    color: #777;
    text-align: center;
  }
</style>
"""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  {styling}
</head>
<body>
  <div class="wrapper">
    <div class="header">
      <h1>📋 Resumen de proyecto de licitación</h1>
      <p>Resumen generado automáticamente por un modelo de inteligencia artificial</p>
    </div>
    <div class="body">
      <h2>📌 Resumen del proyecto</h2>
      {html_body}
    </div>
    <div class="footer">
      Este correo fue generado automáticamente por un sistema de inteligencia artificial.<br>
      Por favor, no responda directamente a este mensaje.
    </div>
  </div>
</body>
</html>
"""

import config

def generate_report(project_name: str, markdown_content: str, model_name: str = "DEFAULT") -> Path:
    if getattr(config, "IS_VERCEL", False):
        reports_root = Path("/tmp/reports")
    else:
        reports_root = Path("reports")
    reports_root.mkdir(exist_ok=True)
    
    # Define model-specific output directory
    safe_project = "".join([c if c.isalnum() else "_" for c in project_name])
    model_dir = reports_root / safe_project / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert Markdown to HTML
    html_body = markdown.markdown(markdown_content, extensions=['tables', 'fenced_code'])
    
    # Wrap in template
    full_html = HTML_TEMPLATE.format(styling=STYLING, html_body=html_body)
    
    # Define output path
    pdf_path = model_dir / f"Report_{safe_project}_{model_name}.pdf"
    
    # Convert HTML to PDF
    logger.info("Generating PDF report │ project=%s │ model=%s", project_name, model_name)
    with open(pdf_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)
        
    if pisa_status.err:
        logger.error("Failed to generate PDF report.")
        raise RuntimeError("PDF generation failed")
        
    logger.info("PDF report saved to: %s", pdf_path)
    return pdf_path
