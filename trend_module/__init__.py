"""
Trend Analytics Module for Hermes Notebooks.
Provides independent extraction of engineering fields, project types, regions, countries, budget scales,
funding sources, and technical requirements from project report summaries.
"""
from trend_module.schemas import ProjectTrendExtraction
from trend_module.extractor import extract_project_trends
from trend_module.service import process_notebook_trend_analytics, process_all_pending_trends

__all__ = [
    "ProjectTrendExtraction",
    "extract_project_trends",
    "process_notebook_trend_analytics",
    "process_all_pending_trends",
]
