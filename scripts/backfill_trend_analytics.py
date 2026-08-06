"""
Standalone CLI script to batch process all historic reports and populate project_trend_analytics table in Supabase.
"""
import sys
import logging
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trend_module.service import process_all_pending_trends

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

if __name__ == "__main__":
    print("=" * 70)
    print("HERMES NOTEBOOKS — BATCH TREND ANALYTICS BACKFILL")
    print("=" * 70)
    
    try:
        summary = process_all_pending_trends()
        print("\nProcess Completed Successfully!")
        print(f"Total Pending: {summary['total_pending']}")
        print(f"Successfully Processed: {summary['success_count']}")
        print(f"Failed: {summary['fail_count']}")
        print("=" * 70)
    except Exception as e:
        print(f"\nERROR running backfill process: {e}")
        sys.exit(1)
