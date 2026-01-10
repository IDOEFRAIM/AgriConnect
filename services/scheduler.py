import logging
import time
import sys
import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.services_orchestrator import ScraperOrchestrator
from services.utils.ingestor import DataIngestor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgConnect.Scheduler")

def run_daily_pipeline():
    logger.info("⏰ Starting Daily Ingestion Pipeline...")
    try:
        # 1. Scrape Data
        orchestrator = ScraperOrchestrator(headless=True)
        report = orchestrator.run_all(save_report=True)
        
        # 2. Ingest to RAG
        if report.get("overall_status") != "FAILURE":
            scraped_data = []
            # Flatten results from different pipelines
            raw_pipelines = report.get("data_pipelines", {})
            for pipeline_name, result in raw_pipelines.items():
                if result.get("status") == "SUCCESS":
                    items = result.get("results", [])
                    logger.info(f"Using {len(items)} items from {pipeline_name}")
                    scraped_data.extend(items)
            
            if scraped_data:
                logger.info(f"📥 Ingesting {len(scraped_data)} items into Vector Store...")
                ingestor = DataIngestor()
                ingestor.ingest_data_from_orchestrator(scraped_data)
                logger.info("✅ Ingestion complete.")
            else:
                logger.warning("⚠️ No data collected to ingest.")
        else:
            logger.error("❌ Scraping failed. Skipping ingestion.")
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)

def start_scheduler():
    scheduler = BackgroundScheduler()
    # Trigger everyday at 06:00
    scheduler.add_job(run_daily_pipeline, CronTrigger(hour=6, minute=0), id="daily_ingest")
    
    logger.info("⏳ Scheduler started. Next run at 06:00.")
    scheduler.start()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler shutdown.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--now":
        logger.info("🧪 Test Run (Immediate)...")
        run_daily_pipeline()
    else:
        start_scheduler()
