import json
import os
import logging
import time
from datetime import datetime
from typing import Dict, Any, List

# --- Configuration ---
# In a real cloud environment, these would be environment variables
FANFAR_URL = os.environ.get("FANFAR_URL", "https://fanfar.eu/fr/piv/")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "agconnect-flood-data")
LOCAL_OUTPUT_PATH = "data/floods/fanfar_latest.json"

# Setup Logging
logger = logging.getLogger("CloudFloodUpdater")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# --- Mock Scraper Logic (to replace heavy Playwright in Lambda if needed) ---
# In a real scenario, you might use a lighter scraper or a headless browser layer for Lambda.
# For this example, we will simulate the scraping logic or import the existing one if available.

def fetch_flood_data() -> List[Dict[str, Any]]:
    """
    Orchestrates the data fetching.
    In a full implementation, this would call the Playwright scraper.
    Here we simulate a robust fetch or call the existing tool if possible.
    """
    logger.info("Starting flood data fetch...")
    
    # Simulation of fetching data from FANFAR API/Site
    # In production, import services.meteo.fanfar_scrapper here
    
    # Mock Data for demonstration (representing what the scraper would return)
    current_month = datetime.now().month
    is_rainy = 6 <= current_month <= 9
    
    mock_features = [
        {
            "type": "Feature",
            "properties": {
                "station_name": "Ouagadougou-Station",
                "risk_level": "Modéré" if is_rainy else "Faible",
                "water_level": 120 if is_rainy else 45,
                "timestamp": datetime.now().isoformat()
            },
            "geometry": {
                "type": "Point",
                "coordinates": [-1.52, 12.37]
            }
        },
        {
            "type": "Feature",
            "properties": {
                "station_name": "Bobo-Dioulasso-Station",
                "risk_level": "Faible",
                "water_level": 80,
                "timestamp": datetime.now().isoformat()
            },
            "geometry": {
                "type": "Point",
                "coordinates": [-4.29, 11.17]
            }
        }
    ]
    
    logger.info(f"Fetched {len(mock_features)} data points.")
    return mock_features

def save_data(data: List[Dict[str, Any]], destination: str):
    """
    Saves data to the specified destination.
    Handles local file system or Cloud Storage (S3/Blob) based on prefix.
    """
    payload = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "source": "CloudFloodUpdater"
        },
        "features": data
    }
    
    if destination.startswith("s3://"):
        # AWS S3 Logic
        # import boto3
        # s3 = boto3.client('s3')
        # bucket, key = parse_s3_url(destination)
        # s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(payload))
        logger.info(f"Uploading to S3: {destination} (Simulated)")
        
    elif destination.startswith("azure://"):
        # Azure Blob Logic
        logger.info(f"Uploading to Azure Blob: {destination} (Simulated)")
        
    else:
        # Local File System
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved locally to {destination}")

# --- AWS Lambda Handler ---
def lambda_handler(event, context):
    """
    Entry point for AWS Lambda.
    Triggered by EventBridge (Cron).
    """
    logger.info("AWS Lambda triggered")
    try:
        data = fetch_flood_data()
        # Save to S3 in production
        # save_data(data, f"s3://{OUTPUT_BUCKET}/fanfar_latest.json")
        
        # For this workspace context, we save locally to simulate the update
        save_data(data, LOCAL_OUTPUT_PATH)
        
        return {
            'statusCode': 200,
            'body': json.dumps('Flood data updated successfully!')
        }
    except Exception as e:
        logger.error(f"Error in Lambda: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error: {str(e)}")
        }

# --- Azure Function Handler ---
def main(req: Any) -> str:
    """
    Entry point for Azure Functions.
    Triggered by Timer Trigger.
    """
    logger.info("Azure Function triggered")
    try:
        data = fetch_flood_data()
        # Save to Blob Storage in production
        # save_data(data, "azure://container/fanfar_latest.json")
        
        save_data(data, LOCAL_OUTPUT_PATH)
        return "Flood data updated successfully!"
    except Exception as e:
        logger.error(f"Error in Azure Function: {e}")
        return f"Error: {str(e)}"

# --- Local Execution (Test) ---
if __name__ == "__main__":
    print("--- Running Cloud Flood Updater Locally ---")
    # Simulate an AWS event
    result = lambda_handler({}, None)
    print("Result:", result)
