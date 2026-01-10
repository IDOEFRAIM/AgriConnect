import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from orchestrator.main_orchestrator import MainOrchestrator
from orchestrator.state import GlobalAgriState

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgConnectAPI")

app = FastAPI(title="AgConnect API", description="Agricultural Assistant Backend", version="1.0.0")

# Initialize Orchestrator
try:
    orchestrator_instance = MainOrchestrator()
    logger.info("✅ Orchestrator initialized successfully.")
except Exception as e:
    logger.error(f"❌ Failed to initialize Orchestrator: {e}")
    raise e

class UserRequest(BaseModel):
    user_id: str = "user_123"
    zone_id: str = "Centre"
    query: Optional[str] = ""
    flow_type: str = "MESSAGE" # MESSAGE or REPORT

@app.post("/api/v1/ask")
async def ask_agent(req: UserRequest):
    """
    Main endpoint to interact with the AgConnect Orchestrator.
    Handles both conversational queries (MESSAGE) and report generation (REPORT).
    """
    logger.info(f"📨 Received request: flow={req.flow_type}, query='{req.query}'")
    
    # Construct Initial State based on GlobalAgriState definition
    initial_state: GlobalAgriState = {
        "requete_utilisateur": req.query, 
        "zone_id": req.zone_id,
        "flow_type": req.flow_type,
        "execution_path": [],
        # Initialize empty data containers
        "meteo_data": {},
        "market_data": {},
        "user_profile": {"id": req.user_id}
    }
    
    try:
        # Run Orchestrator
        result = orchestrator_instance.run(initial_state)
        
        # Extract relevant response based on flow type
        final_output = ""
        if req.flow_type == "MESSAGE":
            final_output = result.get("final_response", "Je n'ai pas pu générer de réponse.")
        elif req.flow_type == "REPORT":
            report_data = result.get("final_report", {})
            final_output = report_data.get("content", "Rapport vide.")
            
        return {
            "status": "success",
            "response": final_output,
            "trace": result.get("execution_path", []) 
        }
            
    except Exception as e:
        logger.error(f"❌ Orchestrator execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "active", "component": "AgConnect Backend"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
