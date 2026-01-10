import logging
import sys
import os

# Ensure the project root is in sys.path
sys.path.append(os.getcwd())

from orchestrator.main_orchestrator import MainOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestOrchestrator")

def test_orchestrator():
    print("🚀 Initializing MainOrchestrator...")
    try:
        orchestrator = MainOrchestrator()
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        import traceback
        traceback.print_exc()
        return

    # User 1: Message Flow (Météo)
    # Note: 'user_query' key was used in MainOrchestrator, but state.py expected 'requete_utilisateur'
    # Let's check message_flow.py intent classifier. it uses 'requete_utilisateur'.
    # In MainOrchestrator.run, it doesn't change keys.
    # So msg_state should use 'requete_utilisateur'.
    
    msg_state_meteo = {
        "user_id": "user_msg_1",
        "zone_id": "Dédougou",
        "requete_utilisateur": "Va-t-il pleuvoir demain ?",
        "flow_type": "MESSAGE",
        "history": [],
        # Initialize required fields to avoid KeyError in agents if they assume existence
        "meteo_data": {},
        "soil_data": {},
        "health_data": {},
        "market_data": {},
        "global_alerts": [],
        "execution_path": []
    }
    
    print("\n--- TEST 1: MESSAGE FLOW (Intent: METEO) ---")
    try:
        result = orchestrator.run(msg_state_meteo)
        print("✅ Result:", result)
        if "final_response" in result:
            print(f"🤖 Response: {result['final_response']}")
    except Exception as e:
        logger.error(f"Error in MESSAGE flow: {e}")
        import traceback
        traceback.print_exc()

    # User 2: Report Flow
    report_state = {
        "user_id": "user_report_1",
        "zone_id": "Ouagadougou",
        "flow_type": "REPORT",
        "global_alerts": [],
        "execution_path": [],
        "meteo_data": {},
        "market_data": {}
    }
    
    print("\n--- TEST 2: REPORT FLOW ---")
    try:
        result = orchestrator.run(report_state)
        print("✅ Result:", result)
        if "final_report" in result:
            content = result['final_report'].get('content', '')
            print(f"📄 Report Content: {content[:200]}..." if len(content) > 200 else f"📄 Report Content: {content}")
    except Exception as e:
        logger.error(f"Error in REPORT flow: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_orchestrator()
