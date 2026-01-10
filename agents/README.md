# Specialized Agents

This directory contains the **Autonomous Agents** of the AgConnect system. Unlike simple Tools (which are passive functions), these Agents possess an **internal reasoning loop**, manage their own state (via LangGraph), and can orchestrate multiple tools to solve complex problems.

Each agent is a specialized "expert" in a specific domain of agriculture.

## 🧑‍🌾 Agent Roster

### 1. Production Expert (`production_expert.py`)
*   **Role**: Senior Agronomist.
*   **Capabilities**:
    *   Plans the entire crop cycle (Sowing -> Harvest).
    *   Validates technical constraints (Sowing dates per zone, density per hectare).
    *   Calculates fertilizer needs (NPK, Urea) based on surface area.
    *   **Key Feature**: "Blocking Alert" – prevents bad decisions (e.g., sowing a long-cycle variety in the North where the season is too short).
*   **Tools Used**: `BurkinaCropTool`.

### 2. Plant Doctor (`plant_doctor.py`)
*   **Role**: Phytopathologist.
*   **Capabilities**:
    *   **Semantic Diagnosis**: Translates vague farmer descriptions (e.g., "purple flowers killing my millet") into technical disease names (e.g., "Striga/Wongo").
    *   Prescribes biological and chemical treatments.
    *   Provides prevention advice.
*   **Tools Used**: `HealthDoctorTool`.

### 3. Climate Vigilance (`climate_vigilance.py`)
*   **Role**: Climatologist & Risk Manager.
*   **Capabilities**:
    *   Analyzes weather forecasts to advise on daily operations (e.g., "Do not spray today, heavy rain expected").
    *   **Flood Risk**: Calculates localized flood risks using satellite data (Fanfar/Glofas).
    *   Determines optimal sowing windows based on recent rainfall.
*   **Tools Used**: `SahelAgriAdvisor`, `FloodRiskTool`.

### 4. Soil Doctor (`soil_doctor.py`)
*   **Role**: Pedologist (Soil Scientist).
*   **Capabilities**:
    *   Identifies soil types and their suitability for specific crops.
    *   **Compost Tracker**: Special feature to manage organic manure/compost maturity cycles.
    *   Recommends soil amendments to improve fertility.
*   **Tools Used**: `SoilDoctorTool`.

### 5. Agri-Business Coach (`agri_business_coach.py`)
*   **Role**: Market Analyst & Financial Advisor.
*   **Capabilities**:
    *   **Scam Detection**: Uses semantic analysis to flag suspicious offers.
    *   Connects farmers with subsidies and market offers.
    *   Analyzes "BUY" vs "SELL" intents to provide relevant market prices.
*   **Tools Used**: `AgrimarketTool`.

## ⚙️ How Agents Work

All agents follow a similar **Graph Architecture** (using LangGraph):

1.  **Input**: Receives a `User Query` and an `AgentState`.
2.  **Reasoning/Validation Node**: The agent checks if it has enough info (e.g., "Do I know the crop type?"). If not, it asks clarifying questions.
3.  **Tool Execution Node**: Calls the deterministic `tools/` functions to get raw data.
4.  **Synthesis Node**: Uses an LLM (Mistral/Llama3 via Ollama) to translate the raw data into a natural language response tailored to a Sahelian farmer.
5.  **Output**: Returns a final advice string.

