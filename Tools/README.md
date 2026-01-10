# Tools (The Functional Bricks)

This directory contains the "**Atomic Tools**" of AgConnect. Unlike Agents, tools are **deterministic**, **stateless**, and **do not use LLMs**. They are pure Python functions that return structured data (JSON/Dict).

They act as the "hands and eyes" of the intelligent agents.

## 🧰 Tool Inventory

### 1. Crop Tool (`crop/`)
*   **File**: `base_crop.py`
*   **Class**: `BurkinaCropTool`
*   **Description**: The encyclopedic database of crops adapted to Burkina Faso.
*   **Key Data**:
    *   **Crop Cycles**: Duration in days for varieties (e.g., Maize SR21, Millet IKMP).
    *   **Sowing Windows**: Recommended dates per climatic zone (Sahelian, Sudano-Sahelian, Sudanian).
    *   **Density**: Recommended spacing (e.g., "80cm x 40cm").
*   **Usage**: Used by `ProductionExpert` to validate if a farmer is sowing at the right time.

### 2. Meteorological Tool (`meteo/`)
*   **Files**: `basis_tools.py`, `flood_risk.py`
*   **Classes**: `SahelAgriAdvisor`, `FloodRiskTool`
*   **Description**: Handles weather data and risk calculations.
*   **Key Functions**:
    *   **check_flood_risk(lat, lon)**: Connects to APIs (Fanfar/Glofas logic) to estimate flood danger levels (Normal, High, Critical).
    *   **analyze_rainfall(amount_mm)**: Determines if rainfall is beneficial or harmful for specific crop stages.
*   **Usage**: Used by `ClimateVigilance` and `DailyReportFlow`.

### 3. Health Tool (`health/`)
*   **File**: `base_health.py`
*   **Class**: `HealthDoctorTool`
*   **Description**: A diagnostic engine for plant diseases.
*   **Key Features**:
    *   **Keyword Matching**: Matches observed symptoms (e.g., "yellow leaves") against a database of pathogens.
    *   **Treatment Database**: Returns organic (Bio) and chemical prescriptions.
    *   **Severity Scoring**: Estimates how critical the infection is.
*   **Usage**: Used by `PlantDoctor`.

### 4. Soil Tool (`Soils/`)
*   **File**: `base_soil.py`
*   **Class**: `SoilDoctorTool`
*   **Description**: Expert system for soil analysis (Pedology).
*   **Key Concepts**:
    *   **Soil Profiles**: Defines characteristics of local soils (e.g., "Zippélé" - Hardpan, "Baongo" - Lowland clay).
    *   **CES Techniques**: Recommends Water & Soil Conservation methods (Zaï, Stone Lines, Half-Moons) based on the soil type.
*   **Usage**: Used by `SoilDoctor`.

### 5. Market Tool (`subventions/`)
*   **File**: `base_subsidy.py`
*   **Class**: `AgrimarketTool`
*   **Description**: Economic intelligence for the Sahelian market.
*   **Key Data**:
    *   **Market Prices**: Tracks base price (harvest) vs. lean season price to advise on storage (Warrantage).
    *   **Regulation Status**: Flags if a crop price is State-regulated (e.g., Cotton).
    *   **Offers**: Listing of available subsidies or buyers.
*   **Usage**: Used by `AgriBusinessCoach` and `DailyReportFlow`.

## 🔄 Interaction with Agents

The relationship is hierarchical:
1.  **Agent** (Reasoning) -> Decides it needs info (e.g., "What is the sowing date for Maize in the South?").
2.  **Tool** (Execution) -> Looks up the static JSON/DB (`BurkinaCropTool`).
3.  **Agent** (Synthesis) -> Receives `{ "sowing_start": "June 15" }` and formulates a sentence: "You should start sowing around mid-June."

