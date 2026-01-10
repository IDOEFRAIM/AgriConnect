# Services

This folder groups backend services and logic for heavy or external data processing.

## Responsibilities

*   **Scraping**: Retrieval of meteorological data (Fanfar, Satellites) or documents (SONAGESS).
*   **Processing**: Processing raw data (e.g., flood risk calculation).
*   **Evaluation**: Scripts to evaluate system performance (`evaluation.py`).

These services are often invoked periodically or on demand by Agents/Tools.

