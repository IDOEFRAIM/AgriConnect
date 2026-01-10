# AgConnect

AgConnect is an intelligent agricultural assistant designed to help farmers manage their crops, monitor risks (floods, diseases), and access market information.

The system uses a modular architecture based on a **Central Orchestrator**, **Specialized Agents**, and an advanced **RAG (Retrieval-Augmented Generation)** system.

## 🏗️ Architecture

The project is structured into several key components:

*   **Orchestrator (`orchestrator/`)**: The heart of the system. It determines the user's intent (via `intention.py`) and routes the request either to a conversational flow (`MessageResponseFlow`) or an analysis report flow (`DailyReportFlow`).
*   **Tools (`tools/`)**: Functional "bricks" providing specific data (Weather, Soil, Market, Health, Crop).
*   **RAG (`rag/`)**: Vector document retrieval system to enrich responses with technical and contextual data.
*   **Services (`services/`)**: Business services for data scraping, flood report generation, etc.
*   **Frontend (`frontend/`)**: User Interface (Gradio/Web).
*   **Backend (`backend/`)**: API to expose functionalities.

## 🚀 Installation

This project uses **Poetry** for dependency management.

1.  **Prerequisites**
    *   Python 3.10+
    *   [Ollama](https://ollama.com/) (for local LLM models like Mistral/Llama3)

2.  **Installation**
    ```bash
    poetry install
    ```

3.  **Configuration**
    Ensure you have the necessary environment variables configured in a `.env` file.

## 🛠️ Usage

### Run the Orchestrator
To start the main application flow:

```bash
# Example (adapt according to your main entry point)
python -m orchestrator.main_orchestrator
```

### Run Tests
We have a comprehensive test suite covering tools and orchestration.

```bash
# Run all tests
python -m unittest discover -s tests -t .
```

## 📂 Project Structure

*   `agents/`: Logic for specialized agents.
*   `orchestrator/`: Management of decision and execution flow.
*   `rag/`: Vector search engine and re-ranking.
*   `tools/`: Atomic tools (Crop, Flood, Health, Market, Soil).
*   `tests/`: Unit and integration tests.
*   `data/`: Static or downloaded data.

## 🤝 Contribution

Contributions are welcome! Please consult [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for our community rules.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.







