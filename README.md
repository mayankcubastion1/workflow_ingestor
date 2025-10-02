# Workflow Ingestor

Workflow Ingestor is a Streamlit application for turning natural-language workflow descriptions into interactive diagrams and intent schemas. It can run entirely locally using deterministic heuristics, or optionally call Azure OpenAI / OpenAI models when API keys are provided.

## Features
- Build and edit directed acyclic graphs (DAGs) that represent workflows.
- Convert workflows into Mermaid diagrams and Graphviz visualisations.
- Generate intent schemas from workflows and tool catalogs, using heuristics or LLMs.
- Validate schemas and diff regenerated intents against saved versions.

## Prerequisites
- Python 3.9 or newer
- [Graphviz](https://graphviz.org/download/) system binaries (required for `st.graphviz_chart` rendering)

## Installation
1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Optional API Configuration
The app can enrich intent generation with Azure OpenAI or OpenAI. Provide credentials via environment variables before launching Streamlit:

- **Azure OpenAI**
  - `AZURE_OPENAI_ENDPOINT` (e.g. `https://<resource>.openai.azure.com`)
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_DEPLOYMENT` (deployment name, e.g. `gpt-4o`)
  - `AZURE_OPENAI_API_VERSION` (optional, defaults to `2024-08-01-preview`)
- **OpenAI**
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL` (optional, defaults to `gpt-4o-mini`)

You can store these in a `.env` file in the project root—values will be loaded automatically when the app starts.

## Running the Streamlit App
Launch the Workflow Ingestor UI:
```bash
streamlit run streamlit_workflow_builder_azure.py
```

The interface lets you:
- Describe workflows in natural language and convert them into diagrams.
- Edit nodes/edges directly in tabular form.
- Load sample tools and intents from the [`sample jsons/`](sample%20jsons) directory.
- Generate and validate intent schemas, with optional Git commits for accepted changes.

## Running Tests
This project uses `pytest` for unit tests:
```bash
pytest
```

## Project Structure
- `streamlit_workflow_builder_azure.py` – Streamlit application entry point and UI logic.
- `intent_generator_utils.py` – Intent schema utilities used by the app and tests.
- `sample jsons/` – Example tool catalogs and intent schemas for experimentation.
- `tests/` – Automated tests covering the intent generation helpers.

## Troubleshooting
- Ensure Graphviz is installed and on your `PATH` if diagram previews fail to render.
- Missing API keys fall back to deterministic heuristics; check the sidebar status messages for details.
- When running behind a corporate proxy, configure the appropriate proxy environment variables so that Streamlit can load external resources.
