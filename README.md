# Agentic AI Multi‑Agent System

This project demonstrates how to build a collaborative multi‑agent AI system **from scratch** using plain Python and Streamlit. Inspired by modern agentic patterns but deliberately lightweight, it features:

* 🧠 **Specialised agents** for summarisation, research‑article generation, and PHI sanitisation.
* ✅ **Validator agents** that review outputs and provide qualitative feedback.
* 🔁 **Reflection patterns**: agents can critique and revise their own drafts for higher quality.
* 🔌 **Pluggable LLM providers**: switch seamlessly between OpenAI’s API and a local Ollama model via a toggle.
* 🧾 **Protocol & tracing**: every interaction is recorded as a JSONL trace for observability and debugging.
* 🧠 **Memory utilities**: simple sliding‑window and entity memories for continuity across turns.
* 📊 **Smoke tests**: a small suite of evaluation scripts to sanity‑check the system.

The goal of this repository is to provide a transparent, educational example of building agentic AI tooling without relying on orchestration frameworks. Use it as a stepping stone for your own projects or presentations.

## Getting Started

### Prerequisites

* **Python 3.9+** (3.11 recommended) – [Download here](https://www.python.org/downloads/)
* An OpenAI API key **or** a running [Ollama](https://github.com/ollama/ollama) server for local models

### Installation

1. **Clone or download the repository**

   ```bash
   git clone https://github.com/your-org/agentic-ai-multi-agent-system.git
   cd agentic-ai-multi-agent-system
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Copy `env.example` to `.env` and fill in your credentials:

   ```bash
   cp env.example .env
   # edit the .env file in your favourite editor
   ```

   At minimum you must set `OPENAI_API_KEY` to your API key if not using Ollama. To run against a local model, set `USE_OLLAMA=true` and ensure an Ollama server is running at `http://localhost:11434`.

### Running the App

Launch the Streamlit interface with:

```bash
streamlit run app.py
```

Open the provided URL (usually `http://localhost:8501`) in your browser. Use the sidebar to pick a task and toggle between OpenAI and Ollama. The **Run** tab lets you interact with the agents; the **Traces** tab displays recent conversations, token usage and latency.

### Running with Docker

If you prefer containers, build and run everything with a single command using Docker Compose:

```bash
docker compose up --build
```

This spins up the Streamlit app and an Ollama container. You can still provide an OpenAI API key via environment variables if you wish.

### Smoke Tests

A small suite of smoke tests lives in `evals/smoke_tests.py`. To run them:

```bash
python -m evals.smoke_tests
```

These tests will call your configured language model. They are intended as quick sanity checks; more robust evaluation is left to the reader.

## Project Structure

```text
agentic_repo/
├── agents/
│   ├── agent_base.py          # common agent functionality
│   ├── summarize_tool.py      # produces medical summaries
│   ├── write_article_tool.py  # generates research articles
│   ├── sanitize_data_tool.py  # removes PHI from data
│   ├── summarise_validator... # validates summaries
│   ├── write_article_validator_agent.py
│   ├── sanitize_data_validator_agent.py
│   ├── refiner_agent.py       # improves drafts
│   ├── validator_agent.py     # generic article validator
│   ├── patterns.py            # reflection/revision loop
│   └── protocol.py            # message & exchange data models
├── utils/
│   ├── model.py               # pluggable LLM provider
│   ├── tracer.py              # writes JSONL traces
│   ├── memory.py              # simple memory classes
│   └── logger.py              # logging configuration
├── app.py                     # Streamlit UI
├── evals/
│   └── smoke_tests.py         # quick evaluations
├── docker-compose.yml         # container orchestration
├── Dockerfile                 # container build definition
├── requirements.txt           # Python dependencies
├── env.example                # environment template
└── README.md                  # this file
```

## Contributing

Contributions are welcome! If you spot a bug or have ideas for additional features (e.g. more agent types, richer memory, more evaluation metrics), please open an issue or pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.