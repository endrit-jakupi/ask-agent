# Ask Agent

This AI agent answers questions about the content of a specific website URL.

## Requirements

- Python 3.10+
- Ollama installed and running

```bash
ollama pull llama3.2:3b
```

## Setup

From project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root, and add the following:
Get your LangSmith API key from your LangSmith account and replace the placeholder value.

Example:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
```

## Change the Website URL

Edit `main.py` and update:

- `URL` (currently near the top of the file)

Example:

```python
URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
```

## Run

```bash
python3 main.py
```

Then ask questions in the terminal prompt:

```text
You: What is this page about?
Agent: ...
```

Type `exit` to quit.