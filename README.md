# ollama-structured

Lightweight client for structured JSON calls to an Ollama server. Works with dataclasses or Pydantic models, builds a schema-aware prompt, parses the JSON, and instantiates your class. Includes a small demo (`demo_structured.py`).

## Install

SSH (recommended):
```bash
pip install git+ssh://git@github.com/<your-username>/ollama-structured.git@v0.1.1
```
HTTPS:
```bash
pip install git+https://github.com/<your-username>/ollama-structured.git@v0.1.1
```
Editable from a local clone:
```bash
pip install -e /path/to/ollama-structured
```

## Usage
```python
from dataclasses import dataclass
from ollama import OllamaClient

@dataclass
class Joke:
    setup: str
    punchline: str

client = OllamaClient(model="gpt-oss:20b")
joke = client.run("Tell a short clean joke as JSON", Joke)
print(joke.setup, joke.punchline)
```

## Settings
- Default endpoint: `http://localhost:11434/api/generate`
- Logs to `logs/ollama_structured.log` alongside your project root.
- Optional Pydantic support (install `pydantic>=2`).

## License
MIT
