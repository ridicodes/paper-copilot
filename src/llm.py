import requests

def ollama_chat(prompt: str, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434") -> str:
    """
    Calls Ollama's local chat API and returns the assistant message text.
    Requires Ollama running locally (default port 11434).
    """
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You answer using only the provided evidence. If not in evidence, say Not found in the paper."},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content", "").strip() or "No response from model."
    except requests.exceptions.ConnectionError:
        return "Could not connect to Ollama at http://localhost:11434. Open the Ollama app (or run `ollama serve`)."
    except Exception as e:
        return f"Ollama error: {e}"
