import requests


def ollama_chat(
    prompt: str,
    model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
) -> str:
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a research-paper assistant. Answer only from the evidence "
                    "supplied by the user. If the evidence does not support an answer, say so."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.2},
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return ((data.get("message") or {}).get("content", "").strip() or "No response from model.")
    except requests.exceptions.ConnectionError:
        return "Could not connect to Ollama at http://localhost:11434. Open the Ollama app or run `ollama serve`."
    except requests.exceptions.Timeout:
        return "Ollama timed out while generating the answer."
    except Exception as exc:
        return f"Ollama error: {exc}"
