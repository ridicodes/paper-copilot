import requests

from src.config import DEFAULT_OLLAMA_MODEL, OLLAMA_BASE_URL


def ollama_chat(
    prompt: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
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
        return f"Could not connect to Ollama at {base_url}. Open the Ollama app or run `ollama serve`."
    except requests.exceptions.Timeout:
        return "Ollama timed out while generating the answer."
    except Exception as exc:
        return f"Ollama error: {exc}"
