import argparse
import json
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from get_embedding_function import get_embedding_function

load_dotenv()

CHROMA_PATH = "chroma"

# xAI Grok-compatible base URL
GROK_BASE_URL = "https://api.x.ai/v1"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

PROMPT_TEMPLATE = """
Answer the question based on the following context. If the context does not explicitly contain the answer but is related, you can use your general knowledge to construct a helpful response, but make sure to state what information comes from the context and what comes from general knowledge.

Context:
{context}

---

Question: {question}
"""

# ─── Module-level singletons (created once, reused every query) ───────────────
_embedding_function = None
_db = None
_models = {}

def check_ollama_connection(base_url: str = OLLAMA_BASE_URL) -> dict:
    """Check if Ollama server is running and return available models."""
    import urllib.request
    try:
        url = f"{base_url.rstrip('/')}/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                model_names = [m.get("name") for m in data.get("models", [])]
                return {"connected": True, "base_url": base_url, "models": model_names}
    except Exception as e:
        return {"connected": False, "base_url": base_url, "error": str(e), "models": []}
    return {"connected": False, "base_url": base_url, "models": []}


def ensure_ollama_running(base_url: str = OLLAMA_BASE_URL) -> bool:
    """Ensure Ollama service is running; auto-launch it in background if needed."""
    import shutil
    import subprocess
    import time

    status = check_ollama_connection(base_url)
    if status.get("connected"):
        return True

    print("Checking Ollama connection: Ollama is offline. Attempting to start Ollama server...")
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        user_profile = os.getenv("USERPROFILE", "")
        candidate = os.path.join(user_profile, "AppData", "Local", "Programs", "Ollama", "ollama.exe")
        if os.path.exists(candidate):
            ollama_path = candidate

    if ollama_path:
        try:
            flags = 0
            if os.name == "nt":
                flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            subprocess.Popen(
                [ollama_path, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=flags
            )
            for _ in range(10):
                time.sleep(0.5)
                if check_ollama_connection(base_url).get("connected"):
                    print(f"Ollama server connected at {base_url}")
                    return True
        except Exception as e:
            print(f"Failed to auto-start Ollama server: {e}")

    return check_ollama_connection(base_url).get("connected", False)


def _is_grok(model_name: str) -> bool:
    """Return True if the model name should be routed to the Grok/xAI API."""
    return "grok" in model_name.lower()


def _is_gemini(model_name: str) -> bool:
    """Return True if the model name should be routed to the Google Gemini API."""
    return "gemini" in model_name.lower()


def _is_chat_model(model_name: str) -> bool:
    """Return True for cloud chat models (Grok, Gemini) that return .content."""
    return _is_grok(model_name) or _is_gemini(model_name)


def _get_db():
    """Return cached Chroma DB connection — only opens once per process."""
    global _db, _embedding_function
    if _db is None:
        _embedding_function = get_embedding_function()
        _db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=_embedding_function
        )
    return _db


def _get_model(model_name: str = "llama3.2:3b"):
    """Return cached model instance by name.
    - Gemini models → Google Generative AI API
    - Grok models   → xAI API (OpenAI-compatible)
    - Others        → Ollama (local)
    """
    global _models
    if model_name not in _models:
        if _is_gemini(model_name):
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Add it to your .env file: GEMINI_API_KEY=AIza..."
                )
            _models[model_name] = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.7,
                streaming=True,
            )
        elif _is_grok(model_name):
            api_key = os.getenv("GROK_API_KEY")
            if not api_key:
                raise ValueError(
                    "GROK_API_KEY is not set. Add it to your .env file: GROK_API_KEY=xai-..."
                )
            _models[model_name] = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=GROK_BASE_URL,
                temperature=0.7,
                streaming=True,
            )
        elif model_name == "mistral":
            _models[model_name] = OllamaLLM(
                model="mistral",
                base_url=OLLAMA_BASE_URL,
                temperature=0.7,
                num_predict=512,
                num_ctx=2048,
            )
        else:  # default: llama3.2:3b (Ollama, fast)
            _models[model_name] = OllamaLLM(
                model=model_name,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,
                num_predict=256,
                num_ctx=2048,
            )
    return _models[model_name]


def invalidate_db_cache():
    """Call this after adding new documents so the DB reloads fresh."""
    global _db
    if _db is not None:
        try:
            # Release the underlying Chroma client so Windows file locks are freed
            if hasattr(_db, '_client'):
                _db._client.clear_system_cache()
            del _db
        except Exception:
            pass
        import gc
        gc.collect()
    _db = None


# ──────────────────────────────────────────────────────────────────────────────

def warm_up():
    """Warm up cached DB and default model at server startup."""
    try:
        ensure_ollama_running()
        _get_db()
        _get_model("llama3.2:3b")
    except Exception as e:
        print(f"Warm up notice: {e}")


def _build_prompt(query_text: str, k: int = 2):
    """Shared retrieval + prompt build used by all query functions."""
    db = _get_db()
    results = db.similarity_search_with_score(query_text, k=k)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    return prompt, results


def _extract_text(content) -> str:
    """Extract clean string text from ChatMessage content (handles string, list of dicts, etc.)."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(item["text"])
                elif "text" in item:
                    parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "".join(parts)
    if hasattr(content, "text"):
        return str(content.text)
    return str(content)


def query_rag(query_text: str, model_name: str = "llama3.2:3b"):
    """Simple query — returns plain text response."""
    prompt, results = _build_prompt(query_text)
    model = _get_model(model_name)
    if _is_chat_model(model_name):
        response_text = _extract_text(model.invoke(prompt).content)
    else:
        response_text = _extract_text(model.invoke(prompt))
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    print(f"Response: {response_text}\nSources: {sources}")
    return response_text


def query_rag_detailed(query_text: str, model_name: str = "llama3.2:3b"):
    """Returns response + structured source metadata for the UI."""
    prompt, results = _build_prompt(query_text)
    model = _get_model(model_name)
    if _is_chat_model(model_name):
        response_text = _extract_text(model.invoke(prompt).content)
    else:
        response_text = _extract_text(model.invoke(prompt))

    sources = [
        {
            "id": doc.metadata.get("id"),
            "content": doc.page_content,
            "score": float(_score)
        }
        for doc, _score in results
    ]

    return {
        "response": response_text,
        "sources": sources
    }


def query_rag_stream(query_text: str, model_name: str = "llama3.2:3b"):
    """
    SSE generator — streams tokens then sends a final sources event.
    Compatible with EventSource on the frontend.
    Supports Gemini (Google), Grok (xAI), and local Ollama models.
    """
    prompt, results = _build_prompt(query_text)
    model = _get_model(model_name)

    # Stream tokens
    if _is_chat_model(model_name):
        for chunk in model.stream(prompt):
            token = _extract_text(chunk.content)
            if token:
                yield f"data: {json.dumps({'token': token})}\n\n"
    else:
        for chunk in model.stream(prompt):
            token = _extract_text(chunk)
            if token:
                yield f"data: {json.dumps({'token': token})}\n\n"

    # Send sources after streaming completes
    sources = [
        {
            "id": doc.metadata.get("id"),
            "content": doc.page_content,
            "score": float(_score)
        }
        for doc, _score in results
    ]
    yield f"event: sources\ndata: {json.dumps({'sources': sources})}\n\n"
    yield "event: done\ndata: {}\n\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="Model name (e.g. gemini-2.0-flash, grok-3-fast, mistral)")
    args = parser.parse_args()
    query_rag(args.query_text, args.model)


if __name__ == "__main__":
    main()
