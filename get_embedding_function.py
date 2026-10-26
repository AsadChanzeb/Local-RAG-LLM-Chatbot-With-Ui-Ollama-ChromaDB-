import os
from langchain_ollama import OllamaEmbeddings


def get_embedding_function():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=base_url,
    )
    return embeddings
