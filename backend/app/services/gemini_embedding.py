import logging
import asyncio
import google.generativeai as genai
from app.config import settings

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=settings.gemini_api_key)

def get_embedding(text: str) -> list[float]:
    """Get embedding from Gemini."""
    try:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        return response['embedding']
    except Exception as e:
        logger.error(f"Error generating embedding with Gemini: {e}")
        # Return an empty vector or raise. Chroma expects a list of floats.
        raise e

async def get_embedding_async(text: str) -> list[float]:
    """Async wrapper for get_embedding"""
    return await asyncio.to_thread(get_embedding, text)

async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Batch generation of embeddings"""
    try:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=texts
        )
        return response['embedding']
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        raise e
