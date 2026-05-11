import asyncio
import logging
from typing import Dict, Any, List, Optional
from groq import AsyncGroq
from app.config import settings

logger = logging.getLogger(__name__)

# Initialize Groq Client
client = AsyncGroq(api_key=settings.groq_api_key)

async def generate_response(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    model: str = "llama-3.3-70b-versatile"
) -> str:
    """
    Generate response using Groq API.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=settings.llm_timeout,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq API error with model {model}: {e}")
        # Fallback to secondary model
        fallback_model = "mixtral-8x7b-32768"
        if model != fallback_model:
            try:
                logger.info(f"Falling back to {fallback_model}")
                response = await client.chat.completions.create(
                    model=fallback_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=settings.llm_timeout,
                )
                return response.choices[0].message.content
            except Exception as e2:
                logger.error(f"Groq fallback error: {e2}")
        raise e
