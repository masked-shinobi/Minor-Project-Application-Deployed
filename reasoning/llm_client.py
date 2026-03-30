"""
LLM Client Module
Wrapper around the Groq API for LLM interactions.
Supports LLaMA 3 / LLaMA 3.1 models via Groq's fast inference.
"""

import os
from typing import Optional, List, Dict

try:
    from groq import Groq
except ImportError:
    raise ImportError("groq is required. Install via: pip install groq")

from dotenv import load_dotenv


class LLMClient:
    """
    Client for interacting with the Groq LLM API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: Groq API key. If None, reads from GROQ_API_KEY env var.
            model: Model name. If None, reads from GROQ_MODEL env var 
                   or defaults to 'llama-3.1-8b-instant'.
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.client = Groq(api_key=self.api_key)

        print(f"[LLMClient] Initialized with model: {self.model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.3,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system-level instruction.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature (0 = deterministic, 1 = creative).
            top_p: Nucleus sampling parameter.

        Returns:
            The generated text response.
        """
        # Safety: Global context trimming for restricted TPM limits (6000 tokens).
        # We aim for ~1800 tokens of input (approx 6,000 chars) to leave 
        # plenty of room for system prompt and output tokens (max_tokens).
        # Many tiers count Input + Max_Tokens against TPM.
        max_prompt_chars = 6000
        if len(prompt) > max_prompt_chars:
            print(f"[LLMClient] Warning: Prompt length ({len(prompt)}) exceeds restrictive safety limit. Truncating.")
            # Keep end of prompt (the question/instruction) and truncate middle logic if possible,
            # but for simplicity we truncate the whole string to the limit.
            prompt = prompt[:max_prompt_chars] + "\n\n[Prompt truncated to stay within API token limits...]"

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLMClient] Error: {e}")
            raise

    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3
    ) -> str:
        """
        Generate a response using retrieved context (RAG pattern).

        Args:
            query: The user's question.
            context: Retrieved context to ground the response.
            system_prompt: Optional system instruction.
            max_tokens: Maximum tokens.
            temperature: Sampling temperature.

        Returns:
            The generated response.
        """
        if system_prompt is None:
            system_prompt = (
                "You are an expert academic research paper analyzer. "
                "Answer questions accurately based on the provided context. "
                "If the context doesn't contain enough information, say so. "
                "Always cite the relevant section when possible."
            )

        # Safety: Trim context to ~1800 tokens (approx 5,000 characters) 
        # to ensure the full request (input + response budget) fits 
        # within Groq's restricted 6,000 token TPM limit.
        max_context_chars = 5000
        if len(context) > max_context_chars:
            print(f"[LLMClient] Warning: Context length ({len(context)}) exceeds safety limit. Truncating.")
            context = context[:max_context_chars] + "\n\n[Context truncated for token limits...]"

        prompt = (
            f"Context from research papers:\n"
            f"---\n{context}\n---\n\n"
            f"Question: {query}\n\n"
            f"Please provide a comprehensive answer based on the context above."
        )

        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.3
    ) -> str:
        """
        Generate a response from a multi-turn conversation.

        Args:
            messages: List of {"role": "...", "content": "..."} message dicts.
            max_tokens: Maximum tokens.
            temperature: Sampling temperature.

        Returns:
            The generated response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLMClient] Chat error: {e}")
            raise
