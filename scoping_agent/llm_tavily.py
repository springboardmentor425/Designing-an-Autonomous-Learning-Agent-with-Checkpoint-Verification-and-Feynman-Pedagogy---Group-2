import os
from typing import Optional
from google import genai
from tavily import TavilyClient

# -----------------------------
# Gemini Wrapper (TEXT ONLY)
# -----------------------------
class GeminiWrapper:
    def __init__(
        self,
        model: str = "models/gemini-2.5-flash",
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str = "",
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """
        Gemini only supports roles: user, model.
        So we merge system + user into ONE user message.
        """

        full_prompt = system_prompt
        if user_prompt:
            full_prompt += "\n\nUser query:\n" + user_prompt

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": full_prompt}],
                }
            ],
        )

        return response.text.strip()


# -----------------------------
# Tavily (MINIMAL usage)
# -----------------------------
def tavily_search_minimal(query: str, max_results: int = 3) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return ""

    tavily = TavilyClient(api_key=api_key)
    results = tavily.search(query=query, max_results=max_results)

    snippets = []
    for r in results.get("results", []):
        snippets.append(f"- {r.get('title')}: {r.get('content')}")

    return "\n".join(snippets)
