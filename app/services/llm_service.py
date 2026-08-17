import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


class LLMService:
    """
    Gemini service used ONLY when reasoning/summarization/comparison is required.
    Never call this for simple factual retrieval.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            self._client = None
        else:
            self._client = genai.Client(api_key=api_key)

        preferred = os.getenv("GEMINI_MODEL")
        self._models = [m for m in [preferred, "gemini-3.7-flash", "gemini-3.5-flash", "gemini-flash-latest"] if m]

    def generate_answer(
        self,
        question: str,
        context: str,
    ) -> str:
        if not self._client:
            return "I couldn't find relevant information in the uploaded documents."

        prompt = f"""You are a Document Intelligence Assistant.

IMPORTANT RULES:
1. Answer ONLY from the supplied CONTEXT below.
2. Never use outside knowledge or general world knowledge.
3. If the answer cannot be found directly in the context, reply exactly:
I couldn't find relevant information in the uploaded documents.
4. Return clean, formatted Markdown.
5. Never hallucinate or infer unsupported facts.
6. Do not include database IDs, point IDs, or internal hashes.

--------------------
CONTEXT:
{context}
--------------------
QUESTION:
{question}
--------------------
ANSWER:"""

        last_error = None
        for model_name in self._models:
            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                last_error = e
                continue

        if last_error:
            print(f"Gemini generation error: {last_error}")
        return "I couldn't find relevant information in the uploaded documents."