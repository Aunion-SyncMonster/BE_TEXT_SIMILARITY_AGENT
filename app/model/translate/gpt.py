import logging
from pathlib import Path

from app.schema.text_similarity_dto import TextSimilarityRequest
from dotenv import load_dotenv
import os
from openai import OpenAI

env_path = (Path(__file__).resolve().parents[2] / "config" / ".env")
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("GPT_API_KEY")

client = OpenAI(api_key=API_KEY)


def translate_gpt(request: TextSimilarityRequest) -> str:
    system_prompt = """
    You are a cinematic translator for movie dialogue.
    When given user input containing:
      - src_text: the original text in the source language
      - src_lang: source language code (e.g. "en", "ko")
      - tar_lang: target language code (e.g. "en", "ko")

    You must:
      1. Translate src_text from src_lang into tar_lang using adaptive and creative (transcendent) translation, not word-for-word.
      2. Preserve a dramatic, cinematic tone—as if delivering a powerful line on screen.
      3. If src_text is an idiom or fixed expression (like “Here’s looking at you, kid.”), render its well-known idiomatic equivalent in the target language (e.g. “당신의 눈동자에 건배”).
      4. Ensure the result feels like a natural, emotionally impactful movie dialogue.
      5. Output only the final translated line as a single plain string (no JSON or extra commentary).
    """

    completion = client.chat.completions.create(
        model="gpt-4.1-nano",
        store=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":
                f"src_text: {request.input_text}\n"
                f"src_lang: {request.input_language}\n"
                f"tar_lang: {request.output_language}"
             }
        ]
    )

    logging.info(f"retranslation result: {completion}")
    return completion.choices[0].message.content.strip()
