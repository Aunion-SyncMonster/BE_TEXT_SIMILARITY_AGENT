from pathlib import Path

from app.schema.text_similarity_dto import TextSimilarityRequest
from dotenv import load_dotenv
import os
import requests

from app.util.exception import TranslationError

env_path = (Path(__file__).resolve().parents[2] / "config" / ".env")
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("GOOGLE_TRANSLATOR_API_KEY")


def translate_google(request: TextSimilarityRequest) -> str:
    """Google Translation API를 호출해 번역된 텍스트를 반환합니다.
    실패 시 TranslationError를 발생시킵니다.
    """
    url = "https://translation.googleapis.com/language/translate/v2"
    params = {
        "q": request.input_text,
        "target": request.output_language,
        "key": API_KEY
    }
    response = requests.post(url, params=params)

    if not response.ok:
        raise TranslationError(
            f"Google Translator API error ({response.status_code}): {response.text}"
        )

    data = response.json()
    translated_text = data["data"]["translations"][0]["translatedText"]
    return translated_text