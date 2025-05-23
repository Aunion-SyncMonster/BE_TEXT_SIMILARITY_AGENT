import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from app.schema.text_similarity_dto import TextSimilarityResult

env_path = (Path(__file__).resolve().parents[1] / "config" / ".env")
load_dotenv(dotenv_path=env_path)
TEXT_SIMILARITY_BE_URL = os.getenv("TEXT_SIMILARITY_BE_URL")
RESULT_URL = f"{TEXT_SIMILARITY_BE_URL}/api/text-similarities"


def send_result_to_be(result: TextSimilarityResult):
    try:

        logging.info(f"result: {result}")

        payload = result.model_dump()
        resp = requests.post(RESULT_URL, json=payload, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        logging.info(f"Failed to send result to Spring: {e}")