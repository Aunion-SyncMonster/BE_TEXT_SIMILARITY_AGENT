from enum import Enum
from typing import Optional

from pydantic import BaseModel

class Language(str, Enum):
    KOREAN = "ko"
    ENGLISH = "en"
    JAPANESE = "ja"
    HINDI = "hi"

class TranslateType(str, Enum):
    GOOGLE = "GOOGLE"
    M2M = "M2M"
    GPT = "GPT"

class TextSimilarityRequest(BaseModel):
    input_text: str
    output_text: Optional[str] = None
    input_language: Language
    output_language: Language
    translate_type: TranslateType
    input_text_key: str
    output_text_key: Optional[str] = None
    total_project_id: int

class RetranslateRequest(BaseModel):
    input_text: str
    input_language: Language
    output_language: Language
    total_project_id: int

class TextSimilarityResult(BaseModel):
    total_project_id: int
    score: int
    input_text: str
    translation_text: str
    input_text_key: str
    translation_text_key: str
    translation_api_type: str
    inference_time: float
    status: str
    input_language: str
    output_language: str
    task_name: str
    description: str
    e5: float
    labse: float
    bertscore: float
    comet_score: float

class TextSimilarityResponse(BaseModel):
    task_name: str
    status: str