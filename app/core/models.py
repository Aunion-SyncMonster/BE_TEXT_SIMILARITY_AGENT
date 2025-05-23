import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from sentence_transformers import SentenceTransformer
from bert_score import BERTScorer
from comet import download_model, load_from_checkpoint

env_path = (Path(__file__).resolve().parents[1] / "config" / ".env")

load_dotenv(dotenv_path=env_path)

COMET_MODEL_REPO = os.getenv("COMET_MODEL_REPO")

device = "cpu"

model_e5 = None
model_labse = None
bert_scorer = None
model_comet = None

models = {}
tokenizers = {}

SUPPORTED_PAIRS = [
    ("en", "ko"),
    ("en", "ja"),
    ("en", "hi"),
    ("ko", "en"),
    # ("en", "de"),
    # ("en", "fr"),
    # ("en", "es")
]

def init_models():
    global model_e5, model_labse, bert_scorer, model_comet, models, tokenizers
    logging.info("🔄 Loading models at startup…")

    model_e5    = SentenceTransformer('intfloat/multilingual-e5-large', device='cpu')
    model_labse = SentenceTransformer('sentence-transformers/LaBSE',      device='cpu')
    bert_scorer = BERTScorer(
        model_type="xlm-roberta-base",
        lang="ko",
        rescale_with_baseline=False,
        idf=False
    )

    model_comet_path = download_model("wmt20-comet-qe-da")
    model_comet = load_from_checkpoint(model_comet_path)

    for src, tgt in SUPPORTED_PAIRS:
        checkpoint = f"{COMET_MODEL_REPO}/m2m100_{src}-{tgt}"

        # ① 토크나이저 로드
        tokenizer = M2M100Tokenizer.from_pretrained(checkpoint)
        # ② 모델 로드
        model = M2M100ForConditionalGeneration.from_pretrained(checkpoint).to(device)
        tokenizers[(src, tgt)] = tokenizer
        models[(src, tgt)] = model

    logging.info("✅ Models loaded successfully")
