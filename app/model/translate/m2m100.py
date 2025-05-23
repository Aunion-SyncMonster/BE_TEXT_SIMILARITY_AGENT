from app.core.models import SUPPORTED_PAIRS, tokenizers, models, device
from app.schema.text_similarity_dto import TextSimilarityRequest
from app.util.exception import TranslationError


def translate_m2m100(request: TextSimilarityRequest) -> str:
    pair = (request.input_language, request.output_language)
    if pair not in SUPPORTED_PAIRS:
        raise TranslationError(400, f"지원하지 않는 언어쌍: {pair}")
    tokenizer = tokenizers[pair]
    model = models[pair]

    # 언어 설정
    tokenizer.src_lang = request.input_language
    # 번역 언어 강제 지정
    forced_bos = tokenizer.get_lang_id(request.output_language)

    # 토크나이즈 + forced BOS
    inputs = tokenizer(request.input_text,
                       return_tensors="pt",
                       padding=True, truncation=True, max_length=64
                       ).to(device)
    inputs["forced_bos_token_id"] = forced_bos

    # 생성
    model.eval()

    # 토큰화된 아웃풋
    output = model.generate(**inputs, num_beams=4, max_length=64, early_stopping=True)

    # 토큰 -> 텍스트 복호화
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]