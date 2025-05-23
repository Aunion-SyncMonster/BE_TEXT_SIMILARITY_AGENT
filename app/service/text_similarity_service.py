import logging

from app.model.translate.gpt import translate_gpt
from app.model.translate.m2m100 import translate_m2m100
from app.schema.text_similarity_dto import TextSimilarityResult, TextSimilarityRequest, TranslateType
from app.model.similarity.evaluate_similarity_agent import evaluate_dual_similarity
from app.model.translate.google_translate import translate_google, TranslationError
from app.client.spring_client import send_result_to_be
from app.util.s3 import upload_s3, make_public_url


def _perform_translation(request: TextSimilarityRequest) -> str:
    """
    요청에 맞는 번역기를 사용해 텍스트를 번역합니다.
    실패 시 TranslationError를 전파합니다.
    """
    if request.translate_type == TranslateType.GOOGLE:
        return translate_google(request)
    elif request.translate_type == TranslateType.M2M:
        return translate_m2m100(request)
    elif request.translate_type == TranslateType.GPT:
        return translate_gpt(request)
    else:
        raise TranslationError(f"Unsupported translate_type: {request.translate_type}")


def _build_result(
    result_dict: dict,
    request: TextSimilarityRequest,
    task_name: str
) -> TextSimilarityResult:
    """
    evaluate_dual_similarity 결과 dict 으로 TextSimilarityResult DTO 생성
    및 전체 점수(산술 평균) 계산
    """
    e5 = result_dict.get("e5_semantic_similarity", 0)
    labse = result_dict.get("labse_literal_similarity", 0)
    bs_dict = result_dict.get("bertscore") or {}
    bertscore = float(bs_dict.get("f1") or 0)
    comet_score = result_dict.get("comet_score") or 0
    overall_score_float = (e5 + labse + bertscore + comet_score) / 4
    overall_score = round(overall_score_float * 100)

    return TextSimilarityResult(
        total_project_id=request.total_project_id,
        score=overall_score,
        input_text=request.input_text,
        translation_text=result_dict.get("translated_text"),
        input_text_key=make_public_url(request.input_text_key),
        translation_text_key=make_public_url(request.output_text_key),
        translation_api_type=request.translate_type,
        inference_time=result_dict.get("execution_time"),
        status="SUCCESS",
        input_language=request.input_language.name,
        output_language=request.output_language.name,
        task_name=task_name,
        description=result_dict.get("description"),
        e5=e5,
        labse=labse,
        bertscore=bertscore,
        comet_score=comet_score
    )


async def run_text_similarity(
    task_name: str,
    request: TextSimilarityRequest
):
    logging.info(f"🔄 starting text-similarity task: {task_name}")

    # 1) 번역
    try:
        target_text = request.output_text or _perform_translation(request)

        output_txt_key = f"text_similarity/{task_name}/{request.input_text_key.split('/')[2]}.txt"
        upload_s3(output_txt_key, target_text.encode("utf-8"), "text/plain; charset=utf-8")

        request.output_text_key = output_txt_key

    except TranslationError as e:
        logging.error(f"❌ translation failed for {task_name}: {e}")
        from app.web_socket.notifier import notify_progress
        await notify_progress(task_name, -1, error=str(e))
        logging.info("⏹ run_text_similarity exited after translation error")

        empty_result = {
            "original_text": request.input_text,
            "translated_text": request.output_text or "",
            "e5_semantic_similarity": 0,
            "labse_literal_similarity": 0,
            "bertscore": 0,
            "comet_score": 0,
            "execution_time": 0,
            "description": ""
        }
        dto = _build_result(empty_result, request, task_name)
        return send_result_to_be(dto)

    # 2) 유사도 평가
    try:
        result_dict = await evaluate_dual_similarity(
            task_name=task_name,
            original=request.input_text,
            translated=target_text
        )
    except Exception as e:
        logging.error(f"❌ similarity evaluation failed for {task_name}: {e}")
        from app.web_socket.notifier import notify_progress
        await notify_progress(task_name, -1, error="Similarity evaluation error")
        logging.info("⏹ run_text_similarity exited after similarity error")

        error_result = {
            "original_text": request.input_text,
            "translated_text": target_text,
            "e5_semantic_similarity": 0,
            "labse_literal_similarity": 0,
            "bertscore": 0,
            "comet_score": 0,
            "execution_time": 0,
            "description": ""
        }
        dto = _build_result(error_result, request, task_name)
        return send_result_to_be(dto)

    # 3) 결과 전송
    logging.info(f"✅ completed text-similarity task: {task_name}")
    dto = _build_result(result_dict, request, task_name)
    return send_result_to_be(dto)