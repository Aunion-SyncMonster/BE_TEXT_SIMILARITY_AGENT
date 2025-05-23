import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form
from app.schema.text_similarity_dto import TextSimilarityRequest, TextSimilarityResponse, TranslateType, Language, \
    RetranslateRequest
from app.service.text_similarity_service import run_text_similarity
from app.util.s3 import upload_s3
from app.util.task_utils import generate_task_name

router = APIRouter()

@router.post("/text-similarities", response_model=TextSimilarityResponse)
async def submit_translation(
    background_tasks: BackgroundTasks,
    input_file: UploadFile = File(..., description="입력 텍스트(.txt)"),
    output_file: Optional[UploadFile] = File(None, description="비교용 출력 텍스트(.txt)"),
    input_language: Language = Form(...),
    output_language: Language = Form(...),
    translate_type: TranslateType = Form(...),
    total_project_id: int = Form(...)
):
    """
    .txt 파일로부터 input_text/output_text를 읽어서 큐에 등록하고 task_name 반환
    """

    task_name = generate_task_name()
    logging.info(f"task_name: {task_name}")

    try:
        raw_input = await input_file.read()

        input_txt_key = f"text_similarity/{task_name}/{input_file.filename}"
        upload_s3(input_txt_key, raw_input, input_file.content_type)

        input_text = raw_input.decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"input_file 읽기 실패: {e}")

    output_text: Optional[str] = None
    output_txt_key: Optional[str] = None
    if output_file is not None:
        try:
            raw_output = await output_file.read()

            output_txt_key = f"text_similarity/{task_name}/{output_file.filename}"
            upload_s3(output_txt_key, raw_output, output_file.content_type)

            output_text = raw_output.decode("utf-8")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"output_file 읽기 실패: {e}")

    request_dto = TextSimilarityRequest(
        input_text=input_text,
        output_text=output_text,
        input_language=input_language,
        output_language=output_language,
        translate_type=translate_type,
        total_project_id=total_project_id,
        input_text_key=input_txt_key,
        output_text_key=output_txt_key,
    )

    try:
        background_tasks.add_task(run_text_similarity, task_name, request_dto)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue translation: {e}")

    return TextSimilarityResponse(task_name=task_name, status="processing")

@router.post("/text-similarities/retranslate", response_model=TextSimilarityResponse)
async def submit_retranslation(
    background_tasks: BackgroundTasks,
    request: RetranslateRequest,
):
    task_name = generate_task_name()
    logging.info(f"retranslate task_name: {task_name}")

    try:
        raw_input = request.input_text

        input_txt_key = f"text_similarity_re/{task_name}/input_text.txt"
        upload_s3(input_txt_key, raw_input.encode("utf-8"), 'text/plain')

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"input_text 파일 쓰기 실패: {e}")

    request_dto = TextSimilarityRequest(
        input_text=request.input_text,
        input_language=request.input_language,
        output_language=request.output_language,
        translate_type=TranslateType.GPT,
        total_project_id=request.total_project_id,
        input_text_key=input_txt_key,
    )

    try:
        background_tasks.add_task(run_text_similarity, task_name, request_dto)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue translation: {e}")

    return TextSimilarityResponse(task_name=task_name, status="processing")





