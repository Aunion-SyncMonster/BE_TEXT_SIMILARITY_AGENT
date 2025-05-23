import logging
import time
import torch
from sentence_transformers import util
from app.web_socket.notifier import notify_progress
import app.core.models as models


threshold_e5=0.8
threshold_labse=0.7
threshold_bert=0.8
threshold_comet=0.5


def _encode_with_model(model, inputs):
    """
    주어진 모델로 inputs 리스트를 인코딩하고, 결과를 torch.Tensor로 반환합니다.
    """
    emb = model.encode(inputs)
    return emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)


def _compute_e5(original: str, translated: str):
    """
    E5 모델로 의미 유사도 점수 계산
    """
    emb_tensor = _encode_with_model(
        models.model_e5,
        [f"query: {original}", f"passage: {translated}"]
    )
    return util.pytorch_cos_sim(emb_tensor[0], emb_tensor[1]).item()


def _compute_labse(original: str, translated: str):
    """
    LaBSE 모델로 직역 유사도 점수 계산
    """
    emb2_tensor = _encode_with_model(
        models.model_labse,
        [original, translated]
    )
    return util.pytorch_cos_sim(emb2_tensor[0], emb2_tensor[1]).item()


def _compute_bertscore(original: str, translated: str):
    """
    BERTScore 모델로 precision, recall, f1 점수 계산
    """
    p, r, f1 = models.bert_scorer.score([original], [translated])
    return p.item(), r.item(), f1.item()


def _compute_comet(original:str, translated: str):
    """
    comet 모델로 comet score 계산
    """
    data = [
        {
            "src": original,
            "mt": translated,
        }
    ]
    logging.info("data: {}".format(data))
    model_output = models.model_comet.predict(data, batch_size=8, gpus=0)
    return model_output.system_score


async def evaluate_dual_similarity(
    task_name: str,
    original: str,
    translated: str,
    threshold_e5_good: float = 0.8,
    threshold_labse_good: float = 0.7
) -> dict:
    """
    세 단계(E5, LaBSE, BERTScore)를 순차적으로 실행하며,
    진행률을 WebSocket으로 전송하고 최종 유사도 결과를 반환합니다.
    """
    start = time.time()
    await notify_progress(task_name, 0)
    logging.info(f"📝 Original: {original}")
    logging.info(f"🈶 Translated: {translated}")

    # 단계별 계산 및 진행률 전송
    step_funcs = [
        ("E5", _compute_e5),
        ("LaBSE", _compute_labse),
        ("BERTScore", _compute_bertscore),
        ("comet", _compute_comet)
    ]
    total_steps = len(step_funcs)
    sim_e5 = sim_labse = None
    p = r = f1 = None
    comet_score = None

    for idx, (name, func) in enumerate(step_funcs, start=1):
        # 함수 호출
        result = func(original, translated)
        # 결과 저장
        if name == "E5":
            sim_e5 = result
        elif name == "LaBSE":
            sim_labse = result
        elif name == "BERTScore":
            p, r, f1 = result
        else:
            comet_score = result

        # 진행률 계산 및 알림
        progress = int(idx / total_steps * 100)
        await notify_progress(task_name, progress)
        logging.info(f"Step '{name}' completed, progress={progress}%")

    # 종합 판단
    if sim_e5 is None or sim_labse is None or p is None:
        err = "Similarity computation failed"
        logging.error(f"❌ {err} for task {task_name}")
        await notify_progress(task_name, -1, error=err)
        return {}

    descriptions = []

    if sim_e5 > threshold_e5_good and sim_labse > threshold_labse_good:
        descriptions.append(
            f"✅ 직역 가능성 높음 (E5: {sim_e5:.2f} ≥ {threshold_e5}, "
            f"LaBSE: {sim_labse:.2f} ≥ {threshold_labse})"
        )
    elif sim_e5 > threshold_e5_good:
        f"✏️ 의역 가능성 있음 (E5: {sim_e5:.2f} ≥ {threshold_e5}, "
        f"LaBSE: {sim_labse:.2f} < {threshold_labse})"

    elif sim_labse > threshold_labse_good:
        descriptions.append(
            f"📖 직역 유사성만 높음 (E5: {sim_e5:.2f} < {threshold_e5}, "
            f"LaBSE: {sim_labse:.2f} ≥ {threshold_labse})"
        )
    else:
        descriptions.append(
            f"⚠️ 의미 차이 큼 (E5: {sim_e5:.2f}, LaBSE: {sim_labse:.2f}); "
            "COMET score 확인 요망"
        )

    # BERTScore 평가
    if f1 >= threshold_bert:
        descriptions.append(
            f"👍 단어 단위 의미 유사도 우수 (BERTScore F1: {f1:.2f} ≥ {threshold_bert})"
        )
    else:
        descriptions.append(
            f"👎 단어 단위 의미 유사도 부족 (BERTScore F1: {f1:.2f} < {threshold_bert})"
        )

    # COMET 평가 추가
    if comet_score >= threshold_comet:
        descriptions.append(
            f"🎯 번역 품질 우수 (COMET: {comet_score:.2f} ≥ {threshold_comet})"
        )
    else:
        descriptions.append(
            f"❗️ 번역 품질 미흡 (COMET: {comet_score:.2f} < {threshold_comet})"
        )

    execution_time = time.time() - start
    logging.info(f"E5 의미 유사도: {sim_e5:.4f}")
    logging.info(f"LaBSE 직역 유사도: {sim_labse:.4f}")
    logging.info(f"BERTScore - P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")
    logging.info(f"comet score: {comet_score:.4f}")
    logging.info(f"⏱ 실행 시간: {execution_time:.2f}s | ✅ completed similarity for task {task_name}")

    return {
        "original_text": original,
        "translated_text": translated,
        "e5_semantic_similarity": round(sim_e5, 4),
        "labse_literal_similarity": round(sim_labse, 4),
        "bertscore": {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)},
        "comet_score": comet_score,
        "description": "\n".join(descriptions) + "\n",
        "execution_time": round(execution_time, 2)
    }
