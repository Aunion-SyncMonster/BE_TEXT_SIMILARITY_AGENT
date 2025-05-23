import uuid

def generate_task_name() -> str:
    """
    UUID 기반 고유 task_name 생성
    """
    return "text_similarity_"+uuid.uuid4().hex