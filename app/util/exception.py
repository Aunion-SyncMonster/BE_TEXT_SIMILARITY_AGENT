class S3UploadError(Exception):
    """S3 업로드 실패 시 던지는 예외"""
    pass

class TranslationError(Exception):
    """번역 API 호출 실패 시 던지는 예외"""
    pass