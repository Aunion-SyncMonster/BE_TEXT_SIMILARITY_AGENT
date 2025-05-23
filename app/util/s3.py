import logging
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

from app.util.exception import S3UploadError

env_path = (Path(__file__).resolve().parents[1] / "config" / ".env")

load_dotenv(dotenv_path=env_path)

AWS_REGION = os.getenv('AWS_DEFAULT_REGION', "ap-northeast-2")
S3_BUCKET = os.getenv('S3_BUCKET')

s3_client = boto3.client(
    "s3",
    region_name="ap-northeast-2"
)


def make_public_url(key: str) -> str:
    # 1) 버킷의 리전 가져오기
    loc = s3_client.get_bucket_location(Bucket=S3_BUCKET)['LocationConstraint']
    # us-east-1 은 URL에 region 부분이 생략됩니다
    if loc is None or loc == 'us-east-1':
        endpoint = f"https://{S3_BUCKET}.s3.amazonaws.com"
    else:
        endpoint = f"https://{S3_BUCKET}.s3.{loc}.amazonaws.com"
    # 2) key 부분 URL 인코딩
    from urllib.parse import quote_plus
    return f"{endpoint}/{quote_plus(key)}"


def upload_s3(
    key: str,
    body: bytes,
    content_type: str,
):
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=body,
            ContentType=content_type,
        )
    except Exception as e:
        logging.error(f"S3 upload error: {e}")
        raise S3UploadError(f"S3 upload error: {e}")