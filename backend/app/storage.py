import boto3
from botocore.config import Config
from .config import settings

def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

def generate_signed_url(key: str, expires_in: int = 900) -> str:
    client = get_r2_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.r2_bucket_name, "Key": key},
        ExpiresIn=expires_in,
    )

def upload_file(key: str, data: bytes, content_type: str = "audio/wav") -> None:
    client = get_r2_client()
    client.put_object(
        Bucket=settings.r2_bucket_name,
        Key=key,
        Body=data,
        ContentType=content_type,
    )

def delete_file(key: str) -> None:
    client = get_r2_client()
    client.delete_object(Bucket=settings.r2_bucket_name, Key=key)

def delete_files(keys: list[str]) -> None:
    if not keys:
        return
    client = get_r2_client()
    client.delete_objects(
        Bucket=settings.r2_bucket_name,
        Delete={"Objects": [{"Key": k} for k in keys]},
    )
