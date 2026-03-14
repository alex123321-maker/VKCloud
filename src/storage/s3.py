import os
from pathlib import Path

import boto3


def s3_enabled() -> bool:
    return all(
        [
            os.getenv("S3_ACCESS_KEY"),
            os.getenv("S3_SECRET_KEY"),
            os.getenv("S3_BUCKET"),
        ]
    )


def create_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv(
            "S3_ENDPOINT_URL", "https://hb.ru-msk.vkcloud-storage.ru"
        ),
        region_name=os.getenv("S3_REGION", "ru-msk"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
    )


def upload_directory(local_dir: str, bucket: str, prefix: str) -> None:
    client = create_s3_client()
    base_path = Path(local_dir)

    for file_path in base_path.rglob("*"):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(base_path).as_posix()
        object_key = f"{prefix.rstrip('/')}/{relative_path}"
        client.upload_file(str(file_path), bucket, object_key)


def download_prefix(bucket: str, prefix: str, local_dir: str) -> None:
    client = create_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            relative_path = key[len(prefix.rstrip("/") + "/") :]
            if not relative_path:
                continue

            target_path = Path(local_dir) / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target_path))
