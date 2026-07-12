import boto3, hashlib, os
from dotenv import load_dotenv
load_dotenv()

r2 = boto3.client(
    "s3",
    endpoint_url="https://{}.r2.cloudflarestorage.com".format(os.environ["R2_ACCOUNT_ID"]),
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
)
bucket = os.environ["R2_BUCKET_NAME"]
pid = "67192f5c-2d12-4c42-bd3d-002621194aab"

for n in [1, 2, 3]:
    key = "projects/{}/beat_attempt_{}.wav".format(pid, n)
    try:
        obj  = r2.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        sha  = hashlib.sha256(data).hexdigest()
        print("Beat {}".format(n))
        print("  db_key: {}".format(key))
        print("  size:   {:,} bytes ({:.1f} KB)".format(len(data), len(data)/1024))
        print("  sha256: {}".format(sha))
        print()
    except Exception as e:
        print("Beat {}: MISSING — {}".format(n, e))
