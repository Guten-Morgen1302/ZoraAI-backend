import boto3

s3 = boto3.client(
    "s3",
    aws_access_key_id="YOUR_KEY",
    aws_secret_access_key="YOUR_SECRET",
    region_name="ap-south-1"
)

s3.upload_file(
    "test.mp4",
    "runwayos-dev-bucket",
    "voice/test.mp4"
)

print("Uploaded 🚀")