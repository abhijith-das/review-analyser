import json
import boto3

def get_aws_s3_client():
    # Read AWS credentials from JSON file
    with open('/home/abhi/airflow/Pipeline/stages/credentials.json', "r") as f:
        creds = json.load(f)
    aws = creds.get("aws", {})

    # Create and return boto3 client with credentials and region
    client = boto3.client(
        "s3",
        aws_access_key_id=aws.get("aws_access_key_id"),
        aws_secret_access_key=aws.get("aws_secret_access_key"),
        region_name=aws.get("region")
    )
    return client