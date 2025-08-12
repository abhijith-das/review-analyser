import json
import boto3

def get_aws_s3_client():
    '''
    store AWS credentials in a JSON file at the specified path.
    {
    "aws": {
        "aws_access_key_id": "<your_access_key_id>",
        "aws_secret_access_key": "<your_secret_access_key>",
        "region": "<your_region>"
        }
    }'''
    # Read AWS credentials from JSON file
    with open('/home/abhi/airflow/Pipeline/utils/credentials.json', "r") as f:
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