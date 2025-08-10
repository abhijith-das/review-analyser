import pandas as pd
import boto3
import io
import os
from utils.read_config import get_preprocess_config, get_source_file_cols
from utils.aws_connector import get_aws_s3_client

# function to read the reviews stored in a CSV file in an S3 bucket,
def load_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = get_aws_s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))

# function to preprocess the reviews by removing NaN values, empty strings, and duplicates
def preprocess_reviews(df: pd.DataFrame, column: str = "text") -> pd.DataFrame:
    df = df.dropna(subset=[column])
    df = df[df[column] != ""]
    # df[column] = df[column].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    df = df.drop_duplicates(subset=[column])
    return df.reset_index(drop=True)

# main function which loads the reviews from S3, preprocesses them, and saves the cleaned data to a specified output path
def preprocess_s3_csv(bucket: str, key: str, column: str, output_path: str):
    df = load_csv_from_s3(bucket, key)
    print(f"Loaded {len(df)} rows from S3")
    df_clean = preprocess_reviews(df, column)
    print(f"After cleaning: {len(df_clean)} rows")
    print(df_clean)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_parquet(output_path, index=False)
    print(f"Saved cleaned reviews to {output_path}")

# main function for the airflow DAG
def main():
    configs = get_preprocess_config()
    cols = get_source_file_cols()
    preprocess_s3_csv(
        bucket=configs["source"]["s3_bucket"],
        key=configs["source"]["path"] + "/" + configs["source"]["file_name"],
        column=cols["TEXT"],
        output_path= configs["target"]["parquet_file"]
    )