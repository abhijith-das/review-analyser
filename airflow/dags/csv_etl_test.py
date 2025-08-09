from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

dag = DAG(
    dag_id="csv_etl_test",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["etl", "csv"],
)

BASE_PATH = "/opt/airflow/data"  # Inside container mount

def extract():
    df = pd.read_csv(f"{BASE_PATH}/input.csv")
    print("✅ Extracted data:\n", df.head())

def transform():
    df = pd.read_csv(f"{BASE_PATH}/input.csv")
    df = df.dropna()  # Drop rows with any null
    df["score"] = df["score"].astype(int)
    df.to_csv(f"{BASE_PATH}/output.csv", index=False)
    print("✅ Transformed data:\n", df.head())

def load():
    df = pd.read_csv(f"{BASE_PATH}/output.csv")
    print("📤 Final output:\n", df.to_string(index=False))

extract_task = PythonOperator(task_id="extract", python_callable=extract, dag=dag)
transform_task = PythonOperator(task_id="transform", python_callable=transform, dag=dag)
load_task = PythonOperator(task_id="load", python_callable=load, dag=dag)

extract_task >> transform_task >> load_task
