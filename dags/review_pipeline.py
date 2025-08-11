from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os



sys.path.append(os.path.join(os.path.dirname(__file__), '../Pipeline'))

# print(os.path.join(os.path.dirname(__file__), '../Pipeline'))

from stages import extract_and_clean, embed_reviews, cluster, summarize

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="reviews_pipeline",
    default_args=default_args,
    description="Pipeline orchestrating review data processing daily",
    start_date=datetime(2025, 8, 10),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    task_extract_transform = PythonOperator(
        task_id="extract_and_transform",
        python_callable=extract_and_clean.main, 
    )

    task_embed = PythonOperator(
        task_id="embed_reviews",
        python_callable=embed_reviews.main,
    )

    task_cluster = PythonOperator(
        task_id="cluster_reviews",
        python_callable=cluster.main,
    )

    task_summarize = PythonOperator(
        task_id="summarize_reviews",
        python_callable=summarize.main,
    )

    # execution order
    task_extract_transform >> task_embed >> task_cluster >> task_summarize
