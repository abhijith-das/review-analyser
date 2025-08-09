import json
import random
import time
import re
import numpy as np
import pandas as pd
import nltk
from datetime import datetime
from nltk.data import find
from nltk.sentiment import SentimentIntensityAnalyzer
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.db_operations import retrieve_table_as_df, retrieve_values_closest_to_centroid, create_table_from_df



# function to retrieve all reviews from the database
def get_reviews_from_db(db_name: str, schema_name: str, table_name: str) -> pd.DataFrame:
    df = retrieve_table_as_df(db_name, schema_name, table_name)
    if df.empty:
        raise ValueError(f"No reviews found in table '{table_name}' in database '{db_name}' and schema '{schema_name}'.")
    # df['is_close_to_centroid'] = df['is_close_to_centroid'].astype(bool)
    return df

# function to retrieve reviews closest to the centroid from the database
def get_reviews_closest_to_centroid(db_name: str, schema_name: str, table_name: str) -> pd.DataFrame:
    df = retrieve_values_closest_to_centroid(db_name, schema_name, table_name)
    if df.empty:
        raise ValueError(f"No reviews found in table '{table_name}' in database '{db_name}' and schema '{schema_name}' where 'is_close_to_centroid' is TRUE.")  
    return df


# function to create a Dataframe of reviews closest to the centroid and 20 random reviews in the same cluster
def get_reviews_for_llm(df_closest: pd.DataFrame, df_all: pd.DataFrame) -> pd.DataFrame:

    if df_closest.empty or df_all.empty:
        raise ValueError("DataFrames for closest reviews or all reviews are empty.")
    print(df_all)
    unique_labels = df_closest['cluster_id'].unique()
    reviews_for_cluster = []
    for label in unique_labels: 
        # Get 20 random reviews from the same cluster with flag is_close_to_centroid as False
        random_reviews = df_all[(df_all['cluster_id'] == label) & (df_all['is_close_to_centroid'] == 'false')].sample(n=20, random_state=42)

        reviews_for_cluster.extend(random_reviews.to_dict('records'))

    df_reviews_for_cluster = pd.DataFrame(reviews_for_cluster)
    return df_reviews_for_cluster

# function to summarize reviews 

def summarize_reviews(cluster_id, cluster_texts, api_key, max_retries=15, base_delay=1, max_delay=10):
    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are given a set of product reviews belonging to the same cluster.

    Cluster ID: {cluster_id}
    Reviews:
    {cluster_texts}

    Task:
    1. Generate a short, catchy title (max 6 words) summarizing the main theme.
    2. Write a concise 2-3 sentence description summarizing the main idea of this cluster.
    Format the response as:
    Title: <title>
    Description: <description>
    """
  
    # response = client.models.generate_content(
    #     model = "gemma-3-27b-it",
    #     contents = prompt,
    #     config = types.GenerateContentConfig(
    #         temperature=0.1,
    #         top_p=0.5,
    #         top_k=20
    #     )
    # )

    # print(response.text)


    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemma-3-27b-it",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    top_p=0.5,
                    top_k=20
                )
            )
            text_output = response.text.strip()
            print(f"Response: {text_output}")
            title_match = re.search(r"Title:\s*(.*)", text_output)
            desc_match = re.search(r"Description:\s*(.*)", text_output, re.DOTALL)

            title = title_match.group(1).strip() if title_match else ""
            description = desc_match.group(1).strip() if desc_match else ""

            return title, description

        except Exception as e:
            wait_time = min(max_delay, base_delay * (2 ** attempt))  
            wait_time += random.uniform(0, 1)  
            print(f"[Attempt {attempt+1}/{max_retries}] Error: {e} — retrying in {wait_time:.2f}s...")
            time.sleep(wait_time)

# function to help threading
def process_cluster(cluster_id, df_reviews_for_llm, api_key):
    max_retries=5
    base_delay=1
    max_delay=15
    cluster_reviews = df_reviews_for_llm[df_reviews_for_llm['cluster_id'] == cluster_id]['review']
    cluster_texts = "\n".join(cluster_reviews.tolist())
    summary = summarize_reviews(cluster_id, cluster_texts, api_key, max_retries, base_delay, max_delay)
    return {
        "cluster_id": cluster_id,
        "summary": summary
    }


def classify_sentiment(score):
    if score >= 0.1:
        return "positive"
    elif score <= -0.1:
        return "negative"
    else:
        return "neutral"
    

# main function to retrieve reviews and prepare them for LLM processing
def summarize_and_sentiment_analysis(db_name: str, schema_name: str, table_name: str):

    with open('./stages/credentials.json') as f:
        credentials = json.load(f)
        api_key = credentials['gemini']['api_key']

    df_all_reviews = get_reviews_from_db(db_name, schema_name, table_name)
    df_closest_reviews = get_reviews_closest_to_centroid(db_name, schema_name, table_name)
    df_random_reviews_for_llm = get_reviews_for_llm(df_closest_reviews, df_all_reviews)
    df_reviews_for_llm = pd.concat(
        [df_closest_reviews, df_random_reviews_for_llm], 
        ignore_index=True
        ).drop_duplicates()

    total_reviews = len(df_all_reviews)
    cluster_volumes = (
        df_all_reviews.groupby('cluster_id')
        .size()
        .reset_index(name='cluster_size')
    )
    cluster_volumes['volume_percent'] = (
        cluster_volumes['cluster_size'] / total_reviews * 100
    )
     
    print(cluster_volumes)

    top_reviews_per_cluster = (
        df_all_reviews.sort_values(["cluster_id", "centroid_rank"])
        .groupby("cluster_id", group_keys=False)
        .head(3)
    )
    # create_table_from_df(df_reviews_for_llm, 'reviews_for_llm')


    print(df_all_reviews)
    cluster_summaries = []

    cluster_ids = df_reviews_for_llm['cluster_id'].unique()

    try:
        find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

    sia = SentimentIntensityAnalyzer()
    df_all_reviews["sentiment_score"] = df_all_reviews["review"].apply(lambda x: sia.polarity_scores(x)["compound"])

    df_all_reviews["sentiment_label"] = df_all_reviews["sentiment_score"].apply(classify_sentiment)

    cluster_sentiment_counts = (
        df_all_reviews.groupby("cluster_id")["sentiment_label"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={
            "positive": "positive_count",
            "negative": "negative_count",
            "neutral": "neutral_count"
        })
    )
    print(df_all_reviews)

    now = datetime.now()

    # cluster_summaries = []
    for cluster_id in df_reviews_for_llm['cluster_id'].unique():
        cluster_reviews = df_reviews_for_llm[df_reviews_for_llm['cluster_id'] == cluster_id]['review']
        cluster_texts = "\n".join(cluster_reviews.tolist())
        
        top_reviews = top_reviews_per_cluster[top_reviews_per_cluster["cluster_id"] == cluster_id]["review"].tolist()
        title, description = summarize_reviews(cluster_id, cluster_texts, api_key)

        volume_percent = cluster_volumes.loc[
            cluster_volumes['cluster_id'] == cluster_id, 'volume_percent'
        ].iloc[0]

        sentiment_row = cluster_sentiment_counts[cluster_sentiment_counts["cluster_id"] == cluster_id].iloc[0]


        cluster_summaries.append({
            "cluster_id": cluster_id,
            "title": title,
            "description": description,
            "top_reviews": top_reviews,
            "volume_percent": volume_percent,
            "positive_count": sentiment_row["positive_count"],
            "negative_count": sentiment_row["negative_count"],
            "neutral_count": sentiment_row["neutral_count"],
            "datetime": now
        })
    # # Convert summaries to a DataFrame and save
   
    # with ThreadPoolExecutor(max_workers=15) as executor:
    #     futures = {
    #         executor.submit(process_cluster, cid, df_reviews_for_llm, api_key): cid
    #         for cid in cluster_ids
    #     }
    #     for future in as_completed(futures):
    #         cid = futures[future]
    #         try:
    #             result = future.result()
    #             cluster_summaries.append(result)
    #         except Exception as e:
    #             print(f"Error processing cluster {cid}: {e}")

    df_summaries = pd.DataFrame(cluster_summaries)
    df_summaries['datetime'] = pd.to_datetime(df_summaries['datetime'], errors='coerce')
    print(df_summaries)


    create_table_from_df(df_summaries, 'cluster_summaries')