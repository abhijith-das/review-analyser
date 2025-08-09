import numpy as np
import pandas as pd
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


# main function to retrieve reviews and prepare them for LLM processing
def summarize_and_sentiment_analysis(db_name: str, schema_name: str, table_name: str):
    df_all_reviews = get_reviews_from_db(db_name, schema_name, table_name)
    df_closest_reviews = get_reviews_closest_to_centroid(db_name, schema_name, table_name)
    df_random_reviews_for_llm = get_reviews_for_llm(df_closest_reviews, df_all_reviews)
    df_reviews_for_llm = pd.concat([df_closest_reviews, df_random_reviews_for_llm], ignore_index=True).drop_duplicates()

    print(df_reviews_for_llm)

    create_table_from_df(df_reviews_for_llm, 'reviews_for_llm')