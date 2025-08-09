from stages.cluster import cluster_reviews_from_chroma
from stages.summarize import summarize_and_sentiment_analysis


# cluster_reviews_from_chroma(
#     k=10
# ) 

summarize_and_sentiment_analysis(
    db_name="ANALYSER_DB",
    schema_name="ANALYSER_SCHEMA",
    table_name="CLUSTERED_REVIEWS"
)