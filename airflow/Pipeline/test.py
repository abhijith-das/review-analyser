from stages.cluster import cluster_reviews
# from stages.summarize import summarize_and_sentiment_analysis
from stages.extract_and_clean import main as extract_and_clean_main
from stages.embed_reviews import main as embed_reviews_main
from stages.cluster import main as cluster_reviews_main
from stages.summarize import main as summarize_reviews_main


# cluster_reviews_from_chroma(
#     k=10
# ) 

# summarize_and_sentiment_analysis(
#     db_name="ANALYSER_DB",
#     schema_name="ANALYSER_SCHEMA",
#     table_name="CLUSTERED_REVIEWS"
# )

summarize_reviews_main()