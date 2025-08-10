import json
import numpy as np
import pandas as pd
import uuid
from datetime import datetime   
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.cluster import KMeans

from utils.db_operations import create_table_from_df

# Load documents and embeddings from Chroma
def load_from_chroma_store(persist_directory: str = "chroma_store") -> tuple:
    embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    print(f"Loading embeddings from Chroma store at '{persist_directory}'")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function, collection_name="reviews_collection")
    print("Retrieving documents and embeddings from Chroma store...")
    results = vectorstore.get(include=["embeddings", "documents", "metadatas"])
    # print(results)
    embeddings = np.array(results["embeddings"])
    documents = results["documents"]
    metadatas = results["metadatas"]
    return documents, metadatas, embeddings


# Load documents and embeddings from a saved Parquet file
def load_from_parquet(parquet_path: str = "parquets/reviews_with_embeddings.parquet") -> tuple:
    print(f"Loading reviews and embeddings from parquet: '{parquet_path}'")
    
    df = pd.read_parquet(parquet_path)

    if not {"review", "product", "embedding"}.issubset(df.columns):
        raise ValueError("Parquet file must contain 'review', 'product', and 'embedding' columns.")

    # Convert embeddings from object (list) to np.ndarray
    embeddings = np.vstack(df["embedding"].apply(np.array).values)
    documents = df["review"].tolist()
    metadatas = [{"product": p, "review_id": rid} for p, rid in zip(df["product"], df["review_id"])]

    print(f"Loaded {len(documents)} documents and {embeddings.shape[0]} embeddings.")
    return documents, metadatas, embeddings


# KMeans clustering
def cluster_embeddings_kmeans(embeddings: np.ndarray, k: int = 10) -> list:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans

# Group reviews and products by cluster
def group_reviews_by_cluster(documents: list, metadatas: list, labels: list) -> dict:
    clusters = {}
    for label, doc, meta in zip(labels, documents, metadatas):
        clusters.setdefault(label, []).append({
            "text": doc,
            "product": meta.get("product", "unknown")
        })
    return clusters

# Get closest reviews to each cluster centroid
def get_closest_to_centroid_flags(embeddings, labels, kmeans_model, top_n=30):
    is_close_flag = np.zeros(len(labels), dtype=bool)  # Default all to False
    ranks = np.zeros(len(labels), dtype=int) # To store ranks

    for cluster_id in range(kmeans_model.n_clusters):
        # Indices of reviews in this cluster
        cluster_indices = np.where(labels == cluster_id)[0]

        # Embeddings for this cluster
        cluster_embeddings = embeddings[cluster_indices]

        # Cluster centroid
        centroid = kmeans_model.cluster_centers_[cluster_id]

        # Distance from centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

        # Top N closest
        closest_indices = cluster_indices[np.argsort(distances)[:top_n]]

        # Mark them True
        is_close_flag[closest_indices] = True

        # Get sorted indices (closest first)
        sorted_indices = np.argsort(distances)

        # Assign ranks (1 = closest)
        cluster_ranks = np.arange(1, len(sorted_indices) + 1)
        ranks[cluster_indices[sorted_indices]] = cluster_ranks

    return is_close_flag, ranks

# build a DataFrame from clustered reviews
def build_cluster_dataframe(documents, metadatas, labels, is_close_flag):
    records = []
    now = datetime.now()

    for i, (doc, meta, label, close_flag) in enumerate(zip(documents, metadatas, labels, is_close_flag)):
        record = {
            "review_id": str(uuid.uuid4()),  # Unique identifier
            "cluster_id": int(label),
            "review": doc,
            "datetime": now,
            "product": meta.get("product", "unknown"),
            "is_close_to_centroid": close_flag
        }
        records.append(record)

    return pd.DataFrame(records)

# Main logic
def cluster_reviews_from_chroma(k: int = 10):
    print(f"Loading documents from parquet file")
    documents, metadatas, embeddings = load_from_parquet("stages/parquets/reviews_with_embeddings.parquet")
    print(metadatas[:3])  # Print first 3 metadata for verification
    print(f"Loaded {len(documents)} documents from parquet file")

    labels, kmeans = cluster_embeddings_kmeans(embeddings, k=k)

     # Determine closest-to-centroid flags
    is_close_flag, ranks = get_closest_to_centroid_flags(embeddings, labels, kmeans, top_n=30)


    df = build_cluster_dataframe(documents, metadatas, labels, is_close_flag)

    df['centroid_rank'] = ranks
    print(df.head()) 

    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    create_table_from_df(df, "CLUSTERED_REVIEWS", mode='overwrite')

    # print(labels)
    # print(f"Clustered documents and generated {len(labels)} labels")

    # clusters = group_reviews_by_cluster(documents, metadatas, labels)

    # print(clusters)

    # data = {np.int32(1): "value1", np.int32(2): "value2"}

    # Convert int32 keys to standard Python int
    # converted_data = {k.item(): v for k, v in clusters.items()}
    # save clusters to a JSON file for reference
    # with open("clusters.json", "w") as f:
    #     json.dump(converted_data, f, indent=4)

    # for cluster_id, grouped_items in clusters.items():
    #     print(f"\nCluster {cluster_id} ({len(grouped_items)} reviews):")
    #     for item in grouped_items[:3]:
    #         print(f"- {item['text'][:100]}... ({item['product']})")

