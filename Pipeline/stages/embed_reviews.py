import os
import uuid
import shutil
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
# from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from utils.read_config import get_embed_config, get_source_file_cols

# function to load a specific text column from a parquet file
def load_parquet_columns(parquet_path: str, text_column: str = "text", product_column: str = "product") -> list:
    df = pd.read_parquet(parquet_path)
    print(df)
    if text_column not in df.columns or product_column not in df.columns:
        raise ValueError(f"Columns '{text_column}' and '{product_column}' are mandatory in parquet file.")
    texts = df[text_column].astype(str).tolist()
    products = df[product_column].astype(str).tolist() 
    return texts, products

# function to convert a list of texts and products into Document objects with unique IDs and product metadata
def convert_to_documents(texts: list, products: list) -> list:
    return [
        Document(
            page_content=text,
            metadata={"id": str(uuid.uuid4()), "product": product}
        )
        for text, product in zip(texts, products)
    ]

# function to store documents in a Chroma vector store with embeddings
def store_in_chroma_store(documents: list, embedding_function: HuggingFaceEmbeddings, persist_directory: str = "chroma_store"):
    # remove the directory if it exists
    if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            print(f"Deleted existing Chroma store at '{persist_directory}'")
    print(f"Storing {len(documents)} documents into Chroma store")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=persist_directory,
        collection_name="reviews_collection"
    )
    try:
        vectorstore.persist()
    except Exception as e:
        print(f"Error persisting Chroma vector store: {e}")
    print(f"Chroma vector store persisted at '{persist_directory}'")

# function to store documents with embeddings as a parquet file
def store_as_parquet(documents: list, embedding_function: HuggingFaceEmbeddings, output_path: str):
    print("Generating embeddings and building records...")
    records = []
    total_docs = len(documents)
    for doc in tqdm(documents, desc="Processing documents", unit="doc"):
        embedding = embedding_function.embed_query(doc.page_content)
        records.append({
            "review_id": doc.metadata["id"],
            "product": doc.metadata["product"],
            # "timestamp": doc.metadata["timestamp"],
            "review": doc.page_content,
            "embedding": embedding
        })
    df = pd.DataFrame(records)
    df.to_parquet(output_path, compression='snappy', index=False)
    print(f"Saved {len(df)} records with embeddings to {output_path}")


# main function to process the parquet file, convert texts to Document objects, and store them in Chroma
def process_and_store(parquet_path: str, output_path: str, text_column: str = "text", product_column: str = "product"):
    print(f"Loading texts from parquet file: '{parquet_path}'")
    texts, products = load_parquet_columns(parquet_path, text_column, product_column)
    print(f"Loaded {len(texts)} texts from column '{text_column}' and '{product_column}'")

    print("Loading embedding function...")
    embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    print("Embedding function loaded")

    documents = convert_to_documents(texts, products)
    print(f"Converted {len(documents)} texts to Document objects")
    # print(documents[:3])  

    # store_in_chroma_store(documents, embedding_function, persist_directory)
    # output_path="parquets/reviews_with_embeddings.parquet"
    store_as_parquet(documents, embedding_function, output_path)

# main function for the airflow DAG
def main():
    embed_config = get_embed_config()
    cols = get_source_file_cols()
    process_and_store(
        parquet_path=embed_config["source"]["parquet_file"],
        output_path=embed_config["target"]["parquet_file"],
        text_column=cols["TEXT"],
        product_column= cols["PRODUCT"]
    )
