import pandas as pd
from snowflake.connector.pandas_tools import write_pandas
from utils.snowflake_connector import run_query, get_snowflake_session



def create_table_structure(conn, df: pd.DataFrame, table_name: str):
    cursor = conn.cursor()
    columns_sql = []
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_integer_dtype(dtype):
            sql_type = "NUMBER"
        elif pd.api.types.is_float_dtype(dtype):
            sql_type = "FLOAT"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            sql_type = "TIMESTAMP_NTZ"
        else:
            sql_type = "VARCHAR"
        columns_sql.append(f'"{col}" {sql_type}')

    create_sql = f'CREATE OR REPLACE TABLE "{table_name}" (\n  ' + ",\n  ".join(columns_sql) + "\n);"
    
    
    print("Creating table...")
    cursor.execute(create_sql)
    print(f"Table '{table_name}' created.")


def create_table_from_df(df, table_name):
    conn = get_snowflake_session()
    create_table_structure(conn, df, table_name)
    success, nchunks, nrows, _ = write_pandas(conn, df, table_name, use_logical_type=True)
    print(f"Uploaded {nrows} rows to {table_name}")

def retrieve_table_as_df(db_name, schema_name, table_name):
    query = f"SELECT * FROM {db_name}.{schema_name}.{table_name};"
    df = run_query(query)
    # print(f"Retrieved {len(df)} rows from {table_name}")
    return df

def retrieve_values_closest_to_centroid(db_name, schema_name, table_name):
    query = f"""
    SELECT * FROM {db_name}.{schema_name}.{table_name}
    WHERE "is_close_to_centroid" = TRUE;  
    """
    df = run_query(query)
    # print(f"Retrieved {len(df)} rows from {table_name} where is_close_to_centroid is TRUE")
    return df

def truncate_table(db_name, schema_name, table_name):
    query = f"TRUNCATE TABLE {db_name}.{schema_name}.{table_name};"
    run_query(query)
    print(f"Table {table_name} truncated successfully.")
