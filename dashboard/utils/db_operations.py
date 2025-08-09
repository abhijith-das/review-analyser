from utils.snowflake_connector import run_query
from utils.read_config import get_tbl_clstr_smmrs


def retrieve_table_as_df(db_name, schema_name, table_name):
    query = f"SELECT * FROM {db_name}.{schema_name}.{table_name};"
    df = run_query(query)
    # print(f"Retrieved {len(df)} rows from {table_name}")
    return df

def retrieve_table_as_df_for_a_date(date):

    date_str = date.strftime("%Y-%m-%d") if not isinstance(date, str) else date

    smmrs = get_tbl_clstr_smmrs()
    smmrs_cols = smmrs["COLUMNS"]
    query = f'''
    SELECT * FROM "{smmrs["DATABASE"]}"."{smmrs["SCHEMA"]}"."{smmrs["TABLE"]}" WHERE DATE("{smmrs_cols["DATETIME"]}") = '{date_str}';
    '''
    # print(query)
    df = run_query(query)

    df = df.set_index("cluster_id", drop=False)
    # print(f"Retrieved {len(df)} rows from {table_name}")
    return df

