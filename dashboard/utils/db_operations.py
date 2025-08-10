from utils.snowflake_connector import run_query
from utils.read_config import get_tbl_clstr_smmrs


def retrieve_table_as_df(db_name, schema_name, table_name):
    query = f"SELECT * FROM {db_name}.{schema_name}.{table_name};"
    df = run_query(query)
    # print(f"Retrieved {len(df)} rows from {table_name}")
    return df

def retrieve_table_as_df_for_a_date(date_selection):
    smmrs = get_tbl_clstr_smmrs()
    smmrs_cols = smmrs["COLUMNS"]

    if len(date_selection) == 2:
        from_date, to_date = date_selection if isinstance(date_selection, tuple) else (date_selection, date_selection)
    else:
        from_date = date_selection[0]
        to_date = from_date
    from_date_str = from_date.strftime("%Y-%m-%d") if not isinstance(from_date, str) else from_date
    if to_date and from_date != to_date:
        to_date_str = to_date.strftime("%Y-%m-%d") if not isinstance(to_date, str) else to_date
        query = f'''
        SELECT *
        FROM "{smmrs["DATABASE"]}"."{smmrs["SCHEMA"]}"."{smmrs["TABLE"]}"
        WHERE DATE("{smmrs_cols["DATETIME"]}") BETWEEN '{from_date_str}' AND '{to_date_str}';
        '''

    else:
        query = f'''
        SELECT * FROM "{smmrs["DATABASE"]}"."{smmrs["SCHEMA"]}"."{smmrs["TABLE"]}" WHERE DATE("{smmrs_cols["DATETIME"]}") = '{from_date_str}';
        '''
    # print(query)
    df = run_query(query)

    df = df.set_index("cluster_id", drop=False)
    # print(f"Retrieved {len(df)} rows from {table_name}")
    return df

