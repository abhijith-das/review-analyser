import json
import pandas as pd
import snowflake.connector
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session

# from utils.read_config import get_sf_conn_name, get_sf_role


def get_snowflake_session():
    try:
        get_active_session()
    except Exception:


        with open('/home/abhi/airflow/Pipeline/utils/credentials.json', "r") as f:
            creds = json.load(f)

        sf = creds.get("snowflake", {})
        account = sf.get("account")
        user = sf.get("user")
        role = sf.get("role")
        password = sf.get("password")
        database = sf.get("database")
        schema = sf.get("schema")
        warehouse = sf.get("warehouse")

        # Connect to Snowflake
        conn = snowflake.connector.connect(
            account=account,
            user=user,
            password=password,
            role=role,
            database=database,
            schema=schema,
            warehouse=warehouse
        )
        # print(conn)
        return conn  
        

def run_query(query: str):  
    conn = get_snowflake_session()  
    # metadata = get_connection_info(conn)
    cursor = conn.cursor()
    data = cursor.execute(query).fetchall()
    columns = [col[0] for col in cursor.description]
    cursor.close()
    df = pd.DataFrame(data, columns=columns)
    return df

def get_connection_info(conn):
    cursor = conn.cursor()
    info = {}   

    for key in ["CURRENT_DATABASE()", "CURRENT_SCHEMA()", "CURRENT_WAREHOUSE()", "CURRENT_ROLE()", "CURRENT_USER()"]:
        cursor.execute(f"SELECT {key}")
        value = cursor.fetchone()[0]
        info[key] = value
    
    cursor.close()

    print("Connected to:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return info


# print("Snowflake connection test.")

# data = run_query("SELECT * from ANALYSER_DB.ANALYSER_SCHEMA.CONN_TEST;")
# # data = run_query("CREATE OR REPLACE TABLE CONN_TEST(ID INT, NAME VARCHAR);")
# print(data)