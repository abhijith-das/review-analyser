import yaml
from yaml.loader import SafeLoader
import os
import json

# Function to read a config.yaml file
def read_config() -> dict:
    current_dir = os.getcwd()
    config_path = os.path.join(current_dir, "utils/config.yaml")
    with open(config_path, 'r',  encoding="utf-8",) as file:
        config = yaml.safe_load(file)
    return config

# Function to read a data_config.yaml file
def read_data_config() -> dict:
    current_dir = os.getcwd()
    config_path = os.path.join(current_dir, "utils/data_config.yaml")
    with open(config_path, 'r',  encoding="utf-8",) as file:
        config = yaml.safe_load(file)
    return config

def get_sf_conn_name():
    config = read_config()
    conn = config["SNOWFLAKE"]["CONNECTION_NAME"]
    return conn

def get_sf_role():
    config = read_config()
    role = config["SNOWFLAKE"]["ROLE"]
    return role

def get_tbl_clstr_smmrs():
    config = read_data_config()
    tbl_clstr_smmrs = config["CLUSTER_SUMMARIES"]
    return tbl_clstr_smmrs