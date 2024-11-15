import os

def load_keys(key_vault, key_owner):
    # Load keys for vdb
    # Get the current environment (default to development)
    current_env = os.getenv('ENV', key_owner)

    # Get the ElasticSearch database key for the current environment
    es_database_keys = key_vault[current_env]

    for key in es_database_keys:
        os.environ[key] = es_database_keys[key]

    return None