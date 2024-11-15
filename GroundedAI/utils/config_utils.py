import json

def load_config(file_path="src/config.json"):
    """Load configuration file.
    Args:
        file_path (str): Path to the configuration file.
    Returns:
        dict: Configuration parameters.
    """

    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The configuration file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return None