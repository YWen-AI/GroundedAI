import csv
import tiktoken

import pandas as pd


# Helper function for printing docs
def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}: " + d.metadata["document_title"] + "\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


# Helper function for returning two lists: contexts and metadata
def get_contexts_and_metadata(docs):
    contexts, meta_data = [], []
    for i in range(len(docs)):
        contexts.append(docs[i].page_content)
        meta_data.append(docs[i].metadata)

    return contexts, meta_data


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    """embedding models like 'ada_v2', 'text-embedding-3-small', use the cl100k_base encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Function to detect delimiter
def detect_delimiter(file_path):
    with open(file_path, 'r') as file:
        sample = file.read(40960)  # Read first 40960 bytes (or some other appropriate sample size)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        return dialect.delimiter


def random_rows_from_csv(file1, file2, rows_from_file1, rows_from_file2, output_file):
    """
    Selects a random set of rows from two CSV files and saves them to a new CSV file.

    Args:
    file1 (str): Path to the first CSV file.
    file2 (str): Path to the second CSV file.
    rows_from_file1 (int): Number of rows to select from the first CSV file.
    rows_from_file2 (int): Number of rows to select from the second CSV file.
    output_file (str): Path where the output CSV file will be saved.

    Returns:
    None
    """
    delimiter = detect_delimiter(file1)

    # Load the data from the first CSV file
    data1 = pd.read_csv(file1, delimiter=delimiter)
    # Select a random sample of rows from the first file
    sample1 = data1.sample(n=rows_from_file1)

    delimiter = detect_delimiter(file2)
    # Load the data from the second CSV file
    data2 = pd.read_csv(file2, delimiter=delimiter)
    # Select a random sample of rows from the second file
    sample2 = data2.sample(n=rows_from_file2)

    # Concatenate the two samples into a single DataFrame
    final_sample = pd.concat([sample1, sample2])

    # Save the combined DataFrame to a new CSV file
    final_sample.to_csv(output_file, index=False)

    print(f"Saved {rows_from_file1} rows from {file1} and {rows_from_file2} rows from {file2} to {output_file}")
