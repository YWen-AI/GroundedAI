from langchain_community.document_loaders.csv_loader import CSVLoader

def load_CSV(file_path, delimiter=","):
    loader = CSVLoader(file_path=file_path)

    return loader.load()