import os
import random
import pandas as pd

from langchain.document_loaders import PyPDFLoader
from langchain.evaluation.qa import QAGenerateChain
from langchain.chat_models import ChatOpenAI


def load_pdfs_and_sampling_pages(folder_path='backend/data/pdfdocs/'):
    """
    This function Load PyPDFLoader and  for large documents where number of pages > 30, just sample 30 pages.
    Because we will ask LLM to generate 1 question per page.
    If it is a medium size document, say, page number between 10 to 30, sample 10 pages.
    If it is a small size document, don't need to sample.

    Input:
        folder_path: the folder path of the pdf documents

    Output:
        pages: a list of sampled pages from the pdf documents
    """

    file_list = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            file_list.append(os.path.join(folder_path, filename))

    random.seed(42)
    pages = []

    print("Total number of documents: ", len(file_list))

    for file in file_list:
        # load documents
        loader = PyPDFLoader(file)
        print("Loading document: ", file)
        doc = loader.load()
        # randomly select a few pages from document
        sample_size = len(doc)
        if len(doc) > 30:
            sample_size = 30
        elif len(doc) > 10:
            sample_size = 10
        pages.extend(random.sample(doc, sample_size))

    return pages


def QAGeneration_from_PDF_pages(pages, model_name="gpt-4"):
    """
    This function generate questions and answers from the pdf pages using LLM model.

    Input:
        pages: a list of sampled pages from the pdf documents
        model_name: the name of the LLM model

    Output:
        df: a dataframe of the generated questions and answers
    """

    example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model_name=model_name))
    QAPairs = example_gen_chain.apply_and_parse([{"doc": t} for t in pages])

    df_QA = pd.DataFrame([item['qa_pairs'] for item in QAPairs])

    df_QA.rename(columns={'answer': 'ground_truth'}, inplace=True)  # query, ground_truth, context, answer are needed for RAG evaluation

    return df_QA


def gather_PDF_metadata(pages):
    """
    This function gather the metadata of the pdf pages and output a dataframe.

    Input:
        pages: a list of sampled pages from the pdf documents

    Output:
        df_metadata: a dataframe of the metadata of the pdf pages
    """

    metadata_list = []
    for i in range(len(pages)):
        metadata_list.append(pages[i].metadata)

    df_metadata = pd.DataFrame(metadata_list)

    return df_metadata


def form_test_dataset(df_QA, df_metadata, output_file_name='output_firstQAEval_TEST.csv'):
    """
    This function form the test QA dataset for the RAG evaluation. Write a CSV file of the test QA dataset and output a dataframe.

    Input:
        df_QA: a dataframe of the generated questions and answers from running LangChain QAGeneration chain on PDFs
        df_metadata: a dataframe of the metadata of the pdf titles and pages where each QA could link to
        output_file_name: the name of the output file

    Output:
        df_final: a dataframe of the test QA dataset
    """

    if df_QA.empty or df_metadata.empty:
        raise ValueError("Input DataFrames should not be empty!")

    df_final = df_QA.join(df_metadata)

    if df_final.empty:
        raise ValueError("Join operation resulted in an empty DataFrame!")

    df_final.to_csv(output_file_name, index=False)

    return df_final


if __name__ == "__main__":
    pages = load_pdfs_and_sampling_pages()
    df_QA = QAGeneration_from_PDF_pages(pages)
    df_metadata = gather_PDF_metadata(pages)
    df_final = form_test_dataset(df_QA, df_metadata)
