import re
from typing import List, Set
from pydantic import BaseModel
from langchain_core.documents import Document

def format_docs_with_title(docs: List[Document]) -> str:
    if not all(isinstance(doc, Document) for doc in docs):
        raise ValueError
        
    formatted = [
        f"Source Title: {doc.metadata['document_title']}\nSource Snippet: {doc.page_content}"
        for doc in docs
    ]

    return "\n\n".join(formatted)

def find_references(text: str) -> List[str]:
    """
    Find all references in the given text that match the pattern [source: Source Title].

    Args:
        text (str): The text in which to find references.

    Returns:
        List[str]: A list of references found in the text.

    """

    # Find all occurrences of [source: Source Title]
    reference_pattern = r'\[source:\s*([^\]]+?)\s*\]'
    matches = re.findall(reference_pattern, text)
    
    return matches

def get_unique_references(references: List[str]) -> List[str]:
    """
    Remove duplicate references from the list while maintaining order.

    Args:
        references (List[str]): The list of references.

    Returns:
        List[str]: A list of unique references.
    """

    if not all(isinstance(ref, str) for ref in references):
        raise ValueError
        # Remove duplicates while maintaining order
    seen = set()
    unique_references = []
    for ref in references:
        if ref not in seen:
            unique_references.append(ref)
            seen.add(ref)

    return unique_references

def create_reference_mapping(unique_references: List[str]) -> dict:
    """
    Create a mapping of reference titles to reference numbers.

    Args:
        unique_references (List[str]): A list of unique reference titles.

    Returns:
        dict: A dictionary mapping each reference title to a unique reference number.
    """
    # Ensure all elements in the list are strings
    if not all(isinstance(ref, str) for ref in unique_references):
        raise ValueError
    
    # Create a mapping of reference title to reference number
    return {ref: str(i + 1) for i, ref in enumerate(unique_references)}

def replace_reference(match: re.Match, reference_mapping: dict) -> str:
    """
    Replace a reference title in the match object with its corresponding number from the reference mapping.

    Args:
        match (re.Match): A regex match object containing the reference title.
        reference_mapping (dict): A dictionary mapping reference titles to reference numbers.

    Returns:
        str: The reference number in square brackets.
    """

    if not isinstance(match, re.Match) or not isinstance(reference_mapping, dict):
        raise TypeError
    # Extract the reference title from the match object      
    ref_title = match.group(1).strip()
        
    # Check if the reference title exists in the reference mapping
    if ref_title not in reference_mapping:
        raise KeyError
        
    return f"[{reference_mapping[ref_title]}]"

def replace_inline_citations(text: str, reference_mapping: dict) -> str:
    """
    Replace all inline citations in the text with corresponding reference numbers.

    Args:
        text (str): The text containing inline citations.
        reference_mapping (dict): A dictionary mapping reference titles to reference numbers.

    Returns:
        str: The text with inline citations replaced by reference numbers.
    """
    reference_pattern = r'\[source:\s*([^\]]+)\s*\]'
    result = re.sub(reference_pattern, lambda match: replace_reference(match, reference_mapping), text)

    return result

def create_reference_list(unique_references: List[str]) -> str:
    """
    Create a formatted reference list from a list of unique references.

    Args:
        unique_references (List[str]): A list of unique reference titles.

    Returns:
        str: A formatted string representing the reference list.
    """

    if not all(isinstance(ref, str) for ref in unique_references):
        raise ValueError
        
    # Create the reference list
    reference_list = "\n\n #### References:\n\n" + "\n\n".join(f"[{i+1}] {ref}" for i, ref in enumerate(unique_references))
    return reference_list

def extract_and_replace_references(text:str) -> str: 
    """
    Extract references from the text, replace inline citations with reference numbers, 
    and append a reference list at the end.

    Args:
        text (str): The text containing inline citations.

    Returns:
        str: The updated text with references replaced by numbers and a reference list appended.
    """

    # Find all references in the text
    references = find_references(text)
    # If no references are found, return the original text
    if not references:
        return text
    # Get unique references
    unique_references = get_unique_references(references)
    # Create a mapping of reference title to reference number
    reference_mapping = create_reference_mapping(unique_references)
    # Replace inline citations with reference numbers
    updated_text = replace_inline_citations(text, reference_mapping)
    # Create a reference list
    reference_list = create_reference_list(unique_references)
    # Combine the updated text and reference list
    result_text = updated_text + "\n\n" + reference_list

    return result_text
