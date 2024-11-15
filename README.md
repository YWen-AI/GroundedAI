# GroundedAI
GroundedAI Is All You Need. A responsible, trustworthy RAG application.

## Setting up the project
With the use of Poetry. 

To install Poetry:

`curl -sSL https://install.python-poetry.org | python3 -`

To install the core package for production dependencies, provided the `pyproject.toml` file is already in, run:

`poetry install`

To switch to dev dependencies, run:

`poetry install --with dev`.

## Update the project file
If you changed dependencies, you need to rebuild the Lock File, run:

`poetry lock`

## Run the project
### Run the script
If you have other scripts or tests, you can run them with:
`poetry run python path/to/your_script.py`

### Run the test
`poetry run pytest test/`


## Stucture
```
GroundedAI/
│
├── data/
│   ├── eval_output/     # Evaluation results data goes here
│   ├── test_dataset/    # Test dataset
│   └── vectors_db/      # Vector database
│
├── data_preprocessing/  # Storing scripts needed for making test dataset
│
├── dependencies/
│   └── Grounded-AI.yml  # List of Python dependencies
│
├── GroundedAI/         # Source code and modules for the project
│   └── config.json     # Configuration file
│
├── tests/              # Unit tests for the codebase
│
├── scripts/            # Scripts for generating synthesis dataset, running evaluation, etc.
│   ├── LLM_QAGeneration_ground_truths.py
│   └── rag_eval.py
│
├── README.md           # Project overview, how to use, install, etc.
```