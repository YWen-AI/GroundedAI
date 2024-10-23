# matgpt-evaluation
```
matgpt-evaluation/
│
├── data/
│   ├── eval_output/     # Evaluation results data goes here
│   ├── test_dataset/    # Test dataset
│   └── vectors_db/      # Vector database
│
├── data_preprocessing/  # Storing scripts needed for making test dataset
│
├── dependencies/
│   └── matgpt-evaluation.yml  # List of Python dependencies
│
├── src/                # Source code for the project
│   └── config.json     # Configuration file
│
├── tests/              # Unit tests for the codebase
│
├── scripts/            # Scripts for generating synthesis dataset, running evaluation, etc.
│   ├── LLM_QAGeneration_ground_truths.py
│   └── rag_eval.py
│
├── requirements.txt    # List of Python dependencies
├── README.md           # Project overview, how to use, install, etc.
├── .gitignore          # Files and directories to ignore in Git
└── LICENSE             # License for the project
```