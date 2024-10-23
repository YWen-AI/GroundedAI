## Evaluation and folder structure

These are the Evaluation packages for evaluating our LLM-powered application.

The evaluation comes in several steps:


### Pre-requisites:


### Step 1: Obtaining ground truths
Using human to annote datasets are very expensive, let alone this is a highly speciallized alloy development domains. One option would be to feed the documents to LLM and ask it to generate question and answer (QA) pairs and these answers will be served as ground truths.

We will be using [QAGeneration chain](https://api.python.langchain.com/en/latest/evaluation/langchain.evaluation.qa.generate_chain.QAGenerateChain.html) from [LangChain](https://python.langchain.com/docs/get_started/introduction) to call a LLM model (GPT-4) to generate QA pairs.

To run it, simply run:

`python LLM_QAGeneration_ground_truths.py`

The code will simply sample pages from the PDF files. And will generata 1 QA pair per page. After the first round, there will be nonsence QA pairs. One has to pick up 40-50 representative question manually to form the actual test dataset.