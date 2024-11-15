import pandas as pd

from src.rags.rags import initialize_qa_retrieval_chain, start_prompting

from src.utils.config_utils import load_config
from src.utils.word_processing import detect_delimiter


async def batch_rags(question_file, config_rag, batch_mode=True, batch_num=15, debug=False):

    qa_chain = initialize_qa_retrieval_chain(config_rag)
    delimiter = detect_delimiter(question_file)
    df = pd.read_csv(question_file, delimiter=delimiter)

    df_qa_pairs = pd.DataFrame(columns=['question', 'answer', 'contexts', 'metadata'])

    rows = []
    max_index = df.shape[0]

    if not batch_mode:
        batch_num = 1

    if debug:
        max_index = min(batch_num, max_index)  # Process only one batch if in debug mode

    for start_idx in range(0, max_index, batch_num):
        end_idx = min(start_idx + batch_num, max_index)
        batch_df = df.iloc[start_idx:end_idx]

        batch_questions = batch_df['question'].tolist()
        batch_questions_dict = [{'question': question, 'chat_history': []} for question in batch_questions]

        batch_answers, batch_contexts, batch_metadata = await start_prompting(qa_chain, batch_questions_dict, batch_mode=batch_mode)

        # Append results for this batch
        for question, answer, contexts, meta_data in zip(batch_questions, batch_answers, batch_contexts, batch_metadata):
            rows.append({
                'question': question,
                'answer': answer,
                'contexts': contexts,
                'metadata': meta_data
            })


    new_df = pd.DataFrame(rows)

    df_qa_pairs = pd.concat([df_qa_pairs, new_df], ignore_index=True)

    return df_qa_pairs

async def main():
    config = load_config()
    df_output = await batch_rags("data/test_datasets/ground_truths/golden_v1.csv", config["rag_parameters"])
    df_output.to_csv('output_golden_v1_brute_force.csv')

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())