import argparse
from ragas.metrics import answer_relevancy, faithfulness, context_recall, context_precision, answer_correctness, answer_similarity

from GroundedAI.rags.batch_rags import batch_rags
from GroundedAI.evaluators.ragas import RagasEvaluator

from GroundedAI.utils.config_utils import load_config


async def rag_eval(only_eval, tag, evaluator_type, input_groundtruths_file, input_groundtruths_folder, generated_folder, output_folder, metrics, bad_PDF):
    config = load_config("src/config.json")

    if not only_eval:
        df_rag_output = await batch_rags(input_groundtruths_folder + input_groundtruths_file, config["rag_parameters"])
        df_rag_output.to_csv(generated_folder + 'output_' + tag + '.csv')

    output_rag_filename = 'output_' + tag + '.csv'
    output_eval_file_name = output_folder + 'output_RAGAS_' + tag + '.csv'

    if evaluator_type == "RAGAS":
        evaluator = RagasEvaluator(metrics, input_groundtruths_folder, generated_folder,
                                    input_groundtruths_file, output_rag_filename,
                                    output_eval_file_name, bad_PDF)
        evaluator.evaluate()
    else:
        print("You can only input valid evaluator method, currently supporting RAGAS.")
        raise ValueError

    return


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RAGAs.')
    parser.add_argument('--only_eval', type=bool, default=False, help='Switch on if only need to eval on an existing data.')
    parser.add_argument('--tag', type=str, default='', help='Tag which you think is important.')
    parser.add_argument('--evaluator', type=str, default='RAGAS', help='Evaluator.')
    parser.add_argument('--groundtruths_file', type=str, default='golden_v2.csv', help='CSV file of the ground truths.')
    parser.add_argument('--groundtruths_folder', type=str, default='./data/test_datasets/ground_truths/', help='Folder where groundtruths are stored.')
    parser.add_argument('--generated_folder', type=str, default='./data/test_datasets/generated/', help='Folder where generated data is stored.')
    parser.add_argument('--output_folder', type=str, default='./data/eval_outputs/', help='Folder of the output evaluation.')
    parser.add_argument('--metrics', type=list, default=[context_precision, faithfulness, answer_relevancy, context_recall, answer_correctness, answer_similarity], help='RAGAS Metrics you want to run.')
    parser.add_argument('--bad_PDF', type=str, default=None, help='Path to the bad PDF.')
    # Add more arguments as needed
    return parser.parse_args()


if __name__ == "__main__":
    import asyncio
    args = parse_args()
    args.tag = 'test_20241001'
    args.only_eval = True
    asyncio.run(rag_eval(
        args.only_eval,
        args.tag,
        args.evaluator,
        args.groundtruths_file,
        args.groundtruths_folder,
        args.generated_folder,
        args.output_folder,
        args.metrics,
        args.bad_PDF)
    )
