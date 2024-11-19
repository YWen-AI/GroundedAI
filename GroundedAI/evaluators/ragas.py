import ast
import argparse
import pandas as pd

from datasets import Dataset, DatasetDict
from sentence_transformers import CrossEncoder

from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_recall, context_precision, answer_correctness,answer_similarity

from GroundedAI.utils.word_processing import detect_delimiter

class RagasEvaluator:
    def __init__(self, metrics, groundtruths_folder, generated_folder, groundtruths_file, generated_file, output, bad_PDF):
        self.metrics = metrics
        self.groundtruths_folder = groundtruths_folder
        self.generated_folder = generated_folder
        self.groundtruths_file = groundtruths_file
        self.generated_file = generated_file
        self.output = output
        self.bad_PDF = bad_PDF
        self.cross_encoder = CrossEncoder("cross-encoder/stsb-TinyBERT-L-4")
    
    def load_generated_QA_dataset(self):
        delimiter = detect_delimiter(self.generated_folder + self.generated_file)
        df = pd.read_csv(self.generated_folder + self.generated_file, delimiter=delimiter)
        df.rename(columns={'query': 'question', 'context':'contexts'}, inplace=True)
        return df
    
    def load_groundtruths_QA_dataset(self):
        delimiter = detect_delimiter(self.groundtruths_folder + self.groundtruths_file)
        df = pd.read_csv(self.groundtruths_folder + self.groundtruths_file, delimiter=delimiter)
        df.rename(columns={'query': 'question', 'answer':'ground_truth'}, inplace=True)
        return df

    def merge_generated_groundtruths_QA_dataset(self, df_generated, df_groundtruths):
        if df_generated.empty or df_groundtruths.empty:
            raise ValueError("Input DataFrames should not be empty!")
        if len(df_generated) != len(df_groundtruths):
            raise ValueError("Two files have different number of question! Unable to compare.")
        if not df_generated['question'].equals(df_groundtruths['question']):
            raise ValueError("The questions are not the same!")
        df = df_groundtruths.join(df_generated[['contexts','answer']])
        if df.empty:
            raise ValueError("Join operation resulted in an empty DataFrame!")
        return df
    
    def remove_bad_PDF(self, df):
        df = df.drop(df[df['source'] == self.bad_PDF].index)
        df = df.reset_index(drop=True)
        return df
    
    def QA_dataframe_preprocessing(self, df):
        """
        This function preprocess the dataframe of the QA dataset for the RAG evaluation.

        Input:
            df: a dataframe of the QA dataset to be preprocessed.

        Output:
            df: a dataframe of the preprocessed QA dataset
            
        """
        # Remove metadata
        df = df.drop(columns=['source','page'], axis=1)

        # Convert the ground_truth and contexts columns from string to list
        #df['ground_truth'] = df['ground_truth'].apply(lambda x: [x])

        # drop nan cloumns
        if df["contexts"].isna().all():
            df = df.drop(columns=['contexts'], axis=1)
        else:
            df["contexts"] = df["contexts"].apply(ast.literal_eval)
        
        return df
    
    def transform_df_2_dataset(self, df, dataset_name):
        """
        This function transform the dataframe of the QA dataset to the Huggingface dataset for the RAG evaluation.
        Input:
            df: a dataframe of the QA dataset to be transformed.

        Output:
            test_dataset_dict: a Huggingface dataset of the QA dataset
            
        """

        test_dataset = Dataset.from_pandas(df)

        test_dataset_dict = DatasetDict({dataset_name: test_dataset})

        return test_dataset_dict        
    
    def RAGAS_evaluate(self, test_dataset_dict, dataset_name):
        """
        This function evaluate the QA dataset using RAGAS and output a dataframe and csv file of the evaluation results.
        
        Input:
            test_dataset_dict: a Huggingface dataset of the QA dataset to be evaluated.
            test_dataset_name: the name of the test dataset.
            metrics: a list of metrics to be used for the evaluation.
            output_file_name: the name of the output result file.

        Output:
            df_result: a dataframe of the evaluation results
        """
        if self.metrics==None:
            metrics = [context_precision, faithfulness, answer_relevancy, context_recall, answer_correctness, answer_similarity]
        else:
            metrics = self.metrics
        
        result = evaluate(test_dataset_dict[dataset_name],metrics=metrics, raise_exceptions=False)
        print(result)

        df_result = result.to_pandas()

        df_result.to_csv(self.output, index=False)

        return df_result
    
    def run_BertScore(self, dataframe):
        """
        This function evaluate the QA dataset using BertScore and output a dataframe of the evaluation results.

        Input:
            dataframe: a dataframe of the QA dataset to be evaluated.

        Output:
            avg_score: the average BertScore of the QA dataset
        
        """
        cross_encoder = self.cross_encoder
        ground_truth, answers = dataframe["ground_truth"], dataframe["answer"]
        ground_truth = [item[0] for item in ground_truth]
        inputs = [list(item) for item in list(zip(ground_truth, answers))]
        batch_size = 15
        scores = cross_encoder.predict(inputs, batch_size=batch_size, convert_to_numpy=True)
        avg_score = sum(scores)/len(scores)

        return avg_score
    
    def evaluate(self):
        df_groundtruths = self.load_groundtruths_QA_dataset()
        df_generated = self.load_generated_QA_dataset()
        df_merged = self.merge_generated_groundtruths_QA_dataset(df_generated, df_groundtruths)
        df_cleaned = self.remove_bad_PDF(df_merged)
        df_preprocessed = self.QA_dataframe_preprocessing(df_cleaned)
        test_dataset_dict = self.transform_df_2_dataset(df_preprocessed, 'test_data')
        df_result = self.RAGAS_evaluate(test_dataset_dict, 'test_data')
        avg_score = self.run_BertScore(df_preprocessed)
        #print(df_result)
        print("The BertScore of the QA dataset is: ", avg_score)
        print("Done!")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RAGAs.')
    parser.add_argument('--metrics', type=list, default=[context_precision, faithfulness, answer_relevancy, context_recall, answer_correctness, answer_similarity], help='RAGAS Metrics you want to run.')
    parser.add_argument('--groundtruths_folder', type=str, default='./data/test_datasets/ground_truths/', help='Folder where groundtruths are stored.')
    parser.add_argument('--generated_folder', type=str, default='./data/test_datasets/generated/', help='Folder where generated data is stored.')
    parser.add_argument('--groundtruths_file', type=str, default='golden_v1.csv', help='CSV file of the ground truths.')
    parser.add_argument('--generated_file', type=str, default='output_golden_v1_brute_force_prompt_template.csv', help='CSV file of the MatGPT generated answers and contexts retrieved.')
    parser.add_argument('--output', type=str, default='./data/eval_outputs/output_RAGAS_golden_v1_brute_force_prompt_template.csv', help='CSV file of the output evaluation.')
    parser.add_argument('--bad_PDF', type=str, default='./LLM_Demo_pdf/McMahon_2019_-_Evaluation_of_Metal-Rich_Primers_for_the_Mitigation_of_Intergranular_Stress_Corrosion_Cracking_in_Highly_Sensitized_Al-Mg_Alloy_AA5456-H116.pdf', help='Path to the bad PDF.')
    # Add more arguments as needed
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluator = RagasEvaluator(args.metrics, args.groundtruths_folder, args.generated_folder, args.groundtruths_file, args.generated_file, args.output, args.bad_PDF)
    evaluator.evaluate()