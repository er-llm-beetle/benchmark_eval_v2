# import random
# import os
import yaml

# from typing import List
import pandas as pd
import traceback  # To capture the stack trace for logging errors

from multiple_choice import get_model_answer_multiple_options
from multiple_choice import compare_answers

from rag import get_answer_from_local_ollama_context
from rag import get_evaluation_score_context

from qa_quality import get_answer_from_local_ollama
from qa_quality import get_evaluation_score, calculate_rouge_score, calculate_bleu_score, calculate_levenshtein_score

from evalutate_yaml_w_scores_chunked_get_answers import run_benchmark_store_answers
from evalutate_yaml_w_scores_chunked_get_scores import run_benchmark_get_scores



# YAML file Metadata
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access metadata and dataset files
metadata = config['metadata']
dataset_files = config['dataset_files']
results_file = config['output']['results_file']


def get_benchmark_from_filename(filename, metadata):
    for ending, benchmark_type in metadata['dataset_naming_convention'].items():
        print("GET BENCH")
        ending = ending + '.xlsx'
        if filename.endswith(ending):
            return benchmark_type
    raise ValueError(f"Filename {filename} does not match any known benchmark type")



# Call this function to run Step 1: Get answers (predictions) and store them in Excel
def run_step_1_store_answers():
    # from evalutate_yaml_w_scores_chunked_get_answers import run_benchmark_store_answers
    print("Running Step 1: Store Answers")

    for file in dataset_files:
        benchmark_type = get_benchmark_from_filename(file, metadata)
        df = pd.read_excel(file)

        # Limit for testing (Optional)
        df = df[:2]

        for model_name in metadata['supported_models']:
            print(f"Storing answers for {benchmark_type} benchmark with model {model_name}")
            try:
                # print('try 1')
                run_benchmark_store_answers(model_name, benchmark_type, df)
            except Exception as e:
                print(f"Error during answer storage: {str(e)}")
                traceback.print_exc()


# Call this function to run Step 2: Calculate scores from the stored Excel files
def run_step_2_calculate_scores():
    # from evalutate_yaml_w_scores_chunked_get_scores import run_benchmark_get_scores
    print("Running Step 2: Calculate Scores")

    # DataFrame to store final results
    results = pd.DataFrame(columns=metadata['benchmark_types'].keys(), index=metadata['supported_models'])

    for model_name in metadata['supported_models']:
        for benchmark_type in metadata['benchmark_types'].keys():
            try:
                print(f"Running Calculate Scores for model: {model_name}, benchmark: {benchmark_type} ")
                # print('try 2')
                average_score = run_benchmark_get_scores(model_name, benchmark_type)
                results.loc[model_name, benchmark_type] = average_score
            except Exception as e:
                print(f"Error during score calculation: {str(e)}")
                traceback.print_exc()

    # Save the results after calculating all scores
    print("\nAverage Scores:\n", results)
    results.to_excel(results_file)
    print(f"Results saved to {results_file}")


# Combine both steps in one function call
def run_both_steps():
   # First, run Step 1: Get answers (predictions)
    print("RUN FIRST STEP")
    run_step_1_store_answers()

    print("RUN SECOND STEP")
    # Second, run Step 2: Calculate scores
    run_step_2_calculate_scores()


# Main function to trigger both steps
if __name__ == '__main__':
    run_both_steps()
