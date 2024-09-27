# import random
# import os
import yaml

# from typing import List
import pandas as pd

from multiple_choice import get_model_answer_multiple_options
from multiple_choice import compare_answers

from rag import get_answer_from_local_ollama_context
from rag import get_evaluation_score_context

from qa_quality import get_answer_from_local_ollama
from qa_quality import get_evaluation_score, calculate_rouge_score, calculate_bleu_score, calculate_levenshtein_score



# YAML file Metadata
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access metadata and dataset files
metadata = config['metadata']
dataset_files = config['dataset_files']
results_file = config['output']['results_file']

def get_benchmark_from_filename(filename, metadata):
    for ending, benchmark_type in metadata['dataset_naming_convention'].items():
        ending = ending + '.xlsx'
        if filename.endswith(ending):
            return benchmark_type
    raise ValueError(f"Filename {filename} does not match any known benchmark type")

# Prediction handlers
def handle_qa_prediction(question, model):
    return get_answer_from_local_ollama(model, question)

def handle_context_qa_prediction(question, context, model):
    return get_answer_from_local_ollama_context(model, question, context)

def handle_multiple_choice_prediction(question, options, model):
    return get_model_answer_multiple_options(question, options=options, model=model, dstype='mc')

def handle_topic_classification_prediction(question, options, model):
    return get_model_answer_multiple_options(question=question, model=model, options=options, dstype='tc')

def handle_arc_prediction(question, options, model):
    return get_model_answer_multiple_options(question, options=options, model=model, dstype='arc')

# Score handlers
def handle_qa_score(question, actual_answer, predicted_answer):
    score = (0.25 * int(float(get_evaluation_score(question, actual_answer, predicted_answer)))) \
            + calculate_bleu_score(actual_answer, predicted_answer) \
            + calculate_rouge_score(actual_answer, predicted_answer) \
            + calculate_levenshtein_score(actual_answer, predicted_answer)
    return score

def handle_context_qa_score(question, context, actual_answer, predicted_answer):
    score = (0.25 * int(float(get_evaluation_score_context(question, actual_answer, predicted_answer)))) \
            + calculate_bleu_score(actual_answer, predicted_answer) \
            + calculate_rouge_score(actual_answer, predicted_answer) \
            + calculate_levenshtein_score(actual_answer, predicted_answer)
    return score

def handle_multiple_choice_score(actual_answer, predicted_answer):
    return compare_answers(actual_answer=actual_answer, predicted_answer=predicted_answer)

def handle_topic_classification_score(correct_topic, predicted_topic):
    return compare_answers(actual_answer=correct_topic, predicted_answer=predicted_topic)

def handle_arc_score(correct_answer, predicted_option):
    return compare_answers(actual_answer=correct_answer, predicted_answer=predicted_option)

import yaml
import pandas as pd
from multiple_choice import get_model_answer_multiple_options
from rag import get_answer_from_local_ollama_context
from qa_quality import get_answer_from_local_ollama
from qa_quality import calculate_rouge_score, calculate_bleu_score, calculate_levenshtein_score
import traceback  # To capture the stack trace for logging errors



# Store Predictions Function with Error Handling
def store_predictions(df, benchmark_type, model_name):
    predictions = []
    try:
        if benchmark_type == "QA":
            question_col = 'Sual' if 'Sual' in df.columns else df.columns[0]
            answer_col = 'Cavab' if 'Cavab' in df.columns else df.columns[1]

            for index, row in df.iterrows():
                question = row[question_col]
                predicted_answer = handle_qa_prediction(question, model_name)
                predictions.append([question, row[answer_col], predicted_answer])
            
            columns = ['Question', 'Correct Answer', 'Predicted Answer']

        elif benchmark_type == "ContextQA":
            question_col = 'question' if 'question' in df.columns else df.columns[0]
            context_col = 'context' if 'context' in df.columns else df.columns[1]
            answer_col = 'answer' if 'answer' in df.columns else df.columns[2]

            for index, row in df.iterrows():
                question = row[question_col]
                context = row[context_col]
                predicted_answer = handle_context_qa_prediction(question, context, model_name)
                predictions.append([question, context, row[answer_col], predicted_answer])
            
            columns = ['Question', 'Context', 'Correct Answer', 'Predicted Answer']

        elif benchmark_type == "Arzuman":
            question_col = 'text' if 'text' in df.columns else df.columns[0]
            options_col = 'options' if 'options' in df.columns else df.columns[1]
            answer_col = 'answer' if 'answer' in df.columns else df.columns[2]

            for index, row in df.iterrows():
                question = row[question_col]
                options = row[options_col]
                predicted_option = handle_multiple_choice_prediction(question, options, model_name)
                predictions.append([question, row[answer_col], predicted_option])
            
            columns = ['Question', 'Correct Answer', 'Predicted Option']

        elif benchmark_type == "Reshad":
            question_col = 'text' if 'text' in df.columns else df.columns[0]
            options_col = 'options' if 'options' in df.columns else df.columns[1]
            answer_col = 'answer' if 'answer' in df.columns else df.columns[2]

            for index, row in df.iterrows():
                question = row[question_col]
                options = row[options_col]
                predicted_topic = handle_topic_classification_prediction(question, options, model_name)
                predictions.append([question, row[answer_col], predicted_topic])
            
            columns = ['Question', 'Correct Answer', 'Predicted Topic']

        elif benchmark_type == "ARC":
            question_col = 'Azerbaijani_q' if 'Azerbaijani_q' in df.columns else df.columns[0]
            choices_col = 'choices' if 'choices' in df.columns else df.columns[1]
            answer_col = 'answerKey' if 'answerKey' in df.columns else df.columns[2]

            for index, row in df.iterrows():
                question = row[question_col]
                options_txt = row[choices_col]
                array = pd.array
                options_dict = eval(options_txt)
                options = options_dict['az_choices'].tolist()
                predicted_option = handle_arc_prediction(question, options, model_name)
                predictions.append([question, row[answer_col], predicted_option])
            
            columns = ['Question', 'Correct Answer', 'Predicted Option']

        # Dynamically use the column names based on the benchmark type
        output_df = pd.DataFrame(predictions, columns=columns)
        output_filename = f"{benchmark_type}_{model_name}_predictions.xlsx"
        output_df.to_excel(output_filename, index=False)
        print(f"Predictions saved to {output_filename}")

    except Exception as e:
        print(f"Error occurred while storing predictions: {str(e)}")
        traceback.print_exc()  # To print the full error stack trace
        # Save partial results if any errors occur
        if predictions:
            output_df = pd.DataFrame(predictions, columns=columns)
            output_filename = f"{benchmark_type}_{model_name}_predictions_partial.xlsx"
            output_df.to_excel(output_filename, index=False)
            print(f"Partial predictions saved to {output_filename} due to error.")

# Calculate Scores Function with Error Handling
def calculate_scores(predictions_file, benchmark_type):
    scores = []
    try:
        predictions_df = pd.read_excel(predictions_file)

        if benchmark_type == "QA":
            for index, row in predictions_df.iterrows():
                score = handle_qa_score(row['Question'], row['Correct Answer'], row['Predicted Answer'])
                scores.append(score)

        elif benchmark_type == "ContextQA":
            for index, row in predictions_df.iterrows():
                score = handle_context_qa_score(row['Question'], row['Context'], row['Correct Answer'], row['Predicted Answer'])
                scores.append(score)

        elif benchmark_type == "Arzuman":
            for index, row in predictions_df.iterrows():
                score = handle_multiple_choice_score(row['Correct Answer'], row['Predicted Option'])
                scores.append(score)

        elif benchmark_type == "Reshad":
            for index, row in predictions_df.iterrows():
                score = handle_topic_classification_score(row['Correct Answer'], row['Predicted Topic'])
                scores.append(score)

        elif benchmark_type == "ARC":
            for index, row in predictions_df.iterrows():
                score = handle_arc_score(row['Correct Answer'], row['Predicted Option'])
                scores.append(score)

        # Calculate and return average score
        if scores:
            return sum(scores) / len(scores)
        else:
            return 0

    except Exception as e:
        print(f"Error occurred while calculating scores: {str(e)}")
        traceback.print_exc()  # Log the error traceback
        # return None
        if scores: ####
            output_df = pd.DataFrame(scores, columns=benchmark_type)
            output_filename = f"scores_partial.xlsx"
            output_df.to_excel(output_filename, index=False)
            print(f"Partial scores saved to {output_filename} due to error.")

# Main function to run benchmarks and store predictions with error handling
def run_benchmark_store_answers(model_name, benchmark_type, df):
    try:
        store_predictions(df, benchmark_type, model_name)
    except Exception as e:
        print(f"Error while running benchmark for {benchmark_type} with model {model_name}: {str(e)}")
        traceback.print_exc()

# Main function to get scores based on stored answers with error handling
def run_benchmark_get_scores(model_name, benchmark_type):
    try:
        predictions_file = f"{benchmark_type}_{model_name}_predictions.xlsx"
        average_score = calculate_scores(predictions_file, benchmark_type)
        if average_score is not None:
            print(f"Average Score for {model_name} on {benchmark_type}: {average_score}")
        return average_score
    except Exception as e:
        print(f"Error while calculating scores for {benchmark_type} with model {model_name}: {str(e)}")
        traceback.print_exc() ####
        if average_score:
            output_df = pd.DataFrame(average_score, columns=benchmark_type)
            output_filename = f"{benchmark_type}_{model_name}_avg_score_partial.xlsx"
            output_df.to_excel(output_filename, index=False)
            print(f"Average partial scores saved to {output_filename} due to error.")

# Example usage:

results = pd.DataFrame(columns=metadata['benchmark_types'].keys(), index=metadata['supported_models'])

# Step 1: Store all answers first
for file in dataset_files:
    benchmark_type = get_benchmark_from_filename(file, metadata)
    df = pd.read_excel(file)
    df = df[:2]

    for model_name in metadata['supported_models']:
        print(f"Storing answers for {benchmark_type} benchmark with model {model_name}")
        run_benchmark_store_answers(model_name, benchmark_type, df)

# Step 2: Calculate scores for all stored answers
for model_name in metadata['supported_models']:
    for benchmark_type in metadata['benchmark_types'].keys():
        average_score = run_benchmark_get_scores(model_name, benchmark_type)
        results.loc[model_name, benchmark_type] = average_score

# Save the results after calculating all scores
print("\nAverage Scores:\n", results)
results.to_excel(results_file)

