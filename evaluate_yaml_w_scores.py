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


# YAML file Metadata:

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

def handle_qa(question, actual_answer, model):
    predicted_answer = get_answer_from_local_ollama(model, question)
    # score = min(max(int(float(get_evaluation_score(question, predicted_answer, actual_answer))), 0), 100)
    score = (0.25 * int(float(get_evaluation_score(question, actual_answer, predicted_answer)))) + calculate_bleu_score(actual_answer, predicted_answer) + calculate_rouge_score(actual_answer, predicted_answer) + calculate_levenshtein_score(actual_answer, predicted_answer)
    return predicted_answer, score

def handle_multiple_choice(question, options, correct_option, model):
    predicted_option = get_model_answer_multiple_options(question, options=options, model=model, dstype='mc')
    # print("PPPPredicted:", predicted_option) #duzgun
    score = compare_answers(actual_answer=correct_option, predicted_answer=predicted_option)
    return predicted_option, score

def handle_context_qa(question, context, actual_answer, model):
    predicted_answer = get_answer_from_local_ollama_context(model, question, context)
    score = (0.25 * int(float(get_evaluation_score_context(question, actual_answer, predicted_answer)))) + calculate_bleu_score(actual_answer, predicted_answer) + calculate_rouge_score(actual_answer, predicted_answer) + calculate_levenshtein_score(actual_answer, predicted_answer)
    return predicted_answer, score

def handle_topic_classification(question, topic_options, correct_topic, model):
    predicted_topic = get_model_answer_multiple_options(question=question, model=model, options=topic_options, dstype='tc')
    print(predicted_topic)
    score = compare_answers(actual_answer=correct_topic, predicted_answer=predicted_topic)
    return predicted_topic, score

def handle_arc(question, options, correct_answer, model):
    predicted_option = get_model_answer_multiple_options(question, options=options, model=model, dstype='arc')
    # print(predicted_option)
    score = compare_answers(actual_answer=correct_answer, predicted_answer=predicted_option)
    return predicted_option, score



# Function to handle QA benchmarks
def handle_qa_benchmark(df, model_name):
    scores = []
    output_data = []

    for index, row in df.iterrows():
        question = row['Sual']
        actual_answer = row['Cavab']

        # Get prediction and score
        # predicted_answer = get_answer_from_local_ollama(model_name, question)
        predicted_answer, score = handle_qa(question, actual_answer, model_name)

        scores.append(score)
        output_data.append([question, actual_answer, predicted_answer, score])

    return scores, output_data

# Function to handle Reshad benchmarks (Topic Classification)
def handle_topic_classification_benchmark(df, model_name):
    scores = []
    output_data = []

    for index, row in df.iterrows():
        question = row['text']
        options = row['options']
        correct_option = row['answer']

        # Get prediction and score
        # predicted_option = get_model_answer_multiple_options(question, options=options, model=model_name, dstype='tc')
        predicted_option, score = handle_topic_classification(question, options, correct_option, model_name)

        scores.append(score)
        output_data.append([question, correct_option, predicted_option, score])

    return scores, output_data

# Function to handle ContextQA benchmarks
def handle_context_qa_benchmark(df, model_name):
    scores = []
    output_data = []

    for index, row in df.iterrows():
        question = row['question']
        context = row['context']
        actual_answer = row['answer']

        # Get prediction and score
        # predicted_answer = get_answer_from_local_ollama_context(model_name, question, context)
        predicted_answer, score = handle_context_qa(question, context, actual_answer, model_name)

        scores.append(score)
        output_data.append([question, actual_answer, predicted_answer, score])

    return scores, output_data

# Function to handle Arzuman (Multiple Choice) benchmarks
def handle_multiple_choice_benchmark(df, model_name):
    scores = []
    output_data = []

    for index, row in df.iterrows():
        question = row['text']
        options = row['options']
        correct_option = row['answer']

        # Get prediction and score
        # predicted_option = get_model_answer_multiple_options(question, options=options, model=model_name, dstype='mc')
        # print('pppRedcited_option', predicted_option)
        predicted_option, score = handle_multiple_choice(question, options, correct_option, model_name)

        scores.append(score)
        output_data.append([question, correct_option, predicted_option, score])

    return scores, output_data

# Function to handle ARC benchmarks
def handle_arc_benchmark(df, model_name):
    scores = []
    output_data = []

    for index, row in df.iterrows():
        question = row['Azerbaijani_q']
        options_txt = row['choices']
        array = pd.array
        options_dict = eval(options_txt)
        options = options_dict['az_choices'].tolist()
        correct_answer = row['answerKey']

        # Get prediction and score
        # predicted_option = get_model_answer_multiple_options(question, options=options, model=model_name, dstype='arc')
        predicted_option, score = handle_arc(question, options, correct_answer, model_name)

        scores.append(score)
        output_data.append([question, correct_answer, predicted_option, score])

    return scores, output_data


# Main function to run benchmarks and save results
def run_benchmark(model_name, benchmark_type, df, results):
    if benchmark_type == "QA":
        scores, output_data = handle_qa_benchmark(df, model_name)
    elif benchmark_type == "Reshad":
        scores, output_data = handle_topic_classification_benchmark(df, model_name)
    elif benchmark_type == "ContextQA":
        scores, output_data = handle_context_qa_benchmark(df, model_name)
    elif benchmark_type == "Arzuman":
        scores, output_data = handle_multiple_choice_benchmark(df, model_name)
    elif benchmark_type == "ARC":
        scores, output_data = handle_arc_benchmark(df, model_name)
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")

    # Calculate and store average score
    if scores:
        average_score = sum(scores) / len(scores)
        results.loc[model_name, benchmark_type] = average_score

        # Create a DataFrame to save individual row scores along with predictions
        output_df = pd.DataFrame(output_data, columns=['Question', 'Correct Answer', 'Predicted Answer', 'Score'])

        # Generate the output filename
        output_filename = f"{benchmark_type}_{model_name}.xlsx"

        # Write the individual scores and predictions to the output Excel file
        output_df.to_excel(output_filename, index=False)


# Example usage:

results = pd.DataFrame(columns=metadata['benchmark_types'].keys(), index=metadata['supported_models'])

for file in dataset_files:
    benchmark_type = get_benchmark_from_filename(file, metadata)
    print(f"Running {benchmark_type} benchmark for file: {file}")

    df = pd.read_excel(file)
    # df = df[:2]  # For testing purposes, process only the first 2 rows

    for model_name in metadata['supported_models']:
        print(f"Running {benchmark_type} for model {model_name}")
        run_benchmark(model_name, benchmark_type, df, results)

print("\nAverage Scores:\n", results)
results.to_excel(results_file)
