import argparse
import ast
import json
import numpy as np
import pandas as pd
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import nltk

def load_model_and_tokenizer(model_name="tals/albert-xlarge-vitaminc"):
    """Loads the tokenizer and model for the specified model name.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def NLI(text_a, text_b, model, tokenizer):
    """Predicts the relationship between a claim and evidence. 

    Args:
        text_a (str): The evidence statement.
        text_b (str): The claim statement.

    Returns:
        str: The predicted relationship label ("SUPPORTS", "REFUTES", or "NOT ENOUGH INFO").
        float: The confidence score of the prediction.
    """

    # Tokenize the input
    inputs = tokenizer(
        text_a, text_b,
        return_tensors='pt',  # Return PyTorch tensors
        padding=True,         # Pad to the longest sequence in the batch
        truncation=True       # Truncate to the model's max length
    )

    # Make predictions
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Convert logits to probabilities
    probabilities = F.softmax(logits, dim=1)
    
    # Get the predicted class and its score
    predicted_class = torch.argmax(probabilities, dim=1).item()
    predicted_score = probabilities[0, predicted_class].item()

    # Label mapping
    label_map = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}  # Updated label mapping

    return label_map[predicted_class], predicted_score  # Return label and score


def run_nli_on_mnli(model, tokenizer, results_dir, n=200, seed=0):
    splits = {'train': 'data/train-00000-of-00001.parquet', 'validation_matched': 'data/validation_matched-00000-of-00001.parquet', 'validation_mismatched': 'data/validation_mismatched-00000-of-00001.parquet'}
    df_mnli = pd.read_parquet("hf://datasets/nyu-mll/multi_nli/" + splits["validation_mismatched"])

    label_mapping = {0: "SUPPORTS", 1: "NOT ENOUGH INFO", 2: "REFUTES"}

    sample_df_mnli = df_mnli.sample(n, random_state=seed)

    results = {}
    correct = 0
    for index, row in tqdm.tqdm(sample_df_mnli.iterrows(), total=n):
        premise = row['premise']
        hypothesis = row['hypothesis']
        label = label_mapping[row["label"]]
        prediction, score = NLI(premise, hypothesis, model, tokenizer)
        results[index] = {
            "promptID": row["promptID"],
            "pairID": row["pairID"],
            "premise": premise,
            "hypothesis": hypothesis,
            "prediction": prediction,
            "score": score,
            "label": label
        }
        if prediction == label:
            correct += 1
    print(f"Accuracy on MNLI sample size n={n} is {round(correct/n,3)*100}%")

    with open(f"{results_dir}/MNLI_predictions_seed-{seed}_n-{n}.json", 'w') as f:
        json.dump(results, f, indent=2)
        
        
def tokenize_sentences(text):
    # Use NLTK's sent_tokenize to split the text into sentences
    nltk.download('punkt', quiet=True)  # Ensure 'punkt' tokenizer is downloaded
    sentences = nltk.sent_tokenize(text)
    return sentences

def find_span_indices(string, substring):
    start_index = string.find(substring)
    if start_index == -1:
        return None  # Return None if the substring is not found
    end_index = start_index + len(substring)
    return (start_index, end_index)

def load_vp_data(data_dir, n=None, seed=None, fname='questions_and_human_perspectives_with_responses.csv'):
    """Loads the Value Prism data from a specified directory.

    Args:
        data_dir (str): The directory where the data file is located.
        fname (str): The name of the data file to load.
        n (int, optional): Number of samples to return. If None, return all data.
        seed (int, optional): Random seed for sampling.

    Returns:
        DataFrame: A pandas DataFrame containing the loaded data.
    """
    file_path = f'{data_dir}/{fname}'
    vp_data = pd.read_csv(file_path)
    vp_data = vp_data[vp_data.source=='valueprism']
    
    if n is not None:
        vp_data = vp_data.sample(n, random_state=seed)
    
    return vp_data


def run_nli_on_vp(model, tokenizer, data_dir, results_dir, n=None, seed=None, llm='gpt-4o-mini'):
    vp = load_vp_data(data_dir, n=n, seed=seed)

    results = {}

    for id, row in tqdm.tqdm(vp.iterrows(), total=n):
        question = row.question
        perspectives = ast.literal_eval(row.perspectives)
        explanations = [p.split("Explanation: ")[-1] for p in perspectives]
        model_response = row[llm]
        results[question] = {
            "model_response": model_response,
            "values": {},
            "avg": 0
        }
        S = tokenize_sentences(model_response)
        presence = []
        for e in explanations:
            results[question]["values"][e] = {
                "labels": [], 
                "scores": [],
                "spans": []
            }
            for si in S:
                label, score = NLI(si, e, model, tokenizer)
                span = find_span_indices(model_response, si)
                results[question]["values"][e]["labels"].append(label)
                results[question]["values"][e]["scores"].append(score)
                results[question]["values"][e]["spans"].append(span) 
            presence.append(1 if "SUPPORTS" in results[question]["values"][e]["labels"] else 0)
        results[question]["avg"] = np.mean(presence)
    
    with open(f"{results_dir}/NLI_VP_results_{llm}_seed-{seed}_n-{n}.json", 'w') as f:
        json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run NLI tasks on specified datasets.")
    parser.add_argument('task', choices=['mnli', 'vp'], help="Task to run: 'mnli' or 'vp'.")
    parser.add_argument('--data_dir', type=str, default='./data', help="Path to the data directory.")
    parser.add_argument('--results_dir', type=str, default='./data/results', help="Path to results directory.")
    parser.add_argument('--n', type=int, default=None, help="Number of data points to sample.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for sampling.")
    parser.add_argument('--llm', type=str, default='gpt-4o-mini', help="LLM to test (only for VP).")

    args = parser.parse_args()
    
    model, tokenizer = load_model_and_tokenizer()
    if args.task == 'mnli':
        run_nli_on_mnli(model, tokenizer, args.results_dir, args.n, args.seed)
    elif args.task == 'vp':
        run_nli_on_vp(model, tokenizer, args.data_dir, args.results_dir, args.n, args.seed, args.llm)

if __name__ == "__main__":
    main()