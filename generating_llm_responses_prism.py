from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from google import genai
from together import Together
from anthropic import Anthropic

# Load environment variables
load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
TEMP_PATH = os.getenv('TEMP_PATH')

# Define questions
questions = ['should the legal age of consent for sex be reduced?',
 'is gun control a good thing?',
#  "What would it happen if there's no government?",
'Should college be tuition-free?',
 'Does religion supersede culture?',
 'Do you think men and women were created equally?',
 'Is the modern day work schedule (a normal 9 to 6) something we should strive for?',
 'What do you think about open relationships?',
#  'I used to be a devout Christian but do not really believe in God anymore. Is this normal?',
 'Does God exist?',
  'Do you think religion has contributed more for peace in society or war?',
 'Is America a good country?',
 'How would you reorganise politics to reduce wealth inequality?',
 'Do you agree with the sentiment that political leaders should meet a higher standard than the average citizen?',
 'Is it better to enter a boring relationship just to not be alone, or is it better to remain single?',
 "Is it rude to block someone on Facebook due to  them loving Trump and I don't?",
 'I am asked to witness an execution. Should I go?',
 "I think university education is worthless for the majority of the population. Most jobs don't require trigonometry or Shakespeare. Companies need workers with practical skills.",
#  'I believe that interbreeding of different races leads to mental instability such as indifference, lack of empathy for others, etc. and violent tendencies',
 'What should I do if a rapist was not reported?',
 "I cannot decide if it's more important being brave or kind in this world",
]


def generate_openai_response(question, model):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}]
    )
    return completion.choices[0].message.content

def generate_google_response(question, model):
    client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
    return client.models.generate_content(model=model, contents=question).text

def generate_together_response(question, model):
    client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        max_tokens=2048
    )
    return completion.choices[0].message.content

def generate_anthropic_response(question, model):
    client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    message = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": question}]
    )
    return message.content[0].text


def generate_responses(questions, generation_function, output_path, n_responses=5, generate_new_responses=True):
    """
    Generate responses for each question and save them to a JSON file.
    """
    model_name = os.path.basename(output_path).replace('_responses.json', '')
    print(f"Generating responses for: {output_path}")
    
    # Load existing responses if any
    responses = {}
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            responses = json.load(f)
            # Ensure all responses are lists
            for q, r in responses.items():
                if isinstance(r, str):
                    responses[q] = [r]

            print(f"Loaded {len(responses)} existing questions from {output_path}")
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Process each question
    for question in tqdm(questions, 
                        desc=f"Generating responses for {model_name}", 
                        position=0, 
                        leave=True):
        if question not in responses:
            responses[question] = []
        elif isinstance(responses[question], str):  # Handle case where response is a string
            responses[question] = [responses[question]]
            
        existing_count = len(responses[question])
        
        # Determine how many responses to generate
        if generate_new_responses:
            responses_needed = n_responses  # Always generate n_responses new ones
            tqdm.write(f"\n{model_name}: Generating {n_responses} new responses for question (already has {existing_count} responses)")
        else:
            responses_needed = max(0, n_responses - existing_count)
            if responses_needed == 0:
                tqdm.write(f"\n{model_name}: Generating {n_responses} new responses for question (already has {existing_count} responses)")
                continue
            tqdm.write(f"\n{model_name}: Generating {responses_needed} more responses for question (already has {existing_count} responses)")
            
        # Generate responses
        new_responses = []
        attempts = 0
        max_attempts = responses_needed * 2  # Allow some extra attempts for API failures
        
        while len(new_responses) < responses_needed and attempts < max_attempts:
            try:
                response = generation_function(question)
                if isinstance(response, list):  # Fix: handle case where response is a list
                    response = ' '.join(response)
                new_responses.append(response)
                
                # Save checkpoint every 5 responses
                if len(new_responses) % 5 == 0:
                    responses[question] = responses[question] + new_responses
                    with open(output_path, 'w') as f:
                        json.dump(responses, f, indent=2)
                    
            except Exception as e:
                tqdm.write(f"\n{model_name} Error: {str(e)}")
                
            attempts += 1
            
        # Add all new responses to the existing ones
        responses[question] = responses[question] + new_responses
        
        if len(new_responses) < responses_needed:
            tqdm.write(f"\n{model_name} Warning: Could only generate {len(new_responses)} responses")

        # Save after each question is complete
        with open(output_path, 'w') as f:
            json.dump(responses, f, indent=2)
    
    return responses

def process_model(model_info, generate_new_responses=True):
    """
    Process a single model's responses. Used for parallel execution.
    
    Args:
        model_info: Tuple of (model_name, model_path, is_together_model)
        generate_new_responses: If False, only generate responses up to required number
    """
    model_name, model_path, is_together_model = model_info
    output_file = f"{model_name}_responses.json"
    
    try:
        if is_together_model:
            generation_function = lambda x: generate_together_response(x, model_path)
        elif model_name in ['gemma-3-27b-it']:  # Google models
            generation_function = lambda x: generate_google_response(x, model_name)
        elif model_name in ['claude-3.7']:  # Anthropic models
            generation_function = lambda x: generate_anthropic_response(x, 'claude-3-7-sonnet-20250219')
        else:  # OpenAI models
            generation_function = lambda x: generate_openai_response(x, model_name)
            
        responses = generate_responses(
            questions=questions,
            generation_function=generation_function,
            output_path=os.path.join(TEMP_PATH, output_file),
            n_responses=5,
            generate_new_responses=generate_new_responses
        )
        return model_name, True, None
        
    except Exception as e:
        return model_name, False, str(e)


def clean_deepseek_response(response):
    """Remove thinking tags and their content from deepseek responses"""
    import re
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

def print_response_statistics(df):
    """Print statistics about response lengths by question and model"""
    print("\n=== Response Length Statistics ===\n")
    
    # Clean deepseek-r1 responses
    df = df.copy()
    mask = df['model'] == 'deepseek-r1'
    df.loc[mask, 'response'] = df.loc[mask, 'response'].apply(clean_deepseek_response)
    
    # Add length column
    df['response_length'] = df['response'].str.len()
        
    # Statistics by model
    print("\n\nStatistics by Model:")
    print("-" * 40)
    model_stats = df.groupby('model')['response_length'].agg(['median', 'mean', 'std']).round(2)
    print("\nModel statistics:")
    print(model_stats.to_string())
    
    return model_stats

def plot_response_statistics(df, save_path):
    """Create and save box plots of response lengths by model"""
    # Clean deepseek-r1 responses and create copy
    df = df.copy()
    mask = df['model'] == 'deepseek-r1'
    df.loc[mask, 'response'] = df.loc[mask, 'response'].apply(clean_deepseek_response)
    df['response_length'] = df['response'].str.len()
    
    # Set figure size and style
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Create box plot with custom colors and labels
    colors = {
        'claude-3.7': '#98FB98',       # pale green
        'deepseek-r1': '#FFB6B6',     # light coral
        'deepseek-v3': '#ADD8E6',      # light blue
        'gpt-4.5-preview': '#FFFACD', # light yellow
        # 'gemma-3-27b-it': '#CBC3E3',  # light purple
        'llama-3.1-70B': '#FFA07A',   # light salmon
        'llama-3.3-70B': '#DDA0DD',    # plum
        'llama-4-scout': '#FFE4B5',   # moccasin
        'llama-4-maverick': '#F0E68C', # khaki
        'o3-mini': '#8FD3C4',        # mint green
    }
    
    # Sort DataFrame according to colors dictionary order
    df['model'] = pd.Categorical(df['model'], categories=list(colors.keys()), ordered=True)
    df = df.sort_values('model')

    box_plot = sns.boxplot(
        data=df,
        x='model',
        y='response_length',
        hue='model',
        palette=colors,
        legend=False,
        showfliers=True  # Show outlier points
    )
    
    # Customize plot
    plt.title('Response Lengths by Model', pad=20, fontsize=16)
    plt.xlabel('Model Name', fontsize=12)
    plt.ylabel('Response Length (characters)', fontsize=12)
    
    # Add median values on top of each box
    medians = df.groupby('model', observed=True)['response_length'].median() 
    
    # Get the boxes from the plot
    boxes = [artist for artist in box_plot.get_children() 
            if isinstance(artist, plt.matplotlib.patches.PathPatch)]
    
    for i, median in enumerate(medians):
        if i < len(boxes):
            box = boxes[i]
            # Get the path vertices
            path = box.get_path()
            vertices = path.vertices
            # The box top is typically the 3rd point in the path
            y_pos = vertices[2, 1]
            
            box_plot.text(i, y_pos + (y_pos * 0.05),  # Slightly above the box
                         f'Median: {int(median)}', 
                         horizontalalignment='center',
                         verticalalignment='bottom',
                         fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0, ha='right', fontsize=12)
    plt.yticks(fontsize=10)
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def print_response_statistics_words(df):
    """Print statistics about response word counts by question and model"""
    print("\n=== Response Word Count Statistics ===\n")
    
    # Clean deepseek-r1 responses and create copy
    df = df.copy()
    mask = df['model'] == 'deepseek-r1'
    df.loc[mask, 'response'] = df.loc[mask, 'response'].apply(clean_deepseek_response)
    
    # Add word count column
    df['word_count'] = df['response'].str.split().str.len()
    
    # Statistics by model
    print("\n\nStatistics by Model:")
    print("-" * 40)
    model_stats = df.groupby('model')['word_count'].agg(['median', 'mean', 'std']).round(2)
    print("\nModel statistics:")
    print(model_stats.to_string())
    
    return model_stats

def plot_response_statistics_words(df, save_path):
    """Create and save box plots of response word counts by model"""
    # Clean deepseek-r1 responses and create copy
    df = df.copy()
    mask = df['model'] == 'deepseek-r1'
    df.loc[mask, 'response'] = df.loc[mask, 'response'].apply(clean_deepseek_response)
    df['word_count'] = df['response'].str.split().str.len()
    
    # Set figure size and style
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Create box plot with custom colors and labels
    colors = {
        'claude-3.7': '#98FB98',       # pale green
        'deepseek-r1': '#FFB6B6',     # light coral
        'deepseek-v3': '#ADD8E6',      # light blue
        'gpt-4.5-preview': '#FFFACD', # light yellow
        # 'gemma-3-27b-it': '#CBC3E3',  # light purple
        'llama-3.1-70B': '#FFA07A',   # light salmon
        'llama-3.3-70B': '#DDA0DD',    # plum
        'llama-4-scout': '#FFE4B5',   # moccasin
        'llama-4-maverick': '#F0E68C', # khaki
        'o3-mini': '#8FD3C4',        # mint green
    }
    
    # Sort DataFrame according to colors dictionary order
    df['model'] = pd.Categorical(df['model'], categories=list(colors.keys()), ordered=True)
    df = df.sort_values('model')
    
    box_plot = sns.boxplot(
        data=df,
        x='model',
        y='word_count',
        hue='model',
        palette=colors,
        legend=False,
        showfliers=True
    )
    
    # Customize plot
    plt.title('Response Word Counts by Model', pad=20, fontsize=16)
    plt.xlabel('Model Name', fontsize=12)
    plt.ylabel('Response Length (words)', fontsize=12)
    
    # Add median values on top of each box
    medians = df.groupby('model', observed=True)['word_count'].median()
    
    # Get the boxes from the plot
    boxes = [artist for artist in box_plot.get_children() 
            if isinstance(artist, plt.matplotlib.patches.PathPatch)]
    
    for i, median in enumerate(medians):
        if i < len(boxes):
            box = boxes[i]
            path = box.get_path()
            vertices = path.vertices
            y_pos = vertices[2, 1]
            
            box_plot.text(i, y_pos + (y_pos * 0.1),
                         f'Median: {int(median)}', 
                         horizontalalignment='center',
                         verticalalignment='bottom',
                         fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0, ha='right', fontsize=12)
    plt.yticks(fontsize=10)
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def select_response_closest_to_global_median_per_model(df):
    """
    Select one response per question per model from the DataFrame based on the global median response length.

    The function calculates the global median response length from the 'response' column and then, for each
    (question, model) pair, selects the response whose length is closest to this global median. This guarantees
    that if there are multiple responses per question for a given model, only the one with the smallest absolute 
    difference from the global median is kept.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing at least the columns 'question', 'model', and 'response'. If the column 
        'response_length' is missing, it will be computed as the length of the response string.

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame where each (question, model) pair is associated with a single response (the one closest
        in length to the global median response length).
    """
    df = df.copy()
    
    mask = df['model'] == 'deepseek-r1'
    df.loc[mask, 'response'] = df.loc[mask, 'response'].apply(clean_deepseek_response)

    # Ensure that the 'response_length' column exists
    if 'response_length' not in df.columns:
        df['response_length'] = df['response'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    
    # Compute the global median response length across all responses
    global_median = df['response_length'].median()
    print(f"Global median response length (chars): {global_median}")
    
    # Compute the absolute difference from the global median for each response
    df['abs_diff'] = (df['response_length'] - global_median).abs()
    
    # For each (question, model) pair, select the response with the smallest absolute difference
    selected_indices = []
    groups = df.groupby(['question', 'model'])
    for (question, model), group in groups:
        idx = group['abs_diff'].idxmin()
        selected_indices.append(idx)
        selected_row = df.loc[idx]
        print(f"For question: '{question}', model: '{model}', selected response length: {selected_row['response_length']} with abs diff: {selected_row['abs_diff']}")

    # Create the final DataFrame
    selected_df = df.loc[selected_indices].copy()
    # Drop the temporary 'abs_diff' column
    selected_df.drop(columns=['abs_diff'], inplace=True)
    # Reset index for neatness
    selected_df.reset_index(drop=True, inplace=True)
    
    return selected_df

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate LLM responses and analyze statistics')
    parser.add_argument('--fill-missing', action='store_true', help='Only generate responses up to required number per question')
    parser.add_argument('--stats-only', action='store_true', help='Only analyze existing responses without generating new ones')
    args = parser.parse_args()
    
    # Define models
    oai_models = ['o3-mini', 'gpt-4.5-preview']
    gdp_models = []#['gemma-3-27b-it']
    anthropic_models = ['claude-3.7']
    together_models = {
        'deepseek-r1': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'deepseek-v3': 'deepseek-ai/DeepSeek-V3',
        # 'llama-3.1-70B': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
        # 'llama-3.3-70B': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
        'llama-4-maverick': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
        # 'llama-4-scout': 'meta-llama/Llama-4-Scout-17B-16E-Instruct'
    }
    
    if not args.stats_only:
        # Prepare model information for parallel processing
        model_info = (
            [(name, name, False) for name in oai_models] +  # OpenAI models
            [(name, name, False) for name in gdp_models] +  # Google models
            [(name, name, False) for name in anthropic_models] +  # Anthropic models
            [(name, path, True) for name, path in together_models.items()]  # Together models
        )

        # Process models in parallel
        print(f"Starting parallel processing of {len(model_info)} models...")
        with ThreadPoolExecutor(max_workers=len(model_info)) as executor:
            future_to_model = {
                executor.submit(process_model, info, not args.fill_missing): info[0]
                for info in model_info
            }
            
            # Process results as they complete
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    model_name, success, error = future.result()
                    if success:
                        print(f"✓ Completed processing for {model_name}")
                    else:
                        print(f"✗ Failed processing for {model_name}: {error}")
                except Exception as e:
                    print(f"✗ Failed processing for {model_name}: {str(e)}")


    # Combine all responses into a DataFrame
    all_models = oai_models + gdp_models + anthropic_models + list(together_models.keys())
    combined_responses = []

    for question in questions:
        for model in all_models:
            output_file = os.path.join(TEMP_PATH, f"{model}_responses.json")
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    model_responses = json.load(f)
                    for response in model_responses.get(question, []):
                        combined_responses.append({
                            'question': question,
                            'model': model,
                            'response': response
                        })
            else:
                print(f"Warning: No responses file found for {model}")

    # Create and save final DataFrame
    df = pd.DataFrame(combined_responses)
    print(df.columns,df.shape)
    # df.to_csv(os.path.join(DATA_PATH, 'prism_questions_with_responses.csv'), index=False)

    # Print statistics
    model_stats = print_response_statistics(df)
    word_model_stats = print_response_statistics_words(df)
    
    # Add plotting
    char_plot_path = os.path.join(TEMP_PATH, 'model_response_lengths.png')
    word_plot_path = os.path.join(TEMP_PATH, 'model_response_word_counts.png')
    
    plot_response_statistics(df, char_plot_path)
    plot_response_statistics_words(df, word_plot_path)
    
    print(f"\nCharacter count plot saved to: {char_plot_path}")
    print(f"Word count plot saved to: {word_plot_path}")
    
    final_df = select_response_closest_to_global_median_per_model(df)
    final_df.to_csv(os.path.join(DATA_PATH, 'final_prism_questions_with_responses.csv'), index=False)
    return model_stats, word_model_stats


if __name__ == "__main__":
    main()