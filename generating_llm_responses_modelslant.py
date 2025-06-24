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

# Define model providers
MODEL_PROVIDERS = {
    'o3-mini': 'OpenAI',
    'gpt-4.5-preview': 'OpenAI',
    'gemma-3-27b-it': 'Google',
    'claude-3.7': 'Anthropic',
    'deepseek-r1': 'DeepSeek',
    'deepseek-v3': 'DeepSeek',
    'llama-3.1-70B': 'Meta',
    'llama-3.3-70B': 'Meta',
    'llama-4-maverick': 'Meta',
    'llama-4-scout': 'Meta'
}

# Load questions and get the current max ID and call number
def load_data_and_metadata():
    # Check for updated file first, then fall back to original
    csv_path = os.path.join(DATA_PATH, 'modelslant_data_updated.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join(DATA_PATH, 'modelslant_data.csv')
        print(f"Using original data file: {csv_path}")
    else:
        print(f"Using updated data file: {csv_path}")
    
    # Load the data in chunks to handle large files
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=10000):
        chunks.append(chunk)
    
    # Combine all chunks
    full_data = pd.concat(chunks)
    
    # Extract unique prompts
    unique_prompts = list(set(full_data['Prompt'].tolist()))
    print(f"Loaded {len(unique_prompts)} unique prompts from CSV")
    
    # Get max ID
    max_id = full_data['id'].max() if 'id' in full_data.columns else 0
    
    # Create a dictionary of model-question pairs to their maximum call number
    model_question_to_call_num = {}
    if 'Call_number' in full_data.columns and 'Model' in full_data.columns:
        for _, row in full_data.iterrows():
            key = (row['Model'], row['Prompt'])
            call_num = row['Call_number']
            model_question_to_call_num[key] = max(model_question_to_call_num.get(key, 0), call_num)
    
    return unique_prompts, full_data, max_id, model_question_to_call_num

# Load prompts and metadata
questions, original_data, current_max_id, model_question_to_call_num = load_data_and_metadata()

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

def generate_responses(questions, generation_function, model_name, n_responses=10, generate_new_responses=True):
    """
    Generate responses for each question and track metadata.
    """
    output_path = os.path.join(TEMP_PATH, f"{model_name}__modelslant_responses.json")
    provider = MODEL_PROVIDERS.get(model_name, "Unknown")
    print(f"Generating responses for: {model_name} (Provider: {provider})")
    
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
    
    # For tracking new generated responses with metadata
    new_responses_data = []
    global current_max_id, model_question_to_call_num
    
    # Process each question
    for question in tqdm(questions, desc=f"Generating for {model_name}", position=0, leave=True):
        if question not in responses:
            responses[question] = []
        elif isinstance(responses[question], str):
            responses[question] = [responses[question]]
            
        existing_count = len(responses[question])
        
        # Determine how many responses to generate
        if generate_new_responses:
            responses_needed = n_responses
        else:
            responses_needed = max(0, n_responses - existing_count)
            if responses_needed == 0:
                continue
        
        # Get the current call number for this model-question pair
        model_question_key = (model_name, question)
        current_call_num = model_question_to_call_num.get(model_question_key, 0)
        
        # Generate responses
        for _ in range(responses_needed):
            try:
                response_text = generation_function(question)
                if isinstance(response_text, list):
                    response_text = ' '.join(response_text)
                
                # Add to responses dict for checkpoint saving
                responses[question].append(response_text)
                
                # Increment metadata
                current_max_id += 1
                current_call_num += 1
                model_question_to_call_num[model_question_key] = current_call_num
                
                # Add to new responses data with metadata
                new_responses_data.append({
                    'id': current_max_id,
                    'Call_number': current_call_num,
                    'Prompt': question,
                    'Response': response_text,
                    'Model': model_name,
                    'Provider': provider,
                    'Topic': None  # This will be filled later based on the prompt
                })
                
                # Save checkpoint every 5 responses
                if len(new_responses_data) % 5 == 0:
                    with open(output_path, 'w') as f:
                        json.dump(responses, f, indent=2)
                
            except Exception as e:
                tqdm.write(f"\n{model_name} Error: {str(e)}")
        
        # Save after each question is complete
        with open(output_path, 'w') as f:
            json.dump(responses, f, indent=2)
    
    return new_responses_data

def process_model(model_info, generate_new_responses=True, n_responses=10):
    """
    Process a single model's responses. Used for parallel execution.
    """
    model_name, model_path, is_together_model = model_info
    
    try:
        if is_together_model:
            generation_function = lambda x: generate_together_response(x, model_path)
        elif model_name in ['gemma-3-27b-it']:  # Google models
            generation_function = lambda x: generate_google_response(x, model_name)
        elif model_name in ['claude-3.7']:  # Anthropic models
            generation_function = lambda x: generate_anthropic_response(x, 'claude-3-7-sonnet-20250219')
        else:  # OpenAI models
            generation_function = lambda x: generate_openai_response(x, model_name)
            
        new_responses = generate_responses(
            questions=questions,
            generation_function=generation_function,
            model_name=model_name,
            n_responses=n_responses,
            generate_new_responses=generate_new_responses
        )
        return model_name, new_responses, True, None
        
    except Exception as e:
        return model_name, [], False, str(e)

def clean_deepseek_response(response):
    """Remove thinking tags and their content from deepseek responses"""
    import re
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate LLM responses and add to modelslant data')
    parser.add_argument('--fill-missing', action='store_true', help='Only generate responses up to required number per question')
    parser.add_argument('--num-responses', type=int, default=10, help='Number of responses to generate per prompt (default: 10)')
    parser.add_argument('--output-file', type=str, default='modelslant_data_updated.csv', help='Name of the output CSV file')
    args = parser.parse_args()
    
    # Define models
    oai_models = []#['o3-mini', 'gpt-4.5-preview']
    gdp_models = []#['gemma-3-27b-it']
    anthropic_models = []#['claude-3.7']
    together_models = {
        # 'deepseek-r1': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'deepseek-v3': 'deepseek-ai/DeepSeek-V3',
        # 'llama-3.1-70B': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
        # 'llama-3.3-70B': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
        # 'llama-4-maverick': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
        # 'llama-4-scout': 'meta-llama/Llama-4-Scout-17B-16E-Instruct'
    }
    
    # Prepare model information for parallel processing
    model_info = (
        [(name, name, False) for name in oai_models] +  # OpenAI models
        [(name, name, False) for name in gdp_models] +  # Google models
        [(name, name, False) for name in anthropic_models] +  # Anthropic models
        [(name, path, True) for name, path in together_models.items()]  # Together models
    )

    all_new_responses = []
    
    # Process models in parallel
    print(f"Starting parallel processing of {len(model_info)} models...")
    with ThreadPoolExecutor(max_workers=len(model_info)) as executor:
        future_to_model = {
            executor.submit(process_model, info, not args.fill_missing, args.num_responses): info[0]
            for info in model_info
        }
        
        # Process results as they complete
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                model_name, new_responses, success, error = future.result()
                if success:
                    print(f"✓ Completed processing for {model_name}, generated {len(new_responses)} responses")
                    all_new_responses.extend(new_responses)
                else:
                    print(f"✗ Failed processing for {model_name}: {error}")
            except Exception as e:
                print(f"✗ Failed processing for {model_name}: {str(e)}")

    # Create DataFrame from new responses
    if all_new_responses:
        new_responses_df = pd.DataFrame(all_new_responses)
        
        # Transfer topic information from original data
        prompt_to_topic = {}
        for _, row in original_data.iterrows():
            if 'Prompt' in row and 'Topic' in row:
                prompt_to_topic[row['Prompt']] = row['Topic']
        
        # Fill in topics based on prompts
        new_responses_df['Topic'] = new_responses_df['Prompt'].map(prompt_to_topic)
        
        # Combine with original data
        combined_data = pd.concat([original_data, new_responses_df], ignore_index=True)
        
        # Save the combined data
        output_file = os.path.join(DATA_PATH, args.output_file)
        combined_data.to_csv(output_file, index=False)
        print(f"Added {len(new_responses_df)} new responses to the dataset")
        print(f"Final dataset has {len(combined_data)} rows")
        print(f"Saved to {output_file}")
    else:
        print("No new responses were generated")
        
if __name__ == "__main__":
    main()