
from pydantic import BaseModel

import openai, os, json
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class EntailmentMatch(BaseModel):
    text: str
    match_type: str  # "direct", "paraphrase", or "contextual"
    confidence: int  # 0-10 score
    explanation: str  # Why this is a match

class EntailmentStep(BaseModel):
    step_number: int
    concept: str  # The concept from the opinion being analyzed
    analysis: str  # The reasoning process
    matches: list[EntailmentMatch]

class EntailmentAnalysis(BaseModel):
    steps: list[EntailmentStep]
    final_matches: list[str]  # The best, most confident matches
    coverage_score: int  # 0-10 how well the opinion is covered

def entailment_from_gpt_json(question: str, response: str, opinion: str, model='gpt-4o-mini'):
    """
    Find exact text matches between rich text and opinion using GPT-4.
    """
    system_prompt = f"""Task: Precise Text Entailment Analysis. Find and evaluate text in the Response that represents concepts from the Opinion.

Follow these specific steps:
1. Break down the Opinion into key concepts
2. For each concept:
   - Search for direct text matches, this includes single words like "yes" or "no"
   - Identify paraphrased representations
   - Look for contextual/implicit matches
   - Copy the **exact text** in the Response that matches the concept in the Opinion. Copy the text from the response, not the opinion.

3. Evaluate matches by:
   - Precision: How exactly does it match?
   - Context: Is the meaning preserved?
   - Completeness: Is the full concept captured?

4. Score coverage from 0-10 where:
   - 0: No valid matches found
   - 1-3: Few weak/partial matches
   - 4-6: Some good matches but incomplete
   - 7-9: Strong matches for most concepts
   - 10: Complete, precise matches for all concepts

Important:
- Prioritize precision over quantity
- Consider context to avoid false matches
- Explain reasoning for each match
- Always copy the exact text from the response that matches the concept
"""

    prompt = f"""Context question: {question}
Opinion: {opinion}
Response: {response}

Analyze step-by-step following the instructions to find and evaluate all relevant matches."""
    
    chat_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0,  # Use 0 for maximum consistency
        response_format={
            'type': 'json_schema',
            'json_schema': 
                {
                "name": "EntailmentAnalysis", 
                "schema": EntailmentAnalysis.model_json_schema()
                }
            } 
    )

    result_object = json.loads(chat_response.choices[0].message.content)
    return result_object

def process_entailment_result(result_object, response):
    matches = []
    for match in result_object['final_matches']:
        start_index = response.lower().find(match.lower())
        if start_index == -1:
            continue
        end_index = start_index + len(match)
        matches.append((start_index, end_index))
    return matches
