import pandas as pd
import numpy as np
import openai
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# Replace with your actual API key
API_KEY = 'key'
client = OpenAI(api_key=API_KEY)

# Define function to call LLM
def get_completion(prompt):
    for i in range(10000):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",  # Adjust model name according to your needs
                messages=[{"role": "user", "content": prompt}]
            )
            break
        except (openai.APIConnectionError, openai.APITimeoutError,
                openai.RateLimitError, openai.InternalServerError,
                openai.PermissionDeniedError) as e:
            time.sleep(1)
    return completion.choices[0].message.content

# Read CSV file
df = pd.read_csv('llm_results_finetune.csv')

# Define function to process each row
def process_row(row):
    order_intention = row['order_intention']
    predicted_intention = row['predicted_intention']
    prompt = f"The ground-truth living need is: {order_intention}; a flexible need description is: {predicted_intention}. Based on the ground-truth living need, please revise the flexible need description to maintain flexibility without compromising accuracy. Provide only the revised flexible need description, approximately 20 words."
    refined_prediction = get_completion(prompt)
    return refined_prediction

# Wrapper function with index to ensure correct order
def process_row_with_index(args):
    index, row = args
    refined_prediction = process_row(row)
    return index, refined_prediction

# Prepare argument list
args_list = list(df.iterrows())

# Create list to store results with correct indexing
refined_predictions = [None] * len(df)

# Use thread pool executor for multi-threading with progress bar
with ThreadPoolExecutor(max_workers=100) as executor:
    futures = {executor.submit(process_row_with_index, arg): arg[0] for arg in args_list}
    for future in tqdm(as_completed(futures), total=len(futures)):
        index, refined_prediction = future.result()
        refined_predictions[index] = refined_prediction

# Add new column to DataFrame
df['refined_prediction'] = refined_predictions

# Save DataFrame back to CSV
df.to_csv('llm_results_finetune_refined.csv', index=False)
