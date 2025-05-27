import pandas as pd
import numpy as np
import openai
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import datetime
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from annoy import AnnoyIndex

API_KEY = 'key'
client = OpenAI(api_key=API_KEY)

def get_completion(prompt):
    for i in range(10000):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}])
            break
        except (openai.APIConnectionError, openai.APITimeoutError,
                openai.RateLimitError, openai.InternalServerError,
                openai.PermissionDeniedError) as e:
            time.sleep(1)
    return completion.choices[0].message.content

# Read data
train_df = pd.read_csv('data_open/train_data.csv')
test_df = pd.read_csv('data_open/test_data.csv')

# Load cate3_name_to_id mapping
with open('data_open/cate3_name_to_id.json', 'r', encoding='utf-8') as f:
    cate3_name_to_id = json.load(f)
    cate3_id_to_name = {int(v): k for k, v in cate3_name_to_id.items()}

def convert_hour_to_text(hour):
    try:
        hour = float(hour)
    except ValueError:
        return ''
    
    if 0 <= hour < 6:
        period = 'early morning'
    elif 6 <= hour < 12:
        period = 'morning'
    elif 12 <= hour < 13:
        period = 'noon'
    elif 13 <= hour < 18:
        period = 'afternoon'
    elif 18 <= hour <= 24:
        period = 'evening'
    else:
        period = ''
    
    hour_int = int(hour % 12)
    if hour_int == 0:
        hour_int = 12
    
    if hour - int(hour) >= 0.5:
        minute_part = ':30'
    else:
        minute_part = ':00'
    
    return f"{period} {hour_int}{minute_part}"

# Load embeddings and indices
if os.path.exists('data_open/lgn_user_embeddings.npy') and os.path.exists('data_open/lgn_st_embeddings.npy') and os.path.exists('data_open/train_node_embeddings.npy') and os.path.exists('data_open/annoy_index_lgn.ann'):
    user_embeddings = np.load('data_open/lgn_user_embeddings.npy')
    st_embeddings = np.load('data_open/lgn_st_embeddings.npy')
    with open('data_open/lgn_user_id_map.pkl', 'rb') as f:
        user_id_map = pickle.load(f)
    with open('data_open/lgn_st_id_map.pkl', 'rb') as f:
        st_id_map = pickle.load(f)
    train_node_embeddings = np.load('data_open/train_node_embeddings.npy')
    lgn_embedding_dim = train_node_embeddings.shape[1]
    annoy_index_lgn = AnnoyIndex(lgn_embedding_dim, 'angular')
    annoy_index_lgn.load('data_open/annoy_index_lgn.ann')
else:
    pass

# Map training data indices to userids
train_df.reset_index(inplace=True)
index_to_userid = train_df['userid'].to_dict()
userid_to_indices = train_df.groupby('userid', group_keys=False).apply(lambda x: x.index.tolist()).to_dict()

human_needs_system = {
    "Physiological Needs": {
        "Basic Diet": [
            "daily meals",
            "restaurant dining",
            "buffet dining"
        ],
        "Housing": [
            "comfortable accommodation",
            "luxury accommodation"
        ],
        "Daily Supplies": [
            "daily groceries and supplies",
            "fresh food shopping"
        ]
    },
    "Safety Needs": {
        "Health Protection": [
            "health maintenance",
            "maternal and postpartum care"
        ],
        "Home Safety": [
            "home safety and repair",
            "household cleaning"
        ],
        "Security": [
            "smart home security",
            "home care assistance"
        ]
    },
    "Social Belonging Needs": {
        "Social Gatherings": [
            "dining and social activities",
            "leisure entertainment"
        ],
        "Family Interaction": [
            "parent-child activities",
            "family celebrations"
        ]
    },
    "Esteem Needs": {
        "Learning": [
            "educational training",
            "arts and skills development"
        ],
        "Career Growth": [
            "academic advancement",
            "professional development"
        ]
    },
    "Self-Actualization Needs": {
        "Cultural Pursuits": [
            "cultural experiences",
            "artistic expression"
        ],
        "Travel": [
            "travel and sightseeing",
            "outdoor activities"
        ]
    }
}

def process_row(idx):
    row = test_df.iloc[idx]
    userid = row['userid']
    hour = row['hour']
    loc_scene = row['loc_scene']
    converted_hour = convert_hour_to_text(hour)
    
    user_snippets = []
    similar_snippets = []
    
    if userid in userid_to_indices:
        user_indices = userid_to_indices[userid]
        for idx_u in user_indices:
            record = train_df.iloc[idx_u]
            similar_hour = convert_hour_to_text(record['hour'])
            similar_loc_scene = record['loc_scene']
            similar_need = record['order_intention']
            similar_cate3 = record['cate3_name']
            user_snippets.append(f"At {similar_hour} in {similar_loc_scene}, you generated the living need for {similar_need} and ordered {similar_cate3}")
    else:
        user_indices = []

    user_embedding = np.zeros(lgn_embedding_dim)
    if userid in user_id_map:
        user_idx = user_id_map[userid]
        user_embedding = user_embeddings[user_idx]
    else:
        user_embedding = np.random.randn(lgn_embedding_dim)
    
    st_node = f"{hour}_{loc_scene}"
    st_embedding = np.zeros(lgn_embedding_dim)
    if st_node in st_id_map:
        st_idx = st_id_map[st_node]
        st_embedding = st_embeddings[st_idx]
    else:
        st_embedding = np.random.randn(lgn_embedding_dim)
    
    current_node_embedding = user_embedding + st_embedding
    user_indices_set = set(user_indices) if userid in userid_to_indices else set()
    
    indices_lgn = annoy_index_lgn.get_nns_by_vector(current_node_embedding, 100, include_distances=False)
    similar_indices = [idx for idx in indices_lgn if idx not in user_indices_set][:5]
    
    for idx_s in similar_indices:
        record = train_df.iloc[idx_s]
        similar_hour = convert_hour_to_text(record['hour'])
        similar_loc_scene = record['loc_scene']
        similar_need = record['order_intention']
        similar_cate3 = record['cate3_name']
        similar_snippets.append(f"At {similar_hour} in {similar_loc_scene}, a user generated the living need for {similar_need} and ordered {similar_cate3}")
    
    prompt1 = f"""You are a user on a life service platform. At {converted_hour} in {loc_scene}, what kind of living need are you most likely to have?

1. Your following past behaviors are provided for reference: 
{chr(10).join(user_snippets)}

2. Behaviors of other users in similar contexts are: 
{chr(10).join(similar_snippets)}

Considering the current time, location, and preferences indicated by previous consumption behaviors, please infer and describe your potential living needs."""

    response1 = get_completion(prompt1)

    prompt2 = f"""You are a user on a life service platform. An inference about your current living need at {converted_hour} in {loc_scene} is {response1}. Please use the following human living needs framework to further refine this inference, making it align with the framework's scope. Your response should be concise and as informative as possible, around 20 words.

{str(human_needs_system)}"""

    response2 = get_completion(prompt2)
    
    return prompt1, response1, prompt2, response2

# Process test data with multi-threading
prompts1 = [None] * len(test_df)
responses1 = [None] * len(test_df)
prompts2 = [None] * len(test_df)
responses2 = [None] * len(test_df)

with ThreadPoolExecutor(max_workers=100) as executor:
    with tqdm(total=len(test_df), desc="Processing rows", unit="row") as pbar:
        futures = {executor.submit(process_row, idx): idx for idx in range(len(test_df))}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                prompt1, response1, prompt2, response2 = future.result()
                prompts1[idx] = prompt1
                responses1[idx] = response1
                prompts2[idx] = prompt2
                responses2[idx] = response2
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
            pbar.update(1)

test_df['prompt1'] = prompts1
test_df['response1'] = responses1
test_df['prompt2'] = prompts2
test_df['predicted_intention'] = responses2

current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
llm_output_filename = f"llm_results_{current_time}.csv"
test_df.to_csv(llm_output_filename, index=False)
print(f"LLM Results saved to {llm_output_filename}")

# Recall and evaluation
instruction = "Retrieve life service categories that best fulfill this living need:"
model_embedding = SentenceTransformer('output_model/best_model_triplet_loss_llm_refined')

cate3_texts = list(cate3_name_to_id.keys())
cate3_embeddings = model_embedding.encode(
    [instruction + text for text in cate3_texts],
    normalize_embeddings=True,
    show_progress_bar=True
)

embedding_dim = cate3_embeddings.shape[1]
annoy_index_cate3 = AnnoyIndex(embedding_dim, 'angular')

for idx, emb in enumerate(cate3_embeddings):
    annoy_index_cate3.add_item(idx, emb)

annoy_index_cate3.build(10)

index_to_cate3_name = {idx: cate3_texts[idx] for idx in range(len(cate3_texts))}

predicted_intentions = test_df['predicted_intention'].tolist()
q_texts = [instruction + text for text in predicted_intentions]
q_embeddings = model_embedding.encode(
    q_texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

retrieved_cate3_names = []
for q_emb in tqdm(q_embeddings, desc="Retrieving categories", unit="query"):
    indices = annoy_index_cate3.get_nns_by_vector(q_emb, 50, include_distances=False)
    cate3_names = [index_to_cate3_name[idx] for idx in indices]
    retrieved_cate3_names.append(cate3_names)

test_df['retrieved_cate3_names'] = retrieved_cate3_names

if 'cate3_name' in test_df.columns:
    y_true_cate3_names = test_df['cate3_name'].tolist()
else:
    y_true_cate3_names = test_df['order_intention'].tolist()

y_true_cate3_names = np.array(y_true_cate3_names)
y_retrieved_cate3 = np.array(retrieved_cate3_names)

def ndcg_at_k(y_true, y_retrieved, k):
    ndcg_scores = []
    for true_item, retrieved_items in zip(y_true, y_retrieved):
        retrieved_items_k = retrieved_items[:k]
        if true_item in retrieved_items_k:
            rank = np.where(retrieved_items_k == true_item)[0][0] + 1
            ndcg = 1 / np.log2(rank + 1)
        else:
            ndcg = 0.0
        ndcg_scores.append(ndcg)
    return np.mean(ndcg_scores)

def recall_at_k(y_true, y_retrieved, k):
    hits = [1 if y_true[i] in y_retrieved[i][:k] else 0 for i in range(len(y_true))]
    return np.mean(hits)

ndcg_at_3 = ndcg_at_k(y_true_cate3_names, y_retrieved_cate3, 3)
ndcg_at_5 = ndcg_at_k(y_true_cate3_names, y_retrieved_cate3, 5)
ndcg_at_10 = ndcg_at_k(y_true_cate3_names, y_retrieved_cate3, 10)
ndcg_at_20 = ndcg_at_k(y_true_cate3_names, y_retrieved_cate3, 20)
ndcg_at_50 = ndcg_at_k(y_true_cate3_names, y_retrieved_cate3, 50)

recall_at_1 = recall_at_k(y_true_cate3_names, y_retrieved_cate3, 1)
recall_at_3 = recall_at_k(y_true_cate3_names, y_retrieved_cate3, 3)
recall_at_5 = recall_at_k(y_true_cate3_names, y_retrieved_cate3, 5)
recall_at_10 = recall_at_k(y_true_cate3_names, y_retrieved_cate3, 10)
recall_at_20 =  recall_at_k(y_true_cate3_names, y_retrieved_cate3, 20)
recall_at_50 = recall_at_k(y_true_cate3_names, y_retrieved_cate3, 50)

print(f"NDCG@3: {ndcg_at_3:.5f}")
print(f"NDCG@5: {ndcg_at_5:.5f}")
print(f"NDCG@10: {ndcg_at_10:.5f}")
print(f"NDCG@20: {ndcg_at_20:.5f}")
print(f"NDCG@50: {ndcg_at_50:.5f}")

print(f"Recall@1: {recall_at_1:.5f}")
print(f"Recall@3: {recall_at_3:.5f}")
print(f"Recall@5: {recall_at_5:.5f}")
print(f"Recall@10: {recall_at_10:.5f}")
print(f"Recall@20: {recall_at_20:.5f}")
print(f"Recall@50: {recall_at_50:.5f}")
