import logging
import os
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configure logging
logging.getLogger("accelerate.utils.other").setLevel(logging.ERROR)

class ModelTrainer:
    def __init__(self, model_name: str = 'BAAI/bge-base-zh-v1.5', seed: int = 42):
        """
        Initialize the trainer with model configuration and seed setting.
        
        Args:
            model_name: Name of the pre-trained model to use
            seed: Random seed for reproducibility
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.seed = seed
        self._set_seed()
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    def prepare_data(self, data_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare the dataset.
        
        Args:
            data_file: Path to the data file
            
        Returns:
            Tuple of training and validation DataFrames
        """
        data = pd.read_csv(data_file)
        data = data[['refined_prediction', 'cate3_name']].dropna()
        
        instruction = "Generate a representation for this life requirement to retrieve the most suitable service category:"
        data['refined_prediction'] = instruction + data['refined_prediction']
        
        return train_test_split(data, test_size=0.2, random_state=self.seed)
        
    def create_training_examples(self, train_df: pd.DataFrame, cache_file: str) -> List[InputExample]:
        """
        Create or load training examples with triplet loss structure.
        
        Args:
            train_df: Training DataFrame
            cache_file: Path to cache file for training examples
            
        Returns:
            List of InputExample objects
        """
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        all_categories = train_df['cate3_name'].unique().tolist()
        examples = []
        
        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Creating examples"):
            anchor = row['refined_prediction']
            positive = row['cate3_name']
            negative = self._sample_negative(positive, all_categories)
            examples.append(InputExample(texts=[anchor, positive, negative]))
            
        with open(cache_file, 'wb') as f:
            pickle.dump(examples, f)
            
        return examples
    
    def _sample_negative(self, positive: str, categories: List[str]) -> str:
        """Sample a negative example different from the positive one."""
        negative = random.choice(categories)
        while negative == positive:
            negative = random.choice(categories)
        return negative
    
    def evaluate(self, queries: Dict, corpus: Dict, relevant_docs: Dict, top_k: int = 10) -> float:
        """
        Evaluate the model using retrieval metrics.
        
        Args:
            queries: Dictionary of query_id to query_text
            corpus: Dictionary of doc_id to doc_text
            relevant_docs: Dictionary of query_id to list of relevant doc_ids
            top_k: Number of top results to consider
            
        Returns:
            Accuracy score
        """
        query_texts = list(queries.values())
        query_ids = list(queries.keys())
        corpus_texts = list(corpus.values())
        corpus_ids = list(corpus.keys())
        
        query_embeddings = self.model.encode(query_texts, convert_to_tensor=True, show_progress_bar=True, device=self.device)
        corpus_embeddings = self.model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True, device=self.device)
        
        cos_scores = torch.mm(query_embeddings, corpus_embeddings.T)
        results = []
        hits = 0
        
        for idx, query_id in enumerate(query_ids):
            scores = cos_scores[idx]
            top_results = torch.topk(scores, k=top_k)
            retrieved_ids = [corpus_ids[i] for i in top_results.indices.cpu().numpy()]
            
            if any(doc_id in retrieved_ids for doc_id in relevant_docs[query_id]):
                hits += 1
                
            results.append({
                'query': queries[query_id],
                'true_categories': '; '.join([corpus[doc_id] for doc_id in relevant_docs[query_id]]),
                'retrieved_categories': '; '.join([corpus[doc_id] for doc_id in retrieved_ids])
            })
            
        pd.DataFrame(results).to_excel('validation_results.xlsx', index=False)
        return hits / len(queries)
    
    def train(self, train_examples: List[InputExample], valid_data: Tuple, 
              batch_size: int = 64, num_epochs: int = 5, patience: int = 2,
              output_path: str = 'output_model') -> None:
        """
        Train the model using triplet loss.
        
        Args:
            train_examples: List of training examples
            valid_data: Tuple of (queries, corpus, relevant_docs) for validation
            batch_size: Training batch size
            num_epochs: Number of training epochs
            patience: Early stopping patience
            output_path: Path to save the model
        """
        os.makedirs(output_path, exist_ok=True)
        for file in os.listdir(output_path):
            os.unlink(os.path.join(output_path, file))
            
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.TripletLoss(
            model=self.model,
            distance_metric=losses.TripletDistanceMetric.EUCLIDEAN,
            triplet_margin=0.5
        )
        
        warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
        best_score = float('-inf')
        no_improve_count = 0
        
        queries, corpus, relevant_docs = valid_data
        
        for epoch in range(num_epochs):
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=1,
                warmup_steps=warmup_steps,
                show_progress_bar=True,
                output_path=None
            )
            
            accuracy = self.evaluate(queries, corpus, relevant_docs)
            print(f"Epoch {epoch + 1}/{num_epochs} - Validation Accuracy@10: {accuracy:.4f}")
            
            if accuracy > best_score:
                best_score = accuracy
                no_improve_count = 0
                self.model.save(os.path.join(output_path, 'best_model'))
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

def main():
    # Initialize trainer
    trainer = ModelTrainer()
    print(f"Using device: {trainer.device}")
    
    # Prepare data
    train_df, valid_df = trainer.prepare_data('llm_results_finetune_refined.csv')
    
    # Create training examples
    train_examples = trainer.create_training_examples(
        train_df,
        'data_open/bge_base_zh_v1.5_train_examples_triplet_full_llm_refined.pkl'
    )
    
    # Prepare validation data
    queries = {}
    corpus = {}
    relevant_docs = {}
    category_ids = {}
    next_category_id = 0
    
    for idx, row in valid_df.iterrows():
        query_id = idx
        queries[query_id] = row['refined_prediction']
        category = row['cate3_name']
        
        if category not in category_ids:
            category_id = next_category_id
            category_ids[category] = category_id
            corpus[category_id] = category
            next_category_id += 1
        else:
            category_id = category_ids[category]
            
        relevant_docs[query_id] = [category_id]
    
    # Train model
    trainer.train(train_examples, (queries, corpus, relevant_docs))

if __name__ == "__main__":
    main()
