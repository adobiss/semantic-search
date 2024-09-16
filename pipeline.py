import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
import torch
import faiss
import json

# Configuration Management
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Model and Embedder Setup
class SBERTEmbedder:
    def __init__(self, model_name, batch_size=64):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
    
    def embed(self, texts):
        embeddings = self.model.encode(texts, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=True)
        return embeddings.cpu().numpy()
    
# Model and Tokenizer setup
class BERTEmbedder:
    def __init__(self, model_name, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
    
    def embed(self, texts):
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            print(f"Processing batch #{i // self.batch_size + 1}")
            batch_texts = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            # Normalize embeddings for cosine similarity
            batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)

# Data Access Layer
class DataAccessLayer:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def load_data(self):
        df = pd.read_csv(self.data_path)
        return df
    
    def save_vectors(self, vectors, path):
        np.save(path, vectors)
    
    def load_vectors(self, path):
        return np.load(path)

# Preprocessing
def concatenate_columns(df, columns):
    return df[columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Splitting the dataset into fine-tuning and testing sets
def split_dataset(df, sort_column, random_state, fine_tuning_ratio=0.8):
    df.sort_values(sort_column, inplace=True)
    fine_tuning_df, testing_df = train_test_split(df, test_size=(1 - fine_tuning_ratio), random_state=random_state)
    return fine_tuning_df, testing_df

def divide_dataset(df, canonical_cols, matched_cols):
    canonical_df = df[canonical_cols]
    #.drop_duplicates()
    matched_df = df[matched_cols]
    #.drop_duplicates()
    return canonical_df, matched_df
        
def dataset_subset(df, sample_size, seed):
    matched_subset = df.sample(sample_size, random_state=seed)
    return matched_subset

# Vectorize and Save
def vectorize_and_save(canonical_df, embedder, save_path, data_layer):
    canonical_texts = concatenate_columns(canonical_df, ['CAN_Title', 'CAN_Writers_Formatted'])
    canonical_vectors = embedder.embed(canonical_texts.tolist())
    data_layer.save_vectors(canonical_vectors, save_path)
    return canonical_vectors

# Real-time Vectorization and Matching
def match_compositions(matched_subset, canonical_vectors, embedder, canonical_df):
    matched_texts = concatenate_columns(matched_subset, ['MATCHED_Title', 'MATCHED_Writer_1'])
    matched_vectors = embedder.embed(matched_texts.tolist())

    # Using faiss for fast similarity search
    index = faiss.IndexFlatIP(canonical_vectors.shape[1])  # Inner product is equivalent to cosine similarity on normalized vectors
    index.add(canonical_vectors)
    
    distances, indices = index.search(matched_vectors, 1)
    
    results = []
    for i, idx in enumerate(indices):
        can_id = canonical_df.iloc[idx[0]]['CAN_ID']
        can_title = canonical_df.iloc[idx[0]]['CAN_Title']
        can_writers = canonical_df.iloc[idx[0]]['CAN_Writers_Formatted']
        match_title = matched_subset.iloc[i]['MATCHED_Title']
        match_writer = matched_subset.iloc[i]['MATCHED_Writer_1']
        similarity_score = distances[i][0]  # Cosine similarity score

        results.append({
            'CAN_ID': can_id,
            'CAN_Title': can_title,
            'CAN_Writers_Formatted': can_writers,
            'MATCHED_Title': match_title,
            'MATCHED_Writer_1': match_writer,
            'Correct_Match': can_id == matched_subset.iloc[i]['CAN_ID'],
            'Similarity_Score': similarity_score
        })
    
    return pd.DataFrame(results)