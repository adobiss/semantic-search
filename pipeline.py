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
    """
    Loads the configuration from a JSON file.

    Args:
        config_path (str): The file path of the configuration file.

    Returns:
        dict: The configuration settings as a dictionary.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Sentence-BERT Model Setup
class SBERTEmbedder:
    """
    A class for embedding text using a pre-trained Sentence-BERT model.

    Attributes:
        model (SentenceTransformer): The Sentence-BERT model.
        batch_size (int): The batch size used for embedding.
    """
    
    def __init__(self, model_name, batch_size=64):
        """
        Initializes the SBERTEmbedder with the specified model.

        Args:
            model_name (str): The name of the pre-trained model.
            batch_size (int): The batch size for embedding. Defaults to 64.
        """
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
    
    def embed(self, texts):
        """
        Embeds a list of texts using the Sentence-BERT model.

        Args:
            texts (list): A list of texts to embed.

        Returns:
            np.ndarray: An array of embeddings.
        """
        embeddings = self.model.encode(texts, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=True)
        return embeddings.cpu().numpy()

# BERT Model and Tokenizer setup
class BERTEmbedder:
    """
    A class for embedding text using a pre-trained BERT model.

    Attributes:
        device (torch.device): The device used for computation (CPU or GPU).
        tokenizer (BertTokenizer): The tokenizer from Hugging Face.
        model (BertModel): The pre-trained BERT model.
        batch_size (int): The batch size used for embedding.
    """
    
    def __init__(self, model_name, batch_size=32):
        """
        Initializes the BERTEmbedder with the specified model.

        Args:
            model_name (str): The name of the pre-trained BERT model.
            batch_size (int): The batch size for embedding. Defaults to 32.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
    
    def embed(self, texts):
        """
        Embeds a list of texts using the BERT model.

        Args:
            texts (list): A list of texts to embed.

        Returns:
            np.ndarray: An array of embeddings.
        """
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
    """
    A class for interacting with data sources (loading and saving).
    
    Attributes:
        data_path (str): The path to the dataset.
    """
    
    def __init__(self, data_path):
        """
        Initializes the DataAccessLayer with the given data path.

        Args:
            data_path (str): The file path to the dataset.
        """
        self.data_path = data_path
    
    def load_data(self):
        """
        Loads data from the CSV file.

        Returns:
            pd.DataFrame: The loaded dataset as a DataFrame.
        """
        df = pd.read_csv(self.data_path)
        return df
    
    def save_vectors(self, vectors, path):
        """
        Saves the vectors as a NumPy file.

        Args:
            vectors (np.ndarray): The vectors to save.
            path (str): The path to save the file.
        """
        np.save(path, vectors)
    
    def load_vectors(self, path):
        """
        Loads vectors from a NumPy file.

        Args:
            path (str): The path to load the vectors from.

        Returns:
            np.ndarray: The loaded vectors.
        """
        return np.load(path)

# Preprocessing
def concatenate_columns(df, columns):
    """
    Concatenates multiple columns in a DataFrame into a single string for each row.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (list): The list of column names to concatenate.

    Returns:
        pd.Series: A Series of concatenated strings.
    """
    return df[columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Splitting the dataset into fine-tuning and testing sets
def split_dataset(df, sort_column, random_state, fine_tuning_ratio=0.8):
    """
    Splits the dataset into fine-tuning and testing sets.

    Args:
        df (pd.DataFrame): The dataset to split.
        sort_column (str): The column to sort by before splitting.
        random_state (int): The seed for reproducibility.
        fine_tuning_ratio (float): The ratio of the fine-tuning set. Defaults to 0.8.

    Returns:
        tuple: A tuple containing the fine-tuning DataFrame and the testing DataFrame.
    """
    df.sort_values(sort_column, inplace=True)
    fine_tuning_df, testing_df = train_test_split(df, test_size=(1 - fine_tuning_ratio), random_state=random_state)
    return fine_tuning_df, testing_df

# Divide canonical and matched pairs for training and inference
def divide_dataset(df, canonical_cols, matched_cols):
    """
    Divides the dataset into canonical and matched DataFrames.

    Args:
        df (pd.DataFrame): The dataset to divide.
        canonical_cols (list): The list of columns for the canonical data.
        matched_cols (list): The list of columns for the matched data.

    Returns:
        tuple: A tuple containing the canonical DataFrame and matched DataFrame.
    """
    canonical_df = df[canonical_cols]
    matched_df = df[matched_cols]
    return canonical_df, matched_df

# Extract dataset subset         
def dataset_subset(df, sample_size, seed):
    """
    Extracts a random subset of the dataset.

    Args:
        df (pd.DataFrame): The dataset to sample from.
        sample_size (int): The size of the subset.
        seed (int): The seed for reproducibility.

    Returns:
        pd.DataFrame: The sampled subset.
    """
    return df.sample(sample_size, random_state=seed)

# Vectorize and Save
def vectorize_and_save(canonical_df, embedder, save_path, data_layer):
    """
    Vectorizes the canonical dataframe and saves the vectors.

    Args:
        canonical_df (pd.DataFrame): The DataFrame containing canonical texts.
        embedder (object): The embedder to convert texts to vectors.
        save_path (str): The path to save the vectors.
        data_layer (DataAccessLayer): The DataAccessLayer for saving the vectors.

    Returns:
        np.ndarray: The computed canonical vectors.
    """
    canonical_texts = concatenate_columns(canonical_df, ['CAN_Title', 'CAN_Writers_Formatted'])
    canonical_vectors = embedder.embed(canonical_texts.tolist())
    data_layer.save_vectors(canonical_vectors, save_path)
    return canonical_vectors

# Real-time Vectorization and Matching
def match_compositions(matched_subset, canonical_vectors, embedder, canonical_df):
    """
    Matches compositions from the matched subset to the canonical set using vector similarity.

    Args:
        matched_subset (pd.DataFrame): The subset of matched compositions.
        canonical_vectors (np.ndarray): The pre-computed canonical vectors.
        embedder (object): The embedder for vectorizing the matched texts.
        canonical_df (pd.DataFrame): The DataFrame containing canonical information.

    Returns:
        pd.DataFrame: A DataFrame of matching results with similarity scores.
    """
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
