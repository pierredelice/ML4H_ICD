from gensim.models.fasttext import load_facebook_model
from gensim.models import Word2Vec
import nltk, re, string, torch
from unidecode import unidecode
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

## B. Preprocess functions for generating embedding
def preprocess_text(text):
    """Preprocess text by removing special characters and applying unidecode."""
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = unidecode(text)  # Normalize text
    return text.lower()

def tokenize_and_preprocess(text_column):
    """Tokenize and preprocess all texts in the given column."""
    return [word_tokenize(preprocess_text(text)) for text in text_column]

def train_word2vec(sentences, vector_size=100, window=5, min_count=1, sg=0):
    """Train a Word2Vec CBOW model."""
    return Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count, sg=sg)

### Load glove model
def load_glove_model1(file_path):
    glove_model = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) > 2:  # Check if the line has more than just a word and one number
                word = parts[0]
                try:
                    vector = [float(x) for x in parts[1:]]
                    glove_model[word] = vector
                except ValueError:
                    print(f"Skipping line with invalid format: {line.strip()}")
    return glove_model

def load_glove_model(file_path):
    """Loads GloVe model from a file."""
    glove_model = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe Model"):
            parts = line.split()
            word = parts[0]
            vector = [float(x) for x in parts[1:]]
            glove_model[word] = vector
    return glove_model

#Generate embeddings
def generate_embeddings(dataframe, text_column, vocabulary, glove_path, fasttext_path):
    # Tokenize and preprocess texts for Word2Vec
    tokenized_texts = tokenize_and_preprocess(dataframe[text_column])
    
    # Train Word2Vec CBOW model
    word2vec_model = train_word2vec(tokenized_texts)
    # Load GloVe and FastText models
    glove_model = load_glove_model1(glove_path)
    fasttext_model = load_facebook_model(fasttext_path)
    
    # Initialize embedding dictionaries
    glove_embeddings = {}
    fasttext_embeddings = {}
    word2vec_embeddings = {}
    
    for word in tqdm(vocabulary, desc="Generating Embeddings"):
        preprocessed_word = preprocess_text(word)
        # GloVe
        glove_embeddings[word] = glove_model.get(preprocessed_word)
        # FastText
        fasttext_embeddings[word] = fasttext_model.wv[preprocessed_word] if preprocessed_word in fasttext_model.wv else None
        # Word2Vec
        word2vec_embeddings[word] = word2vec_model.wv[preprocessed_word] if preprocessed_word in word2vec_model.wv else None
    
    return glove_embeddings, fasttext_embeddings, word2vec_embeddings


def get_mean_embedding(sentence, embeddings):
    """
    Compute the mean embedding for a sentence, ensuring all embeddings are valid and rounded to 4 decimals.
    
    :param sentence: A string representing the sentence.
    :param embeddings: The GloVe embeddings dictionary.
    :return: A PyTorch tensor representing the mean embedding, rounded to 4 decimal places.
    """
    words = sentence.lower().split()
    valid_embeddings = [embeddings[word] for word in words if word in embeddings and embeddings[word] is not None]
    
    if not valid_embeddings:
        # Assuming the dimension of your embeddings is 100
        mean_embedding = np.zeros(pd.DataFrame(embeddings).shape[0])
    else:
        mean_embedding = np.round(np.mean(valid_embeddings, axis=0), 4)
    
    return torch.tensor(mean_embedding, dtype=torch.float)
