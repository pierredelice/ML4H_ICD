# Import the necessary functions from your_script.py
from modules import install_and_import, is_conda, parse_requirements, freeze_requirements, main
from src.data.data_loader import read_data
from src.data.embeddings import generate_embeddings, get_mean_embedding
from tqdm import tqdm
import nltk, re, unidecode, torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def main_script():
    # Check if conda is available
    # conda = is_conda()
    # print(f"Is conda available? {conda}")

    # Install a specific package if it's not already installed
    install_and_import('numpy')

    # Parse a requirements.txt file
    #requirements = parse_requirements('requirements.txt')
    #print(f"Parsed requirements: {requirements}")

    # Freeze current environment packages to requirements.txt
    #freeze_requirements('frozen_requirements.txt')
    #print("Requirements have been frozen to frozen_requirements.txt")

    
    # Run the main function from your_script.py
    # main()
    # Read data
    path = 'Data/icd_clean.pkl'
    df = read_data(path)
    df = df[['cause','causa_icd']].sample(1_000, random_state = 123)
    label_mapping = {value: label for label, value in enumerate(df['causa_icd'].unique())}
    df['label'] = df['causa_icd'].map(label_mapping)
    text,label = df['cause'].values, df['label'].values
    vocabulary = set([word for item in text for word in str(item).split()])
    
    print("Run embedding function")
    glove_path = '../economic_thesis/data/embeddings/glove/glove.6B.300d.txt'
    fasttext_path = '../economic_thesis/data/embeddings/fasttext/cc.es.300.bin'
    glove_embs, fasttext_embs, word2vec_embs = generate_embeddings(df, 'cause', vocabulary, glove_path, fasttext_path)
    print("-------------------------------------------------------------")
    glove_embedding = torch.stack([get_mean_embedding(sentence, glove_embs) for sentence in tqdm(df['cause'], desc="Computing GloVe embeddings")])
    print("-------------------------------------------------------------")
    word2_vec_embedding = torch.stack([get_mean_embedding(sentence, word2vec_embs) for sentence in tqdm(df['cause'], desc="Computing Word2vec embeddings")])
    print("-------------------------------------------------------------")
    fasttext_embedding = torch.stack([get_mean_embedding(sentence, fasttext_embs) for sentence in tqdm(df['cause'], desc="Computing FastText embeddings")])
    print(fasttext_embedding.shape)

if __name__ == "__main__":
    main_script()

