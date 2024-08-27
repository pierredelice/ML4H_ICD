# Import the necessary functions from your_script.py
import nltk, re, unidecode, torch, pickle, os
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from modules import install_and_import, is_conda, parse_requirements, freeze_requirements, main
from src.data.data_loader import read_data
from src.data.embeddings import generate_embeddings, get_mean_embedding
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.data.dataset import prepare_data, MedicalDataset, DataLoader, collate_fn, apply_flatten
from src.models.seq2seq_model import Seq2Seq
from src.models.train import train_and_evaluate_model
from src.models.plot_results import plot_training_and_evaluation_metrics
from torchsummary import summary


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
    path = 'Data/seedicd.pkl'
    df = read_data(path)
    df = df[['cause','causa_icd']]#.sample(1_000, random_state = 2011)
    print(df)
    label_mapping = {value: label for label, value in enumerate(df['causa_icd'].unique())}
    df['label'] = df['causa_icd'].map(label_mapping)
    text,label = df['cause'].values, df['label'].values
    vocabulary = set([word for item in text for word in str(item).split()])
    list_of_tuples = list(zip(df['cause'], df['causa_icd']))
    

    word_to_idx, label_encoder = prepare_data(list_of_tuples)
    output_size = len(label_encoder.classes_)
    print(f"Print output_size: {output_size}")

    if output_size >0:
        dataset = MedicalDataset(list_of_tuples, word_to_idx, label_encoder)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: collate_fn(x, word_to_idx['<PAD>']))
        test_loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=lambda x: collate_fn(x, word_to_idx['<PAD>']))

        input_size = len(word_to_idx)
        embedding_size = 512
        hidden_size = 64

        # Initialize the model, criterion, optimizer, and other settings
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_a = Seq2Seq(input_size, embedding_size, output_size, hidden_size).to(device)
	# Wrap model with DataParallel
        #model_a = nn.DataParallel(model_a)
        #apply_flatten(model_a)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_a.parameters(), lr=0.01)
        num_epochs = 15
        patience = 3
        print(model_a,'\n')
        print(summary(model_a))
        os.makedirs('results', exist_ok=True)    
        
        training_losses, training_accuracy, evaluation_accuracy, evaluation_precision, evaluation_recall, evaluation_f1 = train_and_evaluate_model(
        model_a, train_loader, test_loader, criterion, optimizer, num_epochs, patience)

    
    plot_training_and_evaluation_metrics(training_losses, training_accuracy, evaluation_accuracy, evaluation_precision, evaluation_recall, evaluation_f1)

    
    # Step 8: Save the model and tokenizer
    torch.save(model_a.state_dict(), 'results/seq2seq_model.pth')
    with open('results/tokenizer.pkl', 'wb') as f:
        pickle.dump({'word_to_idx': word_to_idx, 'label_encoder': label_encoder}, f)
    
    print("--------------------------------------------------------")
    print("Model successfully saved")


if __name__ == "__main__":
    main_script()

