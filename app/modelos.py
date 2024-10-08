# coding:utf8
import torch, pickle
from src.models.seq2seq_model import Seq2Seq
from src.data.dataset import MedicalDataset, collate_fn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

small_data = [
    "1 82 chocar cardiogenico",
    "2 53 covid virus identificar",
    "2 33 hipertension arteriel",
    "2 53 sars cov",
    "2 40 chocar hipovolemico",
    "1 82 choque cardiogenico",
    "2 53 covid virus identificar",
    "2 52 acidosis metabólico",
    "1 57 infarto agudo miocardio",
    "1 63 insuficiencia respiratorio agudo",
    "1 49 hipertensión arterial",
    "2 79 insuficiencia renal crónico",
    "2 69 síndrome dificultad respiratorio adulto",
    "1 70 insuficiencia respiratorio",
    "2 64 choque séptico",
]


def load_model_and_tokenizer(model_path, tokenizer_path):

    # Load the tokenizer and label encoder
    with open(tokenizer_path, 'rb') as f:
        tokenizer_data = pickle.load(f)
        word_to_idx = tokenizer_data['word_to_idx']
        label_encoder = tokenizer_data['label_encoder']
        
    input_size = len(word_to_idx)
    embedding_size = 512
    output_size = len(label_encoder.classes_)
    hidden_size = 64

    model = Seq2Seq(input_size, embedding_size, output_size, hidden_size)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
        
    return model, word_to_idx, label_encoder

class BiGru(object):

    def __init__(self, model_path, tokenizer_path):
        self.loaded_model, self.loaded_word_to_idx, self.loaded_label_encoder = load_model_and_tokenizer(
            f'{model_path}/seq2seq_model.pth', 
            f'{tokenizer_path}/tokenizer.pkl')
        self.output_size = len(self.loaded_label_encoder.classes_)

    def predict_batch(self, small_data): 
        loaded_model = self.loaded_model
        loaded_word_to_idx = self.loaded_word_to_idx
        loaded_label_encoder = self.loaded_label_encoder
        output_size = self.output_size

        small_dataset = MedicalDataset(small_data, loaded_word_to_idx, loaded_label_encoder, is_prediction=True)
        small_loader  = DataLoader(small_dataset, batch_size=16, shuffle=False, 
                                   collate_fn=lambda x: pad_sequence(x, batch_first=True, 
                                                                     padding_value=loaded_word_to_idx['<PAD>']))
        # print(small_loader.dataset)
        for inx in small_loader:
            src = inx.to(device)       
            trg_onehot = torch.zeros((src.size(0), 1, output_size)).to(device)  # Initialize target with zeros
            
    
            outputs = loaded_model(src, trg_onehot)
            _, predicted = torch.max(outputs, dim=-1)
            predicted_labels = [loaded_label_encoder.inverse_transform([pred.item()])[0] for pred in predicted.flatten()]
            return predicted_labels
    
    def predict_single(self, cause): 
        loaded_model = self.loaded_model
        loaded_word_to_idx = self.loaded_word_to_idx
        loaded_label_encoder = self.loaded_label_encoder
        output_size = self.output_size

        


if __name__=='__main__':
    bigru = BiGru()
    small_data = input("Dame el diagnóstico: ")
    print(bigru.predict_batch(small_data=small_data))
