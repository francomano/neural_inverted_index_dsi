import string
import torch
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Corpus dataset class
class corpus_dataset(Dataset):
    def __init__(self, data):
        self.data = data  #una lista di dizionari

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        # Esempio: restituisci i tensori per la query, il documento e l'etichetta di rilevanza
        query_tensor = self.tokenize_text(sample['query'])
        document_tensor = self.tokenize_text(sample['document'])
        relevance_label = torch.tensor(sample['relevance'], dtype=torch.float32)
        docid = torch.tensor(float(sample['document_id']), dtype=torch.float32)

        return {'query': query_tensor, 'document': document_tensor, 'relevance': relevance_label, 'docid':docid}

    def tokenize_text(self, text):
        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stop words
        stop_words = set(stopwords.words('english'))

        tokens = [word for word in tokens if word.lower() not in stop_words and word.lower() not in string.punctuation ]

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return tokens
    
# Embedded dataset class
class embedded_dataset(Dataset):
    def __init__(self, data):
        self.data = data  #una lista di 4-ple

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        query_embedding, document_embedding, relevance, docid = self.data[index]

        return {'query': query_embedding, 'document': document_embedding, 'relevance': relevance, 'docid':docid}
    

class embedded_dataset_sequence(Dataset):
    def __init__(self, data):
        self.data = data  #una lista di 4-ple

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        query_embedding, document_embedding, relevance, docid = self.data[index]

        return {'query': query_embedding, 'document': document_embedding, 'relevance': relevance, 'docid':docid}