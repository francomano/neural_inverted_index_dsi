import string
import torch
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random


class QueryDocumentDataset(Dataset):
    def __init__(self, queries, documents, ret_type='id'):
        """
        Args:
            queries (dict): Dictionary with query information.
            documents (dict): Dictionary with document information.
        """
        self.queries = queries
        self.documents = documents
        self.ret_type = ret_type
        self.data = self.build_dataset(queries, documents)

    @staticmethod
    def build_dataset(queries, documents):
        # Initialize the dataset list
        dataset = []
        
        # For each query
        for query_id, query_data in queries.items():
            # Get the list of correlated documents
            docid_list = set(query_data['docids_list'])
            # Add positive examples
            for doc_id in docid_list:
                dataset.append((query_id, doc_id, 1))

            # Create a set of all document IDs, excluding those in docid_list
            all_doc_ids = set(documents.keys()) - docid_list

            # Add negative examples
            for doc_id in set(random.sample(all_doc_ids, len(docid_list))):
                dataset.append((query_id, doc_id, 0))

        # Return the dataset
        return dataset
    
    def set_ret_type(self, new_ret_type):
        self.ret_type = new_ret_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the query ID, document ID and relevance label
        query_id, doc_id, relevance = self.data[idx]

        # If the return type is 'id', return the IDs
        if self.ret_type == 'id':
            return query_id, doc_id, relevance
        
        # If the return type is 'raw', return the raw text
        elif self.ret_type == 'raw':
            return self.queries[query_id]['raw'], self.documents[doc_id]['raw'], relevance
        
        # If the return type is 'emb', return the embeddings
        elif self.ret_type == 'emb':
            return  torch.tensor(self.queries[query_id]['emb'], dtype=torch.float32), \
                    torch.tensor(self.documents[doc_id]['emb'], dtype=torch.float32), \
                    torch.tensor(relevance, dtype=torch.float32)
        
        # If the return type is 'first_L_emb', return the first L tokens embeddings
        elif self.ret_type == 'first_L_emb':
            return  torch.tensor(self.queries[query_id]['first_L_emb'], dtype=torch.float32), \
                    torch.tensor(self.documents[doc_id]['first_L_emb'], dtype=torch.float32), \
                    torch.tensor(relevance, dtype=torch.float32)
    


class TripletQueryDocumentDataset(Dataset):
    def __init__(self, queries, documents, ret_type='id'):
        """
        Args:
            queries (dict): Dictionary with query information.
            documents (dict): Dictionary with document information.
        """
        self.queries = queries
        self.documents = documents
        self.ret_type = ret_type
        self.triplets = self.build_triplets(queries, documents)

    @staticmethod
    def build_triplets(queries, documents):
        triplets = []
        all_doc_ids = set(documents.keys())

        for query_id, query_data in queries.items():
            positive_docs = set(query_data['docids_list'])
            negative_docs = random.sample(all_doc_ids - set(positive_docs), len(positive_docs))

            for positive_doc, negative_doc in zip(positive_docs, negative_docs):
                triplets.append((query_id, positive_doc, negative_doc))
        
        return triplets
    
    def set_ret_type(self, new_ret_type):
        self.ret_type = new_ret_type

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]
        
        # If the return type is 'id', return the IDs
        if self.ret_type == 'id':
            return anchor, positive, negative
        
        # If the return type is 'raw', return the raw text
        if self.ret_type == 'raw':
            return self.queries[anchor]['raw'], self.documents[positive]['raw'], self.documents[negative]['raw']
        
        # If the return type is 'emb', return the embeddings
        if self.ret_type == 'emb':
            return  torch.tensor(self.queries[anchor]['emb'], dtype=torch.float32), \
                    torch.tensor(self.documents[positive]['emb'], dtype=torch.float32), \
                    torch.tensor(self.documents[negative]['emb'], dtype=torch.float32)
        
        # If the return type is 'first_L_emb', return the first L tokens embeddings
        if self.ret_type == 'first_L_emb':
            return  torch.tensor(self.queries[anchor]['first_L_emb'], dtype=torch.float32), \
                    torch.tensor(self.documents[positive]['first_L_emb'], dtype=torch.float32), \
                    torch.tensor(self.documents[negative]['first_L_emb'], dtype=torch.float32)














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
    

class SiameseDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_embedding = torch.FloatTensor(self.triplets[idx][0]['query_embedding'])
        positive_embedding = torch.FloatTensor(self.triplets[idx][1]['document_embedding'])
        negative_embedding = torch.FloatTensor(self.triplets[idx][2]['document_embedding'])


        return anchor_embedding, positive_embedding, negative_embedding