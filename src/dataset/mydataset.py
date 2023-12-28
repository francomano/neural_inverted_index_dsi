import string
import torch
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
from sklearn.preprocessing import LabelEncoder
from neural_inverted_index_dsi.src.utils import dataset_utils


# (query, document, relevance) dataset class
class QueryDocumentDataset(Dataset):
    def __init__(
            self, 
            queries: dict, 
            documents: dict, 
            ret_type: str ='id'
        ):
        """
        Args:
            queries (dict): Dictionary with query information.
            documents (dict): Dictionary with document information.
            ret_type (str): Return type. Can be 'id', 'raw', 'emb', or 'first_L_emb'.
        """
        self.queries = queries
        self.documents = documents
        self.ret_type = ret_type
        self.data = self.build_dataset(queries, documents)

    @staticmethod
    def build_dataset(queries, documents):
        # Initialize the dataset list
        dataset = []
        # Create a set of all document IDs
        all_doc_ids = set(documents.keys())
        
        # For each query
        for query_id, query_data in queries.items():
            # Get the list of correlated documents
            docid_list = set(query_data['docids_list'])
            # Add positive examples
            for doc_id in docid_list:
                dataset.append((query_id, doc_id, 1))

            # Add negative examples, randomly sampled from all_doc_ids excluding docid_list
            for doc_id in set(random.sample(all_doc_ids - docid_list, len(docid_list))):
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
    

# (query, document (positive), document (negative)) dataset class
class TripletQueryDocumentDataset(Dataset):
    def __init__(
            self, 
            queries: dict, 
            documents: dict, 
            ret_type: str ='id'
        ):
        """
        Args:
            queries (dict): Dictionary with query information.
            documents (dict): Dictionary with document information.
            ret_type (str): Return type. Can be 'id', 'raw', 'emb', or 'first_L_emb'.
        """
        self.queries = queries
        self.documents = documents
        self.ret_type = ret_type
        self.triplets = self.build_dataset(queries, documents)

    @staticmethod
    def build_dataset(queries, documents):
        # Initialize the dataset list
        dataset = []
        # Create a set of all document IDs
        all_doc_ids = set(documents.keys())

        # For each query
        for query_id, query_data in queries.items():
            # Get the list of correlated documents
            positive_docs = set(query_data['docids_list'])
            # Get the list of negative documents, randomly sampled from all_doc_ids excluding positive_docs
            negative_docs = random.sample(all_doc_ids - set(positive_docs), len(positive_docs))
            # Add positive and negative examples
            for positive_doc, negative_doc in zip(positive_docs, negative_docs):
                dataset.append((query_id, positive_doc, negative_doc))
        
        # Return the dataset
        return dataset
    
    def set_ret_type(self, new_ret_type):
        self.ret_type = new_ret_type

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        # Get the query ID, positive document ID and negative document ID
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


# (documend, docid, label) dataset class
class IndexingTrainDataset(Dataset):
    def __init__(
            self,
            documents: dict,
            max_length: int,
            tokenizer: callable,
            label_encoder: LabelEncoder,  # Pass the label encoder as a parameter
            data: list = None
        ):
        """
        Args:
            documents (dict): Dictionary with documents information.
            max_length (int): Maximum length of the input sequence.
            tokenizer (callable): Tokenizer.
            label_encoder (LabelEncoder): Label encoder.
            data (list): List of tuples (input_ids, docid, label).
        """    
        self.documents = documents
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.data = data if data else self.build_dataset()

    # Build the dataset
    def build_dataset(self):
        # Initialize the dataset
        dataset = []

        # For each document (docid) in the documents dictionary
        for doc_id in self.documents:
            # Preprocess document content
            preprocessed_text = " ".join(dataset_utils.preprocess_text(self.documents[doc_id]['raw']))
            # Tokenize the document
            input_ids = self.tokenizer(preprocessed_text, return_tensors="pt", truncation='only_first', max_length=self.max_length).input_ids[0]
            # Encode docid label using the provided label encoder
            label = self.label_encoder.transform([doc_id])[0]
            # Tokenize the docid string
            docid = torch.tensor(self.tokenizer(doc_id, truncation='only_first', max_length=self.max_length).input_ids)

            # Pad the input_ids and docid tensors
            input_ids = torch.cat([input_ids[:self.max_length], torch.zeros(max(0, self.max_length - len(input_ids)), dtype=input_ids.dtype)])
            # Pad the input_ids and docid tensors
            docid = torch.cat([docid[:self.max_length], torch.zeros(max(0, self.max_length - len(docid)), dtype=input_ids.dtype)])

            # Append the input_ids, docid, and label to the dataset
            dataset.append((input_ids, docid, label))

        # Return the dataset
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]