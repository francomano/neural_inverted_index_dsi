import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from neural_inverted_index_dsi.src.utils import dataset_utils
from transformers import T5Tokenizer
# from sklearn.decomposition import IncrementalPCA


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


# (encoded_doc, encoded_docid) dataset class
class DocumentDataset(Dataset):
    def __init__(
            self, 
            documents: dict, 
            doc_max_len: int = 32
        ):
        """
        Args:
            documents (dict): Dictionary with document information.
            doc_max_len (int): Maximum length of the input sequence.
        """
        self.documents = documents
        self.doc_max_len = doc_max_len
        
        # We use the T5 tokenizer to encode the documents
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

        # Define the docid special tokens
        self.docid_eos_token = 10       # End of string token for docids
        self.docid_pad_token = 11       # Padding token for docids
        self.docid_start_token = 12     # Start of string token for docids

        # Compute the maximum docid length
        self.max_docid_len = max(len(str(docid)) for docid in documents.keys()) + 2  # for EOS token and start token

        # Initialize the encoded documents and docids lists
        self.encoded_docs, self.docids = self.build_dataset()
    
    def build_dataset(self):
        # Initialize the encoded documents and docids lists
        docids = []
        encoded_docs = []

        # For each document in the documents dictionary
        for docid, content in self.documents.items():
            # Tokenizing and encoding the document text
            preprocessed_text = dataset_utils.preprocess_text(content['raw'])
            preprocessed_text = " ".join(preprocessed_text)
            encoded_doc = self.tokenizer.encode(preprocessed_text,
                                           add_special_tokens=True,
                                           max_length=self.doc_max_len,
                                           truncation=True)

            # Padding the document sequence to max_length
            encoded_doc = F.pad(torch.tensor(encoded_doc), (0, self.doc_max_len - len(encoded_doc)), value=0)

            # Encoding the docid (treating each character as a digit)
            encoded_docid = torch.tensor([self.docid_start_token] + list(map(int, docid)) + [self.docid_eos_token] +
                                    [self.docid_pad_token] * (self.max_docid_len - len(docid)))

            # Appending the encoded document and docid to the lists
            docids.append(encoded_docid)
            encoded_docs.append(encoded_doc)

        return encoded_docs, docids

    def __len__(self):
        return len(self.encoded_docs)

    def __getitem__(self, idx):
        return self.encoded_docs[idx], self.docids[idx]

    def decode_docid(self, encoded_docid):
        # Convert to numpy array, skip the first element, convert elements to string, and concatenate
        decoded = ''.join(map(str, np.array(encoded_docid)[1:]))

        # Remove end-of-string and padding tokens from the end of the string
        return decoded.rstrip(str(self.docid_eos_token) + str(self.docid_pad_token))
    

# (encoded_query, encoded_docid) dataset class
class RetrievalDataset(Dataset):
    def __init__(
            self, 
            documents: dict, 
            queries: dict, 
            query_max_len: int = 9
        ):
        """
        Args:
            documents (dict): Dictionary with document information.
            queries (dict): Dictionary with query information.
            query_max_len (int): Maximum length of the input sequence.
        """
        self.documents = documents
        self.queries = queries
        self.query_max_len = query_max_len
        
        # We use the T5 tokenizer to encode the documents
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

        # Define the docid special tokens
        self.docid_eos_token = 10   # End of string token for docids
        self.docid_pad_token = 11   # Padding token for docids
        self.docid_start_token = 12

        # Compute the maximum docid length
        self.max_docid_len = max(len(str(docid)) for docid in documents.keys()) + 2  # for EOS token and start token

        # Initialize the encoded documents and docids lists
        self.encoded_queries, self.docids =  self.build_dataset()

    def build_dataset(self):
        # Initialize the encoded documents and docids lists
        docids = []
        encoded_queries = []

        # Iterate over the queries
        for queryid, content in self.queries.items():
            # Tokenizing and encoding the document text
            preprocessed_text = dataset_utils.preprocess_text(content['raw'])
            preprocessed_text = " ".join(preprocessed_text)
            encoded_query = self.tokenizer.encode(preprocessed_text,
                                                add_special_tokens=True,
                                                max_length=self.query_max_len,
                                                truncation=True)

            # Padding the document sequence to max_length
            encoded_query = F.pad(torch.tensor(encoded_query), (0, self.query_max_len - len(encoded_query)), value=0)

            for docid in content['docids_list']:

                # Encoding the docid (treating each character as a digit)
                encoded_docid = torch.tensor([self.docid_start_token] + list(map(int, docid)) + [self.docid_eos_token] +
                                        [self.docid_pad_token] * (self.max_docid_len - len(docid)))


                #self.docids.append(encoded_docid)
                encoded_queries.append(encoded_query)
                docids.append(encoded_docid)

        return encoded_queries, docids

    def __len__(self):
        return len(self.encoded_queries)

    def __getitem__(self, idx):
        return self.encoded_queries[idx], self.docids[idx]

    def decode_docid(self, encoded_docid):
        # Convert to numpy array, skip the first element, convert elements to string, and concatenate
        decoded = ''.join(map(str, np.array(encoded_docid)[1:]))

        # Remove end-of-string and padding tokens from the end of the string
        return decoded.rstrip(str(self.docid_eos_token) + str(self.docid_pad_token))



'''
class SemanticDocIDManager:
    def __init__(self, documents, max_docid_dim=10, batch_size=None):
        self.documents = documents
        self.max_docid_dim = max_docid_dim
        self.batch_size = batch_size
        self.semantic_docids, self.reverse_mapping = self.create_semantic_docids()

    def create_semantic_docids(self):
        embeddings = np.array([content['emb'] for content in self.documents.values()])

        ipca = IncrementalPCA(n_components=self.max_docid_dim, batch_size=self.batch_size)
        reduced_embeddings = ipca.fit_transform(embeddings)

        # Normalize: absolute value and scale
        normalized_embeddings = np.abs(reduced_embeddings) * 1000  # Scale factor

        semantic_docids = {}
        reverse_mapping = {}
        for docid, normalized_embedding in zip(self.documents.keys(), normalized_embeddings):
            # Convert to a flat string of integers
            semantic_id = ''.join(map(lambda x: f"{int(x):01d}", normalized_embedding))  # Zero-padding for uniform length
            semantic_docids[docid] = semantic_id
            reverse_mapping[semantic_id] = docid
        return semantic_docids, reverse_mapping

    def map_semantic_to_original(self, semantic_docid):
        return self.reverse_mapping.get(semantic_docid, None)


class DocumentDataset(Dataset):
    def __init__(self, documents, semantic_docids, doc_max_len=7):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.documents = documents
        self.semantic_docids = semantic_docids
        self.doc_max_len = doc_max_len

        self.encoded_docs = []
        self.encoded_semantic_docids = []

        self.docid_eos_token = 10   # End of string token for docids
        self.docid_pad_token = 11   # Padding token for docids
        self.docid_start_token = 12

        self.max_docid_len = 8

        for docid, content in documents.items():
            # Tokenizing and encoding the document text
            preprocessed_text = dataset_utils.preprocess_text(content['raw'])
            preprocessed_text = " ".join(preprocessed_text)
            encoded_doc = self.tokenizer.encode(preprocessed_text,add_special_tokens=True,max_length=self.doc_max_len,truncation=True)
            encoded_doc = F.pad(torch.tensor(encoded_doc), (0, self.doc_max_len - len(encoded_doc)), value=0)
            self.encoded_docs.append(encoded_doc)

            # Encoding the semantic docid
            semantic_docid = self.semantic_docids[docid]
            #encoded_semantic_docid = self.tokenizer.encode(semantic_docid, add_special_tokens=False)
            # Encoding the docid (treating each character as a digit)
            #print(semantic_docid)
            encoded_semantic_docid = torch.tensor([self.docid_start_token] + list(map(int, semantic_docid)) + [self.docid_eos_token] +
                                     [self.docid_pad_token] * (self.max_docid_len - len(semantic_docid)))
            self.encoded_semantic_docids.append(encoded_semantic_docid)

    def __len__(self):
        return len(self.encoded_docs)

    def __getitem__(self, idx):
        return self.encoded_docs[idx], self.encoded_semantic_docids[idx]

    def decode_docid(self, encoded_docid):
        # Convert to numpy array and skip the first element (start token)
        docid_array = np.array(encoded_docid)[1:]

        # Convert array elements to string and concatenate them
        decoded = ''.join(map(str, docid_array))

        # Remove end-of-string and padding tokens from the end of the string
        decoded = decoded.rstrip(str(self.docid_eos_token))
        decoded = decoded.rstrip(str(self.docid_pad_token))

        return decoded
'''