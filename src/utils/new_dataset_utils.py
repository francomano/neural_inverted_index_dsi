import json
import random
import numpy as np
from nltk.tokenize import word_tokenize
from pyserini.search import get_topics, SimpleSearcher
from gensim.models import Word2Vec


# TODO: manage max_tokens option
# Define the function to get the embeddings
def compute_embedding(sentence, word2vec_model, max_tokens=None):
    # Tokenize the sentence
    tokens = word_tokenize(sentence)

    # Optionally limit the number of tokens
    tokens = tokens if not max_tokens else tokens[:max_tokens] + ['[PAD]'] * max(0, max_tokens - len(tokens))
    
    # Get the word embeddings for each token
    word_embeddings = [word2vec_model.wv[token.lower()] for token in tokens if token in word2vec_model.wv]

    # If no valid word embeddings found, return zero vector
    if not word_embeddings:
        return np.zeros(word2vec_model.vector_size)
    
    # If max_tokens is specified, concatenate the embeddings
    if max_tokens:
        avg_embedding = np.concatenate(word_embeddings, axis=0)
    # Otherwise, calculate the average embedding
    else:
        avg_embedding = np.mean(word_embeddings, axis=0)
    
    # Return the embedding
    return avg_embedding


# Define the function that builds the dictionaries of queries and documents
def build_dicts(max_topics=2, max_docs=3, max_tokens=None, vector_size=None):

    topics = get_topics('msmarco-passage-dev-subset')
    searcher = SimpleSearcher.from_prebuilt_index('msmarco-passage')

    # Initialize the dictionaries
    queries = dict()
    documents = dict()

    # Initialize w2v data
    w2v_train_data = list()

    # Optionally limit the number of topics processed
    topics_items = list(topics.items())[:max_topics] if max_topics else topics.items()

    # For each topic
    for id, topic_info in topics_items:
        # Get the query embedding and update the queries dictionary
        queries[id] = {
            'raw': topic_info['title'],     # Raw query
            'docids_list': list()}          # List of correlated documents
        # Append the query embedding to the w2v data
        w2v_train_data.append(word_tokenize(queries[id]['raw'].lower()))

        # Perform the search
        hits = searcher.search(queries[id]['raw'], max_docs)

        # For each document retrieved
        for i, hit in enumerate(hits):
            # If the document is not already in the documents dictionary
            if hit.docid not in documents:
                # Retrieve document content as a string and update the documents dictionary
                documents[hit.docid] = {'raw': json.loads(searcher.doc(hit.docid).raw())['contents']}   
                # Append the document embedding to the w2v data
                w2v_train_data.append(word_tokenize(documents[hit.docid]['raw'].lower()))        

            # Append the document id to the list of correlated documents
            queries[id]['docids_list'].append(hit.docid)

    # Train the word2vec model
    # TODO: manage the parameters
    w2v_model = Word2Vec(sentences=w2v_train_data, vector_size=vector_size, window=7, min_count=1, sg=0, epochs=10)

    # Compute embeddings for queries
    for id in queries:
        raw_query = queries[id]['raw'].lower()
        queries[id]['emb'] = compute_embedding(raw_query, w2v_model)
        queries[id]['first_L_emb'] = compute_embedding(raw_query, w2v_model, max_tokens)

    # Compute embeddings for documents
    for docid in documents:
        raw_doc = documents[docid]['raw'].lower()
        documents[docid]['emb'] = compute_embedding(raw_doc, w2v_model)
        documents[docid]['first_L_emb'] = compute_embedding(raw_doc, w2v_model, max_tokens)

    # Return the three dictionaries
    return queries, documents


# Define the function that builds the dataset
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


