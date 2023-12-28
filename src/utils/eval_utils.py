import difflib
import random
import torch.nn.functional as F
from sklearn.metrics import precision_score
import torch
import numpy as np

EMBEDDING_SIZE = 120
MAX_TOKENS = 7

def precision_at_k(model, queries, documents, k, max_queries=None, ret_type='emb'):
    """
    Computes the precision at k for a given model and data.

    :param model: PyTorch model used to compute similarity scores.
    :param queries: Dictionary of queries with their embeddings and correlated doc IDs.
    :param documents: Dictionary of document embeddings.
    :param k: Number of top documents to consider for calculating precision.
    :return: Dictionary of precision at k for each query.
    """
    precisions = {}

    # Randomly sample the queries if max_queries is specified
    if max_queries is not None and max_queries < len(queries):
        sampled_query_ids = random.sample(list(queries.keys()), max_queries)
        queries = {qid: queries[qid] for qid in sampled_query_ids}
    
    # Iterate through each query
    for query_id, query_info in queries.items():
        # Get the query embedding and relevant documents
        query_emb = torch.tensor(query_info[ret_type], dtype=torch.float32)
        relevant_docs = query_info['docids_list']
        
        # Initialize list for storing scores
        scores = []

        # Iterate through each document
        for doc_id, doc_info in documents.items():
            # Get the document embedding
            doc_emb = torch.tensor(doc_info[ret_type], dtype=torch.float32)
            
            # If the model is a SiameseNetwork
            if model.__class__.__name__ == 'SiameseNetwork':
                # Compute the score
                score = model(query_emb.unsqueeze(-1).permute(1,0).to(model.device) , doc_emb.unsqueeze(-1).permute(1,0).to(model.device)).item()

            # If the model is a SiameseTransformer
            elif model.__class__.__name__ == 'SiameseTransformer':
                # Pad the embeddings if necessary
                query_emb_padded = F.pad(query_emb.unsqueeze(0), (0, model.embedding_size - query_emb.size(0))).squeeze(0)
                doc_emb_padded = F.pad(doc_emb.unsqueeze(0), (0, model.embedding_size - doc_emb.size(0))).squeeze(0)
                # Compute the score
                score = model(query_emb_padded.unsqueeze(-1).permute(1,0).to(model.device), doc_emb_padded.unsqueeze(-1).permute(1,0).to(model.device)).item()
            
            # If the model is a SiameseNetworkPL
            elif model.__class__.__name__ == 'SiameseNetworkPL':
                # Compute the score
                score = F.cosine_similarity(model(query_emb).unsqueeze(0), model(doc_emb).unsqueeze(0)).item()

            # Append the score to the list
            scores.append((doc_id, score, 1 if doc_id in relevant_docs else 0))

        # Rank documents based on scores comouted by the model
        ranked_docs = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Calculate precision at k by counting the number of relevant documents in the top k documents
        relevant_count = sum(score for _, _, score in ranked_docs[:k])
        
        # Store the precision at k and the top k documents for the query
        precisions[query_id] = {'p@k': relevant_count / k, 'top_k_docs': ranked_docs[:k]}
    
    # Return the precisions
    return precisions


### TODO: to be modified for TF-IDF
def evaluate_precision_at_k(corpus, tfidf_matrix, num_queries_to_sample, k):
    # Extract all queries from the corpus
    all_queries = [sample['query'] for sample in corpus.data]

    # Sample queries
    sampled_queries = random.sample(all_queries, num_queries_to_sample)

    # Initialize list for storing precision at k values
    precision_at_k = []

    for i, query_sample in enumerate(sampled_queries):
        # Find the closest matching query in the dataset
        closest_match = difflib.get_close_matches(query_sample.lower().strip(),
                                                  [sample['query'].lower().strip() for sample in corpus.data])

        if not closest_match:
            print(f"No close match found for query '{query_sample}'.")
            continue

        query_index = [sample['query'].lower().strip() for sample in corpus.data].index(closest_match[0])

        relevant_documents = [j for j, relevance in enumerate(corpus.data) if relevance['relevance'] == 1]

        # Sort documents by Tf-Idf similarity and take the top k
        tfidf_row = np.asarray(tfidf_matrix[query_index].todense()).ravel()
        top_k_documents = tfidf_row.argsort()[-k:][::-1]

        # Calculate precision at k
        precision = precision_score(y_true=[1 if j in relevant_documents else 0 for j in range(len(corpus.data))],
                                    y_pred=[1 if j in top_k_documents else 0 for j in range(len(corpus.data))])
        precision_at_k.append(precision)

    # Return precision at k for each sampled query
    return precision_at_k


