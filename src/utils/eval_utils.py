import difflib
import random
import torch.nn.functional as F
from sklearn.metrics import precision_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

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

    print(type(dict(queries)))

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


# Precision at K for TF-IDF
def precision_at_k_tfidf(queries, doc_ids, tfidf_matrix, vectorizer, k, num_queries=5):
    # Initialize the results dictionary
    results = {}

    # Get num_queries random queries
    query_ids = random.sample(list(queries.keys()), num_queries)

    for query_id in query_ids:
        # Get the relevant documents for the query
        relevant_docs = queries[query_id]["docids_list"]
        # Compute the TF-IDF matrix
        query_tfidf = vectorizer.transform([queries[query_id]['raw'].lower()])
        # Compute cosine similarities
        cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
        # Get the top K retrieved documents
        top_k_retrieved = [doc_ids[i] for i in np.argsort(cosine_similarities, axis=0)[::-1]][:k]
        # Count how many of the top K retrieved documents are relevant
        relevant_count = sum(doc_id in relevant_docs for doc_id in top_k_retrieved)
        # Return the precision at K
        results[query_id] = relevant_count / k

    return results

