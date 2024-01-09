import random
import torch.nn.functional as F
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


def top_k_docids(model, query, k, max_length, start_token_id):
    """
    Generate top-k sequences of document IDs for a given query.

    Args:
    - model: The trained seq2seq model.
    - query: The input query as a tensor (batch_size, seq_len).
    - k: The number of sequences to return.
    - max_length: The maximum length of each generated sequence.
    - start_token_id: The ID of the start token.

    Returns:
    - top_k_sequences: A list of k sequences, each of length max_length.
    """

    model.eval()  # Set the model to evaluation mode

    # Ensure the query tensor has a batch dimension
    if query.dim() == 1:
        query = query.unsqueeze(1)  # Add batch dimension
    elif query.size(0) < query.size(1):
        query = query.permute(1, 0)  # Swap dimensions if necessary

    # Initialize the input tensor with the start token for each item in the batch
    input_seq = torch.full((1, k), start_token_id, dtype=torch.long, device=query.device)

    # Initialize a tensor to store the top k sequences
    top_k_sequences = torch.zeros(max_length, k, dtype=torch.long, device=query.device)

    for t in range(max_length):
        output = model(query.repeat(1, k), input_seq)  # Repeat query for k hypotheses
        next_token_probs, next_tokens = torch.topk(output[-1, :, :], k, dim=-1)

        if t == 0:
            # For the first step, all k tokens are different
            # Select the top 1 token from each of the k hypotheses
            top_k_sequences[t] = next_tokens[0]
        else:
            # For subsequent steps, choose the next token based on the highest probability
            # Use argmax to find the indices of the highest probability tokens
            selected_indices = next_token_probs.argmax(dim=-1)
            top_k_sequences[t] = next_tokens.gather(1, selected_indices.unsqueeze(0)).squeeze()

        # Update input_seq for the next step with the selected tokens
        input_seq = torch.cat((input_seq, top_k_sequences[t].unsqueeze(0)), dim=0)

    # Convert the sequences to a list for easy interpretation
    top_k_sequences = top_k_sequences.t().tolist()

    return top_k_sequences


def top_k_docids_constrained(model, query, trie_data, k, max_length, start_token_id):
    # Set the model to evaluation mode
    model.eval()  

    # Ensure the query tensor has a batch dimension
    if query.dim() == 1:
        query = query.unsqueeze(1)  # Add batch dimension
    elif query.size(0) < query.size(1):
        query = query.permute(1, 0)  # Swap dimensions if necessary

    # Initialize the input tensor with the start token for each item in the batch
    input_seq = torch.full((1, k), start_token_id, dtype=torch.long, device=query.device)

    # Initialize a tensor to store the top k sequences
    top_k_sequences = torch.zeros(max_length, k, dtype=torch.long, device=query.device)

    # Until max_length is reached
    for t in range(max_length):
        # Compute the lists of possible next tokens for each of the k hypotheses
        positions_list = [trie_data.get_next_tokens(input_seq[:, i].tolist()) for i in range(k)]
        # Repeat query for k hypotheses
        output = model(query.repeat(1, k), input_seq)  

        # Apply a mask to the output so that tokens that are not in the list of possible next tokens are never chosen
        for batch_idx, positions in enumerate(positions_list): 
            # Create a mask with False everywhere except for the positions of the possible next tokens
            mask = torch.zeros(output.size(-1), dtype=bool)
            mask[positions] = True
            # Set the output logits of the impossible tokens to -inf
            output[:, batch_idx, ~mask] = -torch.inf
            # If all tokens are impossible, set the output logits of the end token to 0
            # This ensures that the end token is always chosen if all other tokens are impossible
            if not mask.any():
                output[:, batch_idx, 11] = 0
        
        # Compute the top k next tokens and their log probabilities
        next_token_probs, next_tokens = torch.topk(output[-1, :, :], k, dim=-1)

        if t == 0:
            # For the first step, all k tokens are different
            # Select the top 1 token from each of the k hypotheses
            top_k_sequences[t] = next_tokens[0]
        else:
            # For subsequent steps, choose the next token based on the highest probability
            selected_indices = next_token_probs.argmax(dim=-1)
            # Use gather to select the highest probability tokens
            top_k_sequences[t] = next_tokens.gather(1, selected_indices.unsqueeze(0).transpose(0, 1)).squeeze()

        # Update input_seq for the next step with the selected tokens
        input_seq = torch.cat((input_seq, top_k_sequences[t].unsqueeze(0)), dim=0)

    # Convert the sequences to a list for easy interpretation
    top_k_sequences = top_k_sequences.t().tolist()

    # Return the top k sequences
    return top_k_sequences
