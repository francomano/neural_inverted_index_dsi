import difflib
import random
import torch.nn.functional as F
from sklearn.metrics import precision_score
import torch
import numpy as np


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


def evaluate_siamese_query(siamese_model, query_and_document_embeddings, query_index, k=100, threshold=0.6):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize sets for retrieved and relevant documents
    retrieved_docs = set()
    relevant_docs = set()

    # Set the model to evaluation mode
    siamese_model.to(device)
    siamese_model.eval()

    # Retrieve the query embedding
    my_query = query_and_document_embeddings[query_index][0]

    # Iterate over each query, document, relevance, and document id in the embeddings
    for query, doc, relevance, id in query_and_document_embeddings:
        # Predict the relevance
        pred = siamese_model(torch.from_numpy(my_query).unsqueeze(-1).permute(1,0).to(device),
                             torch.from_numpy(doc).unsqueeze(-1).permute(1,0).to(device))
        if pred > threshold:
            tuple_to_add = (int(id.item()), float(pred))
            retrieved_docs.add(tuple_to_add)
            if torch.equal(torch.from_numpy(query), torch.from_numpy(my_query)) and relevance.item() == 1: 
                relevant_docs.add(tuple_to_add)

    # Sort and select top k retrieved documents
    retrieved_docs_sorted = sorted(retrieved_docs, key=lambda x: x[1], reverse=True)
    top_k = retrieved_docs_sorted[:k]

    # Calculate the number of relevant documents in top k
    rel = sum(1 for docid in top_k if docid in relevant_docs)

    # Return the precision metrics
    return {
        "query_index": query_index,
        "retrieved_documents": k,
        "precision_at_k": rel / k,
        "relevant_docids": relevant_docs,
        "top_k_retrieved_docids": top_k
    }


def evaluate_att_siamese_query(siamese_transformer, query_and_document_embeddings_sequence, query_index, size, k=10000, threshold=0.6):
    # Determine the device to use
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using GPU." if device.type == "cuda" else "Using CPU.")

    # Initialize sets for retrieved and relevant documents
    retrieved_docs = set()
    relevant_docs = set()

    # Prepare the model
    siamese_transformer.to(device)
    siamese_transformer.eval()

    # Prepare the query
    my_query = torch.from_numpy(query_and_document_embeddings_sequence[query_index][0])
    my_query_padded = F.pad(my_query.unsqueeze(0), (0, size - my_query.size(0))).squeeze(0)

    # Process each query, document, relevance, and id
    for query, doc, relevance, id in query_and_document_embeddings_sequence:
        queries_padded = F.pad(torch.from_numpy(query).unsqueeze(0), (0, size - query.size)).squeeze(0).to(torch.float32)
        documents_padded = F.pad(torch.from_numpy(doc).unsqueeze(0), (0, size - doc.size)).squeeze(0).to(torch.float32)

        pred = siamese_transformer(queries_padded.unsqueeze(-1).permute(1, 0).to(device),
                                   documents_padded.unsqueeze(-1).permute(1, 0).to(device))

        if pred > threshold:
            tuple_to_add = (int(id.item()), float(pred))
            retrieved_docs.add(tuple_to_add)
            if torch.equal(queries_padded, my_query_padded) and relevance.item() == 1:
                relevant_docs.add(tuple_to_add)

    # Calculate Precision at K
    retrieved_docs_sorted = sorted(retrieved_docs, key=lambda x: x[1], reverse=True)
    top_k = retrieved_docs_sorted[:k]
    rel = sum(1 for docid in top_k if docid in relevant_docs)

    # Return the precision metrics
    return {
        "query_index": query_index,
        "retrieved_documents": k,
        "precision_at_k": rel / k,
        "relevant_docids": relevant_docs,
        "top_k_retrieved_docids": top_k
    }


def precision_at_k(model, dataset, k=10):
    # Initialize precision
    precisions = []

    # Ensure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        for data in dataset:
            query_embedding = torch.FloatTensor(data[0])
            processed_query_emb = model(query_embedding.unsqueeze(0))

            # Process and score each document for the given query
            doc_scores = []
            for doc in dataset:
                if np.array_equal(doc[0], data[0]):
                    document_embedding = torch.FloatTensor(doc[1])
                    processed_doc_emb = model(document_embedding.unsqueeze(0))
                    # Compute cosine similarity
                    score = F.cosine_similarity(processed_query_emb, processed_doc_emb).item()
                    doc_scores.append((score, doc[2]))

            # Sort based on scores
            doc_scores.sort(key=lambda x: x[0], reverse=True)

            # Compute precision at K
            top_k_relevant = sum(relevance for _, relevance in doc_scores[:k])
            precision = top_k_relevant / k
            precisions.append(precision)

    # Return the average precision at K
    return sum(precisions) / len(precisions) if precisions else 0