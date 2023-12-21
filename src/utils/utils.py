import difflib
import random
import numpy as np
from sklearn.metrics import precision_score
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import csv


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


def document_embedding(doc_tokens, word2vec_model):
    #print(doc_tokens[0])
    word_embeddings = [word2vec_model.wv[word] for word in doc_tokens if word in word2vec_model.wv]
    if not word_embeddings:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word_embeddings, axis=0)



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

def document_embedding_sequence(doc_tokens, word2vec_model, max_tokens):
 
    np.random.shuffle(doc_tokens)

    doc_tokens = doc_tokens[:max_tokens] + ['[PAD]'] * max(0, max_tokens - len(doc_tokens))

    word_embeddings = [word2vec_model.wv[word] for word in doc_tokens if word in word2vec_model.wv]

    if not word_embeddings:
        return np.zeros(word2vec_model.vector_size)

    return np.concatenate(word_embeddings, axis=0)


def save_dataset_to_csv(dataset, file_name):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Query Embedding', 'Document Embedding', 'Relevance', 'Type'])

        for triplet in dataset:
            for example_type, example in zip(['anchor', 'positive', 'negative'], triplet):
                # Check and convert query_embedding to string
                if example['query_embedding'] is not None:
                    query_embedding_str = ','.join(map(str, example['query_embedding']))
                else:
                    query_embedding_str = 'None'

                # Check and convert document_embedding to string
                if example['document_embedding'] is not None:
                    document_embedding_str = ','.join(map(str, example['document_embedding']))
                else:
                    document_embedding_str = 'None'

                # Write the row to the CSV
                writer.writerow([query_embedding_str, document_embedding_str, example['document_id'], example['relevance'], example_type])



def read_dataset_from_csv(file_name):
    dataset = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header

        for row in reader:
            example = {
                'query_embedding': None if row[0] == 'None' else np.array(list(map(float, row[0].split(',')))),
                'document_embedding': None if row[1] == 'None' else np.array(list(map(float, row[1].split(',')))),
                'document_id':row[2],
                'relevance': int(row[2]),
                'type': row[3]
            }

            dataset.append(example)
    
    return dataset

