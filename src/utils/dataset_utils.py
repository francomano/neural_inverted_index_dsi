import json
import random
import pyserini.search
import numpy as np
import csv
from nltk.tokenize import word_tokenize

def build_dataset(num_docs_per_query=10, num_topics=None):
    # Define the topics to retrieve
    topics = pyserini.search.get_topics('msmarco-passage-dev-subset')
    # Define the searcher to retrieve the documents
    searcher = pyserini.search.SimpleSearcher.from_prebuilt_index('msmarco-passage')

    # Initialize the dataset
    dataset = []

    # Optionally limit the number of topics processed
    if num_topics is not None:
        topic_items = list(topics.items())[:num_topics]
    else:
        topic_items = topics.items()

    # For each topic
    for topic_id, topic_info in topic_items:
        query = topic_info['title']
        hits = searcher.search(query)

        # For the first num_docs_per_query documents retrieved
        for i, hit in enumerate(hits[:num_docs_per_query]):
            jsondoc = json.loads(hit.raw)
            document_id, document_contents = jsondoc["id"], jsondoc["contents"]
            dataset.append({'query': query, 'document_id': document_id, 'document': document_contents, 'relevance': 1})

            # Find a random document from another query
            other_query_id = random.choice([q_id for q_id in topics if q_id != topic_id])
            other_query = topics[other_query_id]['title']
            other_hits = searcher.search(other_query)
            jsondoc_neg = random.choice([json.loads(hit.raw) for hit in other_hits])
            document_id_neg, document_contents_neg = jsondoc_neg["id"], jsondoc_neg["contents"]
            dataset.append({'query': query, 'document_id': document_id_neg, 'document': document_contents_neg, 'relevance': 0})

    return dataset



def get_embedding(sentence, word2vec_model):
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    
    # Get the word embeddings for each token
    word_embeddings = [word2vec_model.wv[token] for token in tokens if token in word2vec_model.wv]
    
    # If no valid word embeddings found, return None
    if not word_embeddings:
        return np.zeros(word2vec_model.vector_size)
    
    # Calculate the average embedding
    avg_embedding = np.mean(word_embeddings, axis=0)
    
    return avg_embedding

def build_triplet_dataset_with_avg_embeddings(word2vec_model, num_docs_per_query=10, num_topics=None):
    # Define the topics to retrieve
    topics = pyserini.search.get_topics('msmarco-passage-dev-subset')
    # Define the searcher to retrieve the documents
    searcher = pyserini.search.SimpleSearcher.from_prebuilt_index('msmarco-passage')

    # Initialize the dataset and embeddings
    dataset = []

    # Optionally limit the number of topics processed
    if num_topics is not None:
        topic_items = list(topics.items())[:num_topics]
    else:
        topic_items = topics.items()

    # For each topic
    for topic_id, topic_info in topic_items:
        query = topic_info['title']
        hits = searcher.search(query)

        # For the first num_docs_per_query documents retrieved
        for i, hit in enumerate(hits[:num_docs_per_query]):
            jsondoc = json.loads(hit.raw)
            document_id, document_contents = jsondoc["id"], jsondoc["contents"]

            # Positive Example (relevant document)
            positive_example = {
                'query_embedding': get_embedding(query,word2vec_model),
                'document_embedding': get_embedding(document_contents,word2vec_model),
                'document_id': document_id,
                'relevance': 1
            }

            # Find a random document from another query (Negative Example)
            other_query_id = random.choice([q_id for q_id in topics if q_id != topic_id])
            other_query = topics[other_query_id]['title']
            other_hits = searcher.search(other_query)
            jsondoc_neg = random.choice([json.loads(hit.raw) for hit in other_hits])
            document_id_neg, document_contents_neg = jsondoc_neg["id"], jsondoc_neg["contents"]
            negative_example = {
                'query_embedding': get_embedding(query,word2vec_model),
                'document_embedding': get_embedding(document_contents_neg,word2vec_model),
                'document_id': document_id_neg,
                'relevance': 0
            }

            # Choose a random document from the same query as the anchor (Anchor Example)
            anchor_hit = random.choice(hits)
            jsondoc_anchor = json.loads(anchor_hit.raw)
            document_id_anchor, document_contents_anchor = jsondoc_anchor["id"], jsondoc_anchor["contents"]
            anchor_example = {
                'query_embedding': get_embedding(query,word2vec_model),
                'document_embedding': get_embedding(document_contents_anchor,word2vec_model),
                'document_id': document_id_anchor,
                'relevance': 1
            }

            # Add the triplet to the dataset
            dataset.append((anchor_example, positive_example, negative_example))

    return dataset


def save_dataset_to_csv(dataset, file_name):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Query Embedding', 'Document Embedding', 'Document ID', 'Relevance', 'Type'])

        for triplet in dataset:
            for example_type, example in zip(['anchor', 'positive', 'negative'], triplet):
                # Directly convert query_embedding to string
                query_embedding_str = ','.join(map(str, example['query_embedding']))

                # Directly convert document_embedding to string
                document_embedding_str = ','.join(map(str, example['document_embedding']))

                # Write the row to the CSV
                writer.writerow([query_embedding_str, document_embedding_str, example['document_id'], example['relevance'], example_type])


def read_dataset_from_csv(file_name):
    dataset = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header

        # Initialize a temporary list to store the triplet
        triplet = []

        for row in reader:
            example = {
                'query_embedding': None if row[0] == 'None' else np.array(list(map(float, row[0].split(',')))),
                'document_embedding': None if row[1] == 'None' else np.array(list(map(float, row[1].split(',')))),
                'document_id': row[2],
                'relevance': int(row[3]),
                'type': row[4]
            }

            # Add the example to the triplet
            triplet.append(example)

            # When the triplet is complete (3 examples), add it to the dataset
            if len(triplet) == 3:
                dataset.append(tuple(triplet))
                triplet = []  # Reset the triplet list for the next set of examples

    return dataset