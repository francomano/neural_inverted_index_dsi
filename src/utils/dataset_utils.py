import json
import random
import pyserini.search

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