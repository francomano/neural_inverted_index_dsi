import numpy as np


def document_embedding(doc_tokens, word2vec_model):
    #print(doc_tokens[0])
    word_embeddings = [word2vec_model.wv[word] for word in doc_tokens if word in word2vec_model.wv]
    if not word_embeddings:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word_embeddings, axis=0)


def document_embedding_sequence(doc_tokens, word2vec_model, max_tokens):
 
    np.random.shuffle(doc_tokens)

    doc_tokens = doc_tokens[:max_tokens] + ['[PAD]'] * max(0, max_tokens - len(doc_tokens))

    word_embeddings = [word2vec_model.wv[word] for word in doc_tokens if word in word2vec_model.wv]

    if not word_embeddings:
        return np.zeros(word2vec_model.vector_size)

    return np.concatenate(word_embeddings, axis=0)




