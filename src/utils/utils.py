import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

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


def train_model(dataset, model, max_epochs, batch_size=1024, split_ratio=0.8, **dataloader_kwargs):
    # Calculate split sizes
    calculate_split_sizes = lambda dataset_size, split_ratio: (int(split_ratio * dataset_size), dataset_size - int(split_ratio * dataset_size))

    # Splitting the dataset
    train_size, eval_size = calculate_split_sizes(len(dataset), split_ratio)
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # Creating dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **dataloader_kwargs)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, **dataloader_kwargs)

    # Training the model
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(model, train_dataloader, eval_dataloader)




