# Neural Inverted Index for Fast and Effective Information Retrieval

## Introduction
This repository contains the implementation of the Differentiable Search Index (DSI), an innovative approach to information retrieval (IR). Unlike traditional IR systems that follow the index-then-retrieve pipeline, DSI integrates indexing and retrieval within a single Transformer language model to provide a ranked list of documents in response to user queries.

## Task
The goal of this project is to build a model `f` that, given a query `q`, outputs a ranked list of document ids. The model should be a unified system trained on a corpus of documents and thereafter used to retrieve relevant documents.

## Dataset
We utilize the MS Marco dataset for this task. Detailed instructions on how to access and use the dataset with the Pyserini library are included.

## Installation
To set up your environment to use the DSI model, follow these steps:
```bash
git clone https://github.com/yourusername/neural-inverted-index.git
cd neural-inverted-index
```

## Metrics
The performance of the search index is evaluated using the following metrics:

- **Mean Average Precision (MAP)**: Reflects the mean of the average precision scores across a set of queries.
- **Precision at K**: Measures the proportion of relevant documents among the top K retrieved documents.
- **Average Precision (AP)**: The average of precision values calculated at the points in the ranking where each relevant document is retrieved.
- **Recall@1000**: Indicates the proportion of relevant items found in the top-1000 results.

## References
For more information and to understand the underlying concepts, refer to the following papers:

- [Transformer-Memory as a Differentiable Search Index](https://paperswithcode.com/paper/transformer-memory-as-a-differentiable-search)
- [Differentiable Search Index for Information Retrieval](https://arxiv.org/pdf/2305.02073.pdf)

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Authors
* **Marco Francomano** - [francomano](https://github.com/francomano)
* **Luigi Gallo** - [luigi-ga](https://github.com/luigi-ga)

See also the list of [contributors](https://github.com/luigi-ga/neural_inverted_index_dsi/graphs/contributors) who participated in this project.

