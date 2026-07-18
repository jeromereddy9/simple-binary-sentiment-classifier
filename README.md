# Sentiment Classifier Report

**Custom Classifier v1**

---

## Introduction

This project presents the first iteration of a custom binary sentiment classifier that uses trainable word embeddings as its primary learned representation. The model is designed to classify movie reviews as either positive or negative while maintaining a relatively simple architecture, allowing for a deeper understanding of the underlying principles of natural language processing (NLP) and neural network-based text classification.

The model takes inspiration from the Continuous Bag of Words (CBOW) approach to word representation learning in the sense that the embedding vectors are not pre-trained; instead, they are learned during the training process. By learning distributed word representations directly from the corpus, the model is able to capture semantic information that is specific to the training data.

This method of representation learning is particularly interesting because it explores content-specific learning and demonstrates that meaningful semantic relationships can emerge from relatively simple architectures. Although this first iteration, as will be discussed later in this report, has several notable limitations, it presents an interesting perspective that warrants further investigation.

The primary objective of this project is not to reinvent the wheel, but rather to gain a deeper understanding of fundamental concepts in natural language processing and neural networks by exploring existing ideas from different perspectives and implementing them from first principles.

---

## Problem Statement

The amount of semantic information that can be extracted from a corpus is not necessarily proportional to the complexity of model architecture used. While larger and more sophisticated models have proven to perform well across many NLP tasks, simpler models are also capable of producing respectable semantic representations of these corpora.

This project seeks to investigate how much semantic information can be captured using a relatively simple sentiment classification architecture based primarily on trainable word embeddings. In addition to being a learning exercise this project also serves to provide insight into the capabilities and limitations of lightweight models and to establish a foundation on which more sophisticated iterations can be built.

---

## Dataset

The [IMDB Movie Reviews dataset](https://ai.stanford.edu/~amaas/data/sentiment/) was used to train, validate, and test the first iteration of the custom sentiment classifier and to compare its performance against a baseline sentiment classifier. This dataset was selected because it is a widely used benchmark for binary sentiment classification tasks and provides a reliable basis for evaluating model performance.

The dataset consists of movie reviews paired with sentiment labels, where each review is classified as either positive or negative. Its relatively simple structure makes it well suited for experimentation with text preprocessing techniques, representation learning, and binary classification models.

---

## Methodology

### i) Preprocessing

Preprocessing was performed using a dedicated `Preprocessor` class responsible for cleaning and vectorising the raw data, as well as partitioning the dataset into training, validation, and test sets.

The text cleaning stage uses a custom standardisation method that converts all text to lowercase and removes all special characters, with the exception of question marks (`?`) and exclamation marks (`!`). These punctuation marks were retained because they may convey additional sentiment information, such as emphasis or emotional intensity.

Following the cleaning stage, the text is vectorised using Keras' `TextVectorization` layer. The custom cleaning function is supplied to the layer through its `standardize` parameter, ensuring that the same preprocessing pipeline is applied consistently across the entire dataset.

Finally, the processed data is partitioned into training, validation, and test sets using Scikit-learn's `train_test_split` function, allowing for model training, hyperparameter tuning, and unbiased performance evaluation.

### ii) Custom Architecture v1

The core of the architecture is a `Custom_Classifier` class that encapsulates the embedding logic, loss calculation, training procedure, activation functions, classification logic, and prediction pipeline. The class is designed to work closely with an instance of the `Preprocessor` class, and several design decisions were made with this dependency in mind.

The defining characteristic of the custom architecture is that learning occurs solely through the embedded vocabulary. Initially, the embedding matrix is randomly initialised, with each word represented by a vector of dimension `k`, whose values are sampled from the interval `[-0.1, 0.1]`.

The embedding matrix can be viewed as the "memory" or "knowledge base" of the model. During training, these embeddings are updated as the model learns patterns and relationships present in the corpus, allowing the learned representations to become highly specific to the dataset.

For a given sentence, each token is replaced by its corresponding embedding vector. Mean pooling is then applied by averaging across the embeddings of all words in the sentence, producing a single fixed-length representation of the input sequence.

The pooled representation is then passed through a sigmoid activation function. The resulting output values are averaged to produce a single sentiment score. Classification is performed by applying a threshold to this score, where:

- **Score > 0.5** : Positive sentiment
- **Score ≤ 0.5** : Negative sentiment

The model is trained using Binary Cross-Entropy with Logits Loss (`BCEWithLogitsLoss`). During backpropagation, the gradient is computed as:
