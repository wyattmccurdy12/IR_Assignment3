'''
Wyatt McCurdy
Information Retrieval Assignment 3
October 28, 2024
Dr. Behrooz Mansouri

This assignment compares fine-tuned and pre-trained versions of a re-ranking system.
The re-ranking system is made from two models - a bi-encoder which searches over all
documents and a cross-encoder which re-ranks the top 100 results from the 
bi-encoder. 

The query-document pairs will be split into train/test/validation sets
with a 80/10/10 split.

Data in the data directory is in trec format. 
documents: Answers.json
queries: topics_1.json, topics_2.json
qrels:   qrel_1.tsv (qrel_2.tsv is reserved by the instructor)

models used: 
sentence-transformers/all-MiniLM-L6-v2 for the bi-encoder
cross-encoder/ms-marco-MiniLM-L-6-v2
'''

import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import SentenceTransformer, LoggingHandler, evaluation
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
import json
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, handlers=[LoggingHandler()])

def remove_html_tags(text):
    """
    Remove HTML tags from the input text using BeautifulSoup.

    Parameters:
    text (str): The text from which to remove HTML tags.

    Returns:
    str: The text with HTML tags removed.
    """
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = re.sub(r"\\", "", text)
    return text

def load_data(queries_path, documents_path, qrels_path):
    """
    Loads the documents, queries, and qrels from the specified files.

    Parameters:
    queries_path (str): The path to the queries file.
    documents_path (str): The path to the documents file.
    qrels_path (str): The path to the qrels file.

    Returns:
    tuple: A tuple containing the documents, queries, and qrels.
    """
    with open(documents_path, 'r') as f:
        documents = json.load(f)
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    with open(qrels_path, 'r') as f:
        qrels = f.readlines()
    return documents, queries, qrels

def prepare_training_data(queries, documents, qrels):
    """
    Prepares the training data for fine-tuning.

    Parameters:
    queries (list): The list of queries.
    documents (list): The list of documents.
    qrels (list): The list of qrels.

    Returns:
    list: A list of InputExample objects for training.
    """
    query_dict = {query['Id']: query for query in queries}
    doc_dict = {doc['Id']: doc for doc in documents}
    train_examples = []

    for line in qrels:
        query_id, _, doc_id, relevance = line.strip().split()
        query = query_dict[query_id]
        doc = doc_dict[doc_id]
        query_text = f"{remove_html_tags(query['Title'])} {remove_html_tags(query['Body'])} {' '.join(query['Tags'])}"
        doc_text = remove_html_tags(doc['Text'])
        train_examples.append(InputExample(texts=[query_text, doc_text], label=int(relevance)))

    return train_examples

def fine_tune_bi_encoder(model_name, train_examples, output_path):
    """
    Fine-tunes the bi-encoder model.

    Parameters:
    model_name (str): The name of the bi-encoder model.
    train_examples (list): The list of training examples.
    output_path (str): The path to save the fine-tuned model.
    """
    model = SentenceTransformer(model_name)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model=model)

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    model.save(output_path)

def fine_tune_cross_encoder(model_name, train_examples, output_path):
    """
    Fine-tunes the cross-encoder model.

    Parameters:
    model_name (str): The name of the cross-encoder model.
    train_examples (list): The list of training examples.
    output_path (str): The path to save the fine-tuned model.
    """
    model = CrossEncoder(model_name, num_labels=1)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    model.fit(train_dataloader=train_dataloader, epochs=1, warmup_steps=100)
    model.save(output_path)

def main():
    """
    Main function to fine-tune the models.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fine-Tune Models')
    parser.add_argument('-q', '--queries', required=True, help='Path to the queries file')
    parser.add_argument('-d', '--documents', required=True, help='Path to the documents file')
    parser.add_argument('-r', '--qrels', required=True, help='Path to the qrels file')
    parser.add_argument('-be', '--bi_encoder', required=True, help='Bi-encoder model string')
    parser.add_argument('-ce', '--cross_encoder', required=True, help='Cross-encoder model string')
    parser.add_argument('-o', '--output', required=True, help='Output directory to save fine-tuned models')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    documents, queries, qrels = load_data(args.queries, args.documents, args.qrels)
    print("Data loaded successfully.")

    # Prepare training data
    print("Preparing training data...")
    train_examples = prepare_training_data(queries, documents, qrels)
    print("Training data prepared successfully.")

    # Fine-tune bi-encoder
    print("Fine-tuning bi-encoder...")
    fine_tune_bi_encoder(args.bi_encoder, train_examples, f"{args.output}/fine_tuned_bi_encoder")
    print("Bi-encoder fine-tuned successfully.")

    # Fine-tune cross-encoder
    print("Fine-tuning cross-encoder...")
    fine_tune_cross_encoder(args.cross_encoder, train_examples, f"{args.output}/fine_tuned_cross_encoder")
    print("Cross-encoder fine-tuned successfully.")

if __name__ == "__main__":
    main()