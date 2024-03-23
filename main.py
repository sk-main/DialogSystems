import json

import numpy as np
import random
from compare_clustering_solutions import evaluate_clustering
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from collections import defaultdict
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from scipy.spatial.distance import cosine
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import MarianMTModel, MarianTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer



def generate_title_with_bart(sentences, model, tokenizer):
    # Concatenate sentences with a delimiter
    input_text = " ".join(sentences)

    # Tokenize input text
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).input_ids

    # Generate summary
    summary_ids = model.generate(input_ids, max_length=10, num_beams=1, early_stopping=True)

    # Decode summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

def generate_cluster_titles_with_bart(clusters, centroids, request_embeddings, requests):
    # Load pre-trained BART-large model and tokenizer
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    cluster_labels = []

    for centroid, (cluster_idx, cluster_requests) in zip(centroids, clusters.items()):
        cluster_embeddings = [request_embeddings[idx] for idx in cluster_requests]
        input_sentences = [requests[idx] for idx in cluster_requests]
        generated_label = generate_title_with_bart(input_sentences, model, tokenizer)
        cluster_labels.append(generated_label)

    return cluster_labels





def generate_label(cluster_requests, requests, model, tokenizer):
    input_sentences = [requests[idx] for idx in cluster_requests]
    # Encode input sentences
    input_ids = tokenizer(input_sentences, return_tensors="pt", padding=True, truncation=True).input_ids

    # Generate labels
    generated_ids = model.generate(input_ids, max_length=20, num_return_sequences=1)

    # Decode generated labels
    generated_labels = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_labels

def generate_cluster_labels_using_transformers(clusters, centroids, request_embeddings, requests):
    # Load a pretrained XLM model for translation
    model_name = "Helsinki-NLP/opus-mt-en-de"  # Example: translate from English to German
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    cluster_labels = []

    for centroid, (cluster_idx, cluster_requests) in zip(centroids, clusters.items()):
        cluster_embeddings = [request_embeddings[idx] for idx in cluster_requests]
        generated_label = generate_label(cluster_requests, requests, model, tokenizer)
        cluster_labels.append(generated_label)

    return cluster_labels



def closest_sentence_to_centroid(cluster_requests, cluster_embeddings, centroid, requests):
    closest_idx = None
    min_distance = float('inf')

    for idx, embedding in enumerate(cluster_embeddings):
        distance = cosine(embedding, centroid)  # Calculate cosine similarity
        if distance < min_distance:
            min_distance = distance
            closest_idx = idx

    closest_sentence = requests[cluster_requests[closest_idx]]
    return closest_sentence

def generate_cluster_labels(clusters, centroids, request_embeddings, requests):
    cluster_labels = []

    for centroid, (cluster_idx, cluster_requests) in zip(centroids, clusters.items()):
        cluster_embeddings = [request_embeddings[idx] for idx in cluster_requests]
        closest_label = closest_sentence_to_centroid(cluster_requests, cluster_embeddings, centroid, requests)
        cluster_labels.append(closest_label)

    return cluster_labels


def label_clusters_with_closest_to_centroid(requests, clusters, cluster_centers, request_embeddings):
    # Label clusters
    cluster_labels = {}
    for label, indices in clusters.items():
        # Compute centroid representation of the cluster
        centroid_idx = cluster_centers[label]
        centroid_embedding = centroid_idx.reshape(1, -1)
        # Find nearest request to centroid
        nearest_request_idx = indices[0]
        nearest_distance = float('inf')
        for idx in indices:
            dist = np.linalg.norm(request_embeddings[idx] - centroid_embedding)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_request_idx = idx
        # Label the cluster with the nearest request
        cluster_labels[label] = requests[nearest_request_idx]

    return cluster_labels
def label_clusters_with_tfidf(requests, clusters):
    # Convert requests to text format
    request_texts = [' '.join(request.split()) for request in requests]

    # Vectorize the text data using TF-IDF with bigrams
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w+\b', ngram_range=(1, 2))
    X_tfidf = tfidf_vectorizer.fit_transform(request_texts)

    # Get feature names (vocabulary) from the vectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Get top keywords for each cluster based on TF-IDF scores
    cluster_labels = {}

    # for i, req in enumerate(requests):
    #     print(i, " ", req)

    # Calculate TF-IDF scores for each cluster
    for label, cluster in clusters.items():
        print(label, " ", cluster)
        # Get indices of requests in the current cluster
        # cluster_indices = [i for i, req in enumerate(requests) if req in clusters]
        cluster_indices = cluster

        # print(cluster_indices)

        # Check if there are valid indices for the current cluster
        if cluster_indices:
            # Get TF-IDF scores for the current cluster
            cluster_tfidf = X_tfidf[cluster_indices]

            # Calculate mean TF-IDF scores for each term across documents in the cluster
            cluster_tfidf_scores = np.asarray(cluster_tfidf.mean(axis=0)).ravel()

            # Get indices of top TF-IDF scores
            top_tfidf_indices = cluster_tfidf_scores.argsort()[::-1][:5]  # Top 5 keywords for each cluster

            # Get corresponding feature names (words) for top TF-IDF indices
            top_keywords = [feature_names[idx] for idx in top_tfidf_indices]

            # Join top keywords to create cluster label
            cluster_labels[label] = ' '.join(top_keywords)
        else:
            # Handle case where no requests are found in the cluster
            cluster_labels[label] = "No requests found in the cluster"

    return cluster_labels


def label_clusters_with_lda(requests, clusters):
    # Convert requests to text format
    request_texts = [' '.join(request.split()) for request in requests]

    # Vectorize the text data using a simple bag-of-words approach
    # vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    vectorizer = CountVectorizer(stop_words='english', token_pattern=r'\b\w+\b')

    X = vectorizer.fit_transform(request_texts)

    # Apply Latent Dirichlet Allocation (LDA) to discover topics
    lda = LatentDirichletAllocation(n_components=len(clusters), random_state=42)
    lda.fit(X)

    # Get the most probable words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = []
    for topic_idx, topic in enumerate(lda.components_):
        top_keywords_idx = topic.argsort()[::-1][:2]  # Top 2 keywords for each topic
        top_keywords = [feature_names[i] for i in top_keywords_idx]
        topic_keywords.append(top_keywords)

    # Assign topic keywords as labels to clusters
    cluster_labels = {}
    for label, indices in clusters.items():
        cluster_labels[label] = ', '.join(topic_keywords[label])

    return cluster_labels


def embed_requests(requests):
    # Load Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Convert requests to embeddings
    request_embeddings = model.encode(requests, show_progress_bar=True)
    # Normalize embeddings
    return normalize(request_embeddings)


def assign_to_cluster(request_embedding, clusters, cluster_centers, similarity_threshold):
    # Calculate distances to each cluster centroid
    distances = [np.linalg.norm(request_embedding - centroid) for centroid in cluster_centers]
    # Find the closest cluster
    closest_cluster_idx = np.argmin(distances)
    # Check if the distance is within the similarity threshold
    if distances[closest_cluster_idx] <= similarity_threshold:
        return closest_cluster_idx  # Assign to existing cluster
    else:
        return None  # Request initiates its own cluster


def cluster_requests(request_embeddings, min_size):
    # Initialize clusters and cluster centers
    clusters = defaultdict(list)
    random_init_ind = random.randint(0, len(request_embeddings) - 1)
    cluster_centers = [request_embeddings[random_init_ind]]
    similarity_threshold = 0.895  # so far the best threshold

    # Assign requests to clusters
    for request, request_embedding in enumerate(request_embeddings):
        cluster_idx = assign_to_cluster(request_embedding, clusters, cluster_centers, similarity_threshold)
        if cluster_idx is not None:
            clusters[cluster_idx].append(request)  # Assign to existing cluster
            # Update cluster centroid
            cluster_centers[cluster_idx] = np.mean(request_embeddings[clusters[cluster_idx]], axis=0)
        else:
            # Initiate new cluster
            clusters[len(cluster_centers)].append(request)
            cluster_centers.append(request_embedding)

    return clusters, cluster_centers


def label_clusters(requests, clusters, cluster_centers, request_embeddings):
    # Label clusters
    cluster_labels = {}
    for label, indices in clusters.items():
        # Compute centroid representation of the cluster
        centroid_idx = cluster_centers[label]
        centroid_embedding = centroid_idx.reshape(1, -1)
        # Find nearest request to centroid
        nearest_request_idx = indices[0]
        nearest_distance = float('inf')
        for idx in indices:
            dist = np.linalg.norm(request_embeddings[idx] - centroid_embedding)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_request_idx = idx
        # Label the cluster with the nearest request
        cluster_labels[label] = requests[nearest_request_idx]

    return cluster_labels


def temp_label(clusters):
    labeled_clusters = {}
    label_counter = 0
    for cluster_idx, requests in clusters.items():
        # Only take the index of the first request
        if requests:  # Check if the cluster is not empty
            labeled_clusters[label_counter] = [requests[0]]
        else:
            labeled_clusters[label_counter] = []  # Empty cluster
        label_counter += 1

    return labeled_clusters


def analyze_unrecognized_requests(data_file, output_file, min_size):
    # todo: implement this function
    #  you are encouraged to break the functionality into multiple functions,
    #  but don't split your code into multiple *.py files
    #
    #  todo: the final outcome is the json file with clustering results saved as output_file
    requests = []
    with open(data_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip the header
        for row in reader:
            requests.append(row[1].strip().lower())

    # Embed requests
    request_embeddings = embed_requests(requests)

    # Cluster requests
    clusters, cluster_centers = cluster_requests(request_embeddings, min_size)

    # Label clusters
    # cluster_labels = label_clusters_with_lda(requests, clusters)
    cluster_labels = generate_cluster_titles_with_bart(clusters, cluster_centers, request_embeddings, requests)

    # Prepare JSON structure for clusters
    cluster_list = []
    for label, indices in clusters.items():
        if len(indices) >= int(min_size):
            cluster_data = {}
            cluster_data["cluster_name"] = cluster_labels[label]  # Use LDA topic keywords as the cluster name
            cluster_data["requests"] = [requests[idx] for idx in indices]  # List of requests in the cluster
            cluster_list.append(cluster_data)

    print(len(cluster_list))
    # Prepare JSON structure for unclustered requests
    unclustered = []
    for label, indices in clusters.items():
        if len(indices) < int(min_size):
            unclustered.extend([requests[idx] for idx in indices])

    # Create final JSON data with cluster_list and unclustered sections
    json_data = {"cluster_list": cluster_list}
    if unclustered:
        json_data["unclustered"] = unclustered

    # Save results to output file
    with open(output_file, 'w') as file:
        json.dump(json_data, file, indent=4)


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['example_solution_file'])  # invocation example
    #evaluate_clustering(config['example_solution_file'], config['output_file'])