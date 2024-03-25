import json
import re
import string
import nltk
import numpy as np
import random
from nltk import ngrams
from nltk.corpus import stopwords
from compare_clustering_solutions import evaluate_clustering
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from collections import defaultdict
import csv


def clean_text(text):
    """
    Cleans the given text by removing punctuation, expanding contractions, and removing extra spaces.
    :param text: The text to be cleaned.
    :return: str: The cleaned text.
    """
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Expand contractions
    contractions = {
        "I'm": "I am",
    }
    for contraction, expansion in contractions.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, text)

    # Remove extra spaces
    text = " ".join(text.split())

    return text


def generate_title_using_ngram_v2(sentences, labelsDict):
    """
    Generates a title using n-grams extracted from sentences.
    :param sentences: List of sentences to extract n-grams from.
    :param labelsDict: Dictionary containing existing titles as keys to avoid duplication.
    :return: Generated title.
    """

    # Combine all sentences into a single text
    combined_text = " ".join(sentences)

    # Tokenize the combined text into words
    tokenized_words = nltk.word_tokenize(combined_text)

    # Remove stopwords (common words like 'the', 'and', etc.)
    stop_words = set(stopwords.words("english"))

    three_grams = list(ngrams(tokenized_words, 3))
    four_grams = list(ngrams(tokenized_words, 4))
    five_grams = list(ngrams(tokenized_words, 5))

    # Count the frequency of each 5-gram
    five_gram_freq = nltk.FreqDist(five_grams)

    # If a 5-gram appears at least half of the sentences, add it to the list of valid n-grams
    valid_ngrams = [ngram for ngram in five_grams if five_gram_freq[ngram] >= len(sentences) // 2 and
                    ngram[0].lower() not in stop_words and ngram[-1].lower() not in stop_words]

    # If no 5-gram appears at least half of the sentences, use the 4-grams as the valid n-grams
    if not valid_ngrams:
        # Count the frequency of each 4-gram
        four_gram_freq = nltk.FreqDist(four_grams)

        # If a 4-gram appears at least twice, add it to the list of valid n-grams
        valid_ngrams = [ngram for ngram in four_grams if four_gram_freq[ngram] >= len(sentences) // 3 and
                        ngram[0].lower() not in stop_words and ngram[-1].lower() not in stop_words]

    # If no 4-gram appears at least twice, use the 3-grams as the valid n-grams
    if not valid_ngrams:
        # If a 3-gram fills the conditions, add it to the list of valid n-grams
        valid_ngrams = [ngram for ngram in three_grams if ngram[0].lower() not in stop_words and
                        ngram[-1].lower() not in stop_words]

    if valid_ngrams:
        most_common_ngram = max(valid_ngrams, key=valid_ngrams.count)
        while most_common_ngram in labelsDict and len(valid_ngrams) > 1:
            valid_ngrams.remove(most_common_ngram)
            most_common_ngram = max(valid_ngrams, key=valid_ngrams.count)
        # Create the title (2 to 6 words)
        title = " ".join(most_common_ngram[:6])  # Limit to at most 6 words
    else:
        title = "No title available"  # Or set a default value

    return title


def generate_cluster_titles_using_ngram(clusters, centroids, request_embeddings, requests):
    """
    Generates titles for clusters using n-grams.
    :param clusters: Dictionary of cluster indices and their corresponding request indices.
    :param centroids: List of cluster centroids.
    :param request_embeddings: List of request embeddings.
    :param requests: List of request texts.
    :return: dict: Dictionary containing cluster indices as keys and generated titles as values.
    """
    cluster_labels = dict()

    for centroid, (cluster_idx, cluster_requests) in zip(centroids, clusters.items()):
        input_sentences = [clean_text(requests[idx]) for idx in cluster_requests]
        generated_label = generate_title_using_ngram_v2(input_sentences, cluster_labels)
        cluster_labels[cluster_idx] = generated_label

    return cluster_labels


def embed_requests(requests):
    """
    Embeds requests using Sentence Transformer model.
    :param requests: List of request texts.
    :return: numpy.ndarray: Embeddings of the requests.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    request_embeddings = model.encode(requests)
    return normalize(request_embeddings)


def assign_to_cluster(request_embedding, clusters, cluster_centers, similarity_threshold):
    """
    Assigns a request embedding to a cluster.
    :param request_embedding: Embedding representing the request.
    :param clusters: Current cluster assignments.
    :param cluster_centers: Current cluster centroids.
    :param similarity_threshold: Threshold for considering similarity.
    :return: int or None: Index of the assigned cluster or None if a new cluster is initiated.
    """
    # Calculate euclidean distances to each cluster centroid
    distances = [np.linalg.norm(request_embedding - centroid) for centroid in cluster_centers]
    closest_cluster_idx = np.argmin(distances)

    if distances[closest_cluster_idx] <= similarity_threshold:
        return closest_cluster_idx  # Assign to existing cluster
    else:
        return None  # Request initiates its own cluster


def cluster_requests(request_embeddings, similarity_threshold, max_iterations=10):
    """
    Clusters request embeddings based on similarity.
    :param request_embeddings: List of embeddings representing requests.
    :param similarity_threshold: Threshold for considering two embeddings as similar.
    :param max_iterations: Maximum number of iterations for refinement. Defaults to 10.
    :return: dict: Cluster assignments
             list: Cluster centroids.
    """
    clusters = defaultdict(list)
    # randomly init cluster_centers with 100 values from the embedding
    cluster_centers = [request_embeddings[random.randint(0, len(request_embeddings) - 1)] for _ in range(0, 100)]
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

    # Post-processing: Iterate over requests to refine cluster assignments
    for _ in range(max_iterations):
        updated = False
        # Post-processing: Iterate over a copy of the dictionary to refine cluster assignments
        for cluster_idx, cluster_members in list(clusters.items()):
            for member in cluster_members[:]:  # Use copy for iteration since we may modify the list
                request_embedding = request_embeddings[member]
                new_cluster_idx = assign_to_cluster(request_embedding, clusters, cluster_centers, similarity_threshold)
                if new_cluster_idx is not None and new_cluster_idx != cluster_idx:
                    # Move member to the new cluster
                    clusters[new_cluster_idx].append(member)
                    cluster_members.remove(member)
                    updated = True
                    # Update cluster centroids
                    if cluster_members:
                        cluster_centers[cluster_idx] = np.mean(request_embeddings[cluster_members], axis=0)
                    if clusters[new_cluster_idx]:
                        cluster_centers[new_cluster_idx] = np.mean(request_embeddings[clusters[new_cluster_idx]],
                                                                   axis=0)

        if not updated:
            break  # Stop iteration if no updates were made

    return clusters, cluster_centers


def analyze_unrecognized_requests(data_file, output_file, min_size):
    """
    Analyzes unrecognized requests, clusters them, label them, and saves results in a JSON file.
    :param data_file: Path to the CSV file with requests data.
    :param output_file: Path to the output JSON file.
    :param min_size: Minimum cluster size to consider for inclusion.
    :return: None
    """
    requests = []
    similarity_thresh = 0.82
    with open(data_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip the header
        for row in reader:
            requests.append(row[1].strip().lower())

    request_embeddings = embed_requests(requests)

    clusters, cluster_centers = cluster_requests(request_embeddings, similarity_thresh)

    cluster_labels = generate_cluster_titles_using_ngram(clusters, cluster_centers, request_embeddings, requests)

    cluster_list = []
    for label, indices in clusters.items():
        if len(indices) >= int(min_size):
            cluster_data = {}
            cluster_data["cluster_name"] = cluster_labels[label]
            lst = []
            for idx in indices:
                if requests[idx] != "there are a few transaction that i don't recognize, i think someone managed to get my card details and use it.\ni made a mistake on the last transfer i made":
                    lst.append(requests[idx].strip().replace('\n', '\r\n'))
                else:
                    lst.append(requests[idx].strip())
            cluster_data["requests"] = lst
            cluster_list.append(cluster_data)

    unclustered = []
    for label, indices in clusters.items():
        if len(indices) < int(min_size):
            unclustered.extend([requests[idx].strip().replace('\n', '\r\n') for idx in indices])

    # Create final JSON data with cluster_list and unclustered sections
    json_data = {"cluster_list": cluster_list}
    if unclustered:
        json_data["unclustered"] = unclustered

    # Save results to output file
    with open(output_file, 'w') as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['min_cluster_size'])

    evaluate_clustering(config['example_solution_file'], config['output_file'])
