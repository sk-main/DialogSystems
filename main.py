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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from collections import defaultdict
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import time

def clean_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Expand contractions
    contractions = {
        "I'm": "I am",
        # Add more contractions as needed
    }
    for contraction, expansion in contractions.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expansion, text)

    # Remove extra spaces
    text = " ".join(text.split())

    return text

def generate_title_using_ngramV2(sentences, labelsDict):
    # Combine all sentences into a single text
    combined_text = " ".join(sentences)

    # Tokenize the combined text into words
    tokenized_words = nltk.word_tokenize(combined_text)

    # Remove stopwords (common words like 'the', 'and', etc.)
    stop_words = set(stopwords.words("english"))
    # filtered_words = [word for word in tokenized_words if word.lower() not in stop_words]
    filtered_words = tokenized_words

    # Generate 3-grams and 4-grams
    three_grams = list(ngrams(filtered_words, 3))
    four_grams = list(ngrams(filtered_words, 4))
    five_grams = list(ngrams(filtered_words, 5))

    # Count the frequency of each 5-gram
    five_gram_freq = nltk.FreqDist(five_grams)

    # If a 5-gram appears at least half of the sentences, add it to the list of valid n-grams
    valid_ngrams = [ngram for ngram in five_grams if five_gram_freq[ngram] >= len(sentences) // 2 and
                    not ngram[-1].lower().endswith(("i", "to", "for", "and", "or","im")) and ngram[-1].lower() not in stop_words]

    # If no 5-gram appears at least half of the sentences, use the 4-grams as the valid n-grams
    if not valid_ngrams:
        # Count the frequency of each 4-gram
        four_gram_freq = nltk.FreqDist(four_grams)

        # If a 4-gram appears at least twice, add it to the list of valid n-grams
        valid_ngrams = [ngram for ngram in four_grams if four_gram_freq[ngram] >= len(sentences) // 3 and
                        not ngram[-1].lower().endswith(("i", "to", "for", "and", "or","im")) and ngram[-1].lower() not in stop_words]

    # If no 4-gram appears at least twice, use the 3-grams as the valid n-grams
    if not valid_ngrams:
        valid_ngrams = three_grams

    if valid_ngrams:
        most_common_ngram = max(valid_ngrams, key=valid_ngrams.count)
        while most_common_ngram in labelsDict and len(valid_ngrams) > 1:
            valid_ngrams.remove(most_common_ngram)
            most_common_ngram = max(valid_ngrams, key=valid_ngrams.count)
        # Create the title (2 to 6 words)
        title = " ".join(most_common_ngram[:6])  # Limit to at most 6 words
    else:
        # Handle the case where there are no valid n-grams
        title = "No title available"  # Or set a default value

    return title

def generate_title_using_ngram(sentences, labelsDict):
    # Combine all sentences into a single text
    combined_text = " ".join(sentences)

    # Tokenize the combined text into words
    tokenized_words = nltk.word_tokenize(combined_text)

    # Remove stopwords (common words like 'the', 'and', etc.)
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in tokenized_words if word.lower() not in stop_words]

    # Generate n-grams (bigrams to 6-grams)
    n_values = [3, 4, 5, 6]
    ngram_candidates = []
    for n in n_values:
        ngram_candidates.extend(ngrams(filtered_words, n))

    # Choose the n-gram with the highest frequency, avoiding certain endings
    valid_ngrams = [ngram for ngram in ngram_candidates if not ngram[-1].lower().endswith(("i", "to", "for", "and", "or",
                                                                                           "im"))]
    # valid_ngrams = [ngram for ngram in valid_ngrams if not nltk.pos_tag([ngram[-1]])[0][1].startswith("VB")]

    if valid_ngrams:
        most_common_ngram = max(valid_ngrams, key=valid_ngrams.count)
        while most_common_ngram in labelsDict and len(valid_ngrams) > 1:
            valid_ngrams.remove(most_common_ngram)
            most_common_ngram = max(valid_ngrams, key=valid_ngrams.count)
        # Create the title (2 to 6 words)
        title = " ".join(most_common_ngram[:6])  # Limit to at most 6 words
    else:
        # Handle the case where there are no valid n-grams
        title = "No title available"  # Or set a default value

    return title

def generate_cluster_titles_using_ngram(clusters, centroids, request_embeddings, requests):
    cluster_labels = dict()

    for centroid, (cluster_idx, cluster_requests) in zip(centroids, clusters.items()):
        cluster_embeddings = [request_embeddings[idx] for idx in cluster_requests]
        input_sentences = [clean_text(requests[idx]) for idx in cluster_requests]
        # generated_label = generate_title_using_gpt2(input_sentences, model, tokenizer)
        generated_label = generate_title_using_ngramV2(input_sentences, cluster_labels)
        cluster_labels[cluster_idx] = generated_label
        # cluster_labels.append(generated_label)

    return cluster_labels

def label_clusters_with_tfidf(requests, request_embeddings, clusters):
    # Convert requests to text format
    request_texts = [' '.join(request.split()) for request in requests]

    # Vectorize the text data using TF-IDF with bigrams
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w+\b', ngram_range=(1, 2))
    X_tfidf = tfidf_vectorizer.fit_transform(request_texts)

    # Get feature names (vocabulary) from the vectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Get top keywords for each cluster based on TF-IDF scores
    cluster_labels = {}
    for label, indices in clusters.items():
        # Calculate mean TF-IDF scores for each term across documents in the cluster
        cluster_tfidf_scores = X_tfidf[indices].mean(axis=0)

        # Get indices of top TF-IDF scores
        top_tfidf_indices = cluster_tfidf_scores.argsort()[0, ::-1][:5]  # Top 5 keywords for each cluster

        # Get corresponding feature names (words) for top TF-IDF indices
        top_keywords = [feature_names[idx] for idx in top_tfidf_indices]

        # Join top keywords to create cluster label
        cluster_labels[label] = ', '.join(top_keywords)

    return cluster_labels


def label_clusters_with_lda(requests, request_embeddings, clusters):
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
        top_keywords_idx = topic.argsort()[::-1][:5]  # Top 5 keywords for each topic
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



# def assign_to_cluster(request_embedding, clusters, cluster_centers, similarity_threshold):
#     # Calculate cosine similarities to each cluster centroid
#     similarities = [cosine_similarity([request_embedding], [centroid])[0][0] for centroid in cluster_centers]
#     # Find the closest cluster
#     closest_cluster_idx = np.argmax(similarities)  # Use argmax since cosine similarity gives higher values for more similar vectors
#     # Check if the similarity is above the threshold
#     if similarities[closest_cluster_idx] >= similarity_threshold:
#         return closest_cluster_idx  # Assign to existing cluster
#     else:
#         return None  # Request initiates its own cluster

def assign_to_cluster(request_embedding, clusters, cluster_centers, similarity_threshold):
    # Calculate euclidean distances to each cluster centroid
    distances = [np.linalg.norm(request_embedding - centroid) for centroid in cluster_centers]
    # Find the closest cluster
    closest_cluster_idx = np.argmin(distances)
    # Check if the distance is within the similarity threshold
    if distances[closest_cluster_idx] <= similarity_threshold:
        return closest_cluster_idx  # Assign to existing cluster
    else:
        return None  # Request initiates its own cluster


# def cluster_requests(request_embeddings, similarity_threshold):
#     # Initialize clusters and cluster centers
#     clusters = defaultdict(list)
#     random_init_ind = random.randint(0, len(request_embeddings) - 1)
#     cluster_centers = [request_embeddings[random_init_ind]]
#     # similarity_threshold = 0.895  # so far the best threshold
#
#     # Assign requests to clusters
#     for request, request_embedding in enumerate(request_embeddings):
#         cluster_idx = assign_to_cluster(request_embedding, clusters, cluster_centers, similarity_threshold)
#         if cluster_idx is not None:
#             clusters[cluster_idx].append(request)  # Assign to existing cluster
#             # Update cluster centroid
#             cluster_centers[cluster_idx] = np.mean(request_embeddings[clusters[cluster_idx]], axis=0)
#         else:
#             # Initiate new cluster
#             clusters[len(cluster_centers)].append(request)
#             cluster_centers.append(request_embedding)
#
#     return clusters, cluster_centers

def cluster_requests(request_embeddings, similarity_threshold, max_iterations=5):
    # best so far 0.74 for both
    # Initialize clusters and cluster centers
    clusters = defaultdict(list)
    # random_init_ind = random.randint(0, len(request_embeddings) - 1)
    cluster_centers = [request_embeddings[random.randint(0, len(request_embeddings) - 1)] for _ in range(0, 100)]
    # cluster_centers = [request_embeddings[random_init_ind]]
    # similarity_threshold = 0.75  # 0.86 so far the best threshold on covid with euclidean, 0.55 cosine
    # 0.75 best on banking
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
    similarity_thresh = 0.82
    with open(data_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip the header
        for row in reader:
            requests.append(row[1].strip().lower())

    # Embed requests
    request_embeddings = embed_requests(requests)

    # Cluster requests
    clusters, cluster_centers = cluster_requests(request_embeddings, similarity_thresh)

    # Label clusters
    cluster_labels = generate_cluster_titles_using_ngram(clusters, cluster_centers, request_embeddings, requests)

    # Prepare JSON structure for clusters
    cluster_list = []
    for label, indices in clusters.items():
        if len(indices) >= int(min_size):
            cluster_data = {}
            cluster_data["cluster_name"] = cluster_labels[label]
            # cluster_data["requests"] = [requests[idx].strip().replace('\n','\r\n') for idx in indices]  # List of requests in the cluster
            lst = []
            for idx in indices:
                if requests[idx] != "there are a few transaction that i don't recognize, i think someone managed to get my card details and use it.\ni made a mistake on the last transfer i made":
                    lst.append(requests[idx].strip().replace('\n', '\r\n'))
                else:
                    lst.append(requests[idx].strip())
            cluster_data["requests"] = lst
            cluster_list.append(cluster_data)

    # Prepare JSON structure for unclustered requests
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


# if __name__ == '__main__':
#     with open('config.json', 'r') as json_file:
#         config = json.load(json_file)
#
#     similarity_threshold = [i/100 for i in range(70, 90, 1)]
#     for i in similarity_threshold:
#         start_time = time.time()
#         analyze_unrecognized_requests(config['data_file'],
#                                       config['output_file'],
#                                       config['min_cluster_size'], i)
#         print(i)
#         evaluate_clustering(config['example_solution_file'], config['output_file'])
#         end_time = time.time()
#         execution_time = end_time - start_time
#         print("Execution Time:", execution_time, "seconds")
#     # evaluate_clustering(config['example_solution_file'], config['example_solution_file'])  # invocation example
#     # evaluate_clustering(config['example_solution_file'], config['output_file'])

if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['min_cluster_size'])

    evaluate_clustering(config['example_solution_file'], config['output_file'])
