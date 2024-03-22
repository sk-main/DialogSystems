import json

import numpy as np
import random
from compare_clustering_solutions import evaluate_clustering
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from collections import defaultdict
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import csv


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


# def cluster_requests(request_embeddings, min_size):
#     # Initialize clusters and cluster centers
#     clusters = defaultdict(list)
#     random_init_ind = random.randint(0, len(request_embeddings) - 1)
#     cluster_centers = [request_embeddings[random_init_ind]]
#     similarity_threshold = 0.895  # so far the best threshold
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

def cluster_requests(request_embeddings, min_size, max_iterations=10):
    # Initialize clusters and cluster centers
    clusters = defaultdict(list)
    random_init_ind = random.randint(0, len(request_embeddings) - 1)
    cluster_centers = [request_embeddings[random_init_ind]]
    similarity_threshold = 0.86  # 0.86 so far the best threshold

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
        for cluster_idx, cluster_members in clusters.items():
            for member in cluster_members[:]:  # Use copy for iteration since we may modify the list
                request_embedding = request_embeddings[member]
                new_cluster_idx = assign_to_cluster(request_embedding, clusters, cluster_centers, similarity_threshold)
                if new_cluster_idx is not None and new_cluster_idx != cluster_idx:
                    # Move member to the new cluster
                    clusters[new_cluster_idx].append(member)
                    cluster_members.remove(member)
                    updated = True
                    # Update cluster centroids
                    cluster_centers[cluster_idx] = np.mean(request_embeddings[cluster_members], axis=0)
                    cluster_centers[new_cluster_idx] = np.mean(request_embeddings[clusters[new_cluster_idx]], axis=0)
        if not updated:
            break  # Stop iteration if no updates were made

    # Filter clusters with less than min_size values
    # filtered_clusters = {idx: members for idx, members in clusters.items() if len(members) >= min_size}
    # unclustered = [member for members in clusters.values() for member in members if len(members) < min_size]

    return clusters, cluster_centers  #, unclustered


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
    clusters, cluster_centers = cluster_requests(request_embeddings, min_size, 50)

    # Label clusters
    cluster_labels = temp_label(clusters)

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
    # evaluate_clustering(config['example_solution_file'], config['example_solution_file'])  # invocation example
    evaluate_clustering(config['example_solution_file'], config['output_file'])
