import json

import numpy as np

from compare_clustering_solutions import evaluate_clustering
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from collections import defaultdict


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
    cluster_centers = []

    # Assign requests to clusters
    for idx, request_embedding in enumerate(request_embeddings):
        cluster_idx = assign_to_cluster(request_embedding, clusters, cluster_centers, similarity_threshold)
        if cluster_idx is not None:
            clusters[cluster_idx].append(idx)  # Assign to existing cluster
            # Update cluster centroid
            cluster_centers[cluster_idx] = np.mean(request_embeddings[clusters[cluster_idx]], axis=0)
        else:
            # Initiate new cluster
            clusters[len(cluster_centers)].append(idx)
            cluster_centers.append(request_embedding)


    return clusters, cluster_centers










    # # Determine optimal number of clusters (optional)
    # # You may want to tune this hyperparameter based on your data
    # # For simplicity, let's use silhouette score
    # best_score = -1
    # best_k = 2
    # for k in range(2, min(11, len(request_embeddings) + 1)):
    #     kmeans = KMeans(n_clusters=k, random_state=42)
    #     cluster_labels = kmeans.fit_predict(request_embeddings)
    #     silhouette_avg = silhouette_score(request_embeddings, cluster_labels)
    #     if silhouette_avg > best_score:
    #         best_score = silhouette_avg
    #         best_k = k
    #
    # # Perform K-means clustering
    # kmeans = KMeans(n_clusters=best_k, random_state=42)
    # cluster_labels = kmeans.fit_predict(request_embeddings)
    #
    # # Assign requests to clusters
    # clusters = defaultdict(list)
    # for idx, label in enumerate(cluster_labels):
    #     clusters[label].append(idx)
    #
    # # Filter clusters based on min_size
    # filtered_clusters = {label: cluster for label, cluster in clusters.items() if len(cluster) >= int(min_size)}
    #
    # return filtered_clusters, kmeans.cluster_centers_

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


def analyze_unrecognized_requests(data_file, output_file, min_size):
    # todo: implement this function
    #  you are encouraged to break the functionality into multiple functions,
    #  but don't split your code into multiple *.py files
    #
    #  todo: the final outcome is the json file with clustering results saved as output_file

    with open(data_file, 'r') as file:
        requests = [line.strip() for line in file]

    # Embed requests
    request_embeddings = embed_requests(requests)

    # Cluster requests
    clusters, cluster_centers = cluster_requests(request_embeddings, min_size)

    # Label clusters
    cluster_labels = label_clusters(requests, clusters, cluster_centers, request_embeddings)

    cluster_labels_str = {str(key): value for key, value in cluster_labels.items()}

    # Save results to output file
    with open(output_file, 'w') as file:
        json.dump(cluster_labels_str, file, indent=4)



    pass


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['example_solution_file'])  # invocation example
    #evaluate_clustering(config['example_solution_file'], config['output_file'])
