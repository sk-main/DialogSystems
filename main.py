import json

import numpy as np

from compare_clustering_solutions import evaluate_clustering
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from collections import defaultdict
import csv


def embed_requests(requests):
    # Load Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Convert requests to embeddings
    request_embeddings = model.encode(requests, show_progress_bar=True)
    request_embeddings = request_embeddings.reshape(-1, 1)
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
    cluster_centers = [request_embeddings[0]]
    similarity_threshold = 0.5

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
        reader = csv.reader(file)
        for row in reader:
            requests = row[1]



    # Embed requests
    request_embeddings = embed_requests(requests)

    # Cluster requests
    clusters, cluster_centers = cluster_requests(request_embeddings, min_size)

    # Label clusters
    cluster_labels = label_clusters(requests, clusters, cluster_centers, request_embeddings)

    cluster_labels_str = {str(key): value for key, value in cluster_labels.items()}

    # Prepare JSON structure
    json_data = {"cluster_list": []}
    for label, indices in clusters.items():
        cluster_data = {}
        cluster_data["cluster_name"] = requests[indices[0]]  # Use the first request in the cluster as the cluster name
        cluster_data["requests"] = [requests[idx] for idx in indices]  # List of requests in the cluster
        json_data["cluster_list"].append(cluster_data)

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
