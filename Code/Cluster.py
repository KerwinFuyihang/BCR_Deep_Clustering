import torch
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances

class EmbeddingProcessor:
    def __init__(self, embeddings: torch.Tensor):
        """
        Initialize the EmbeddingProcessor with embeddings.

        Args:
            embeddings (torch.Tensor): The embeddings to be processed.
        """
        self.embeddings = embeddings

    def normalize_embeddings(self) -> torch.Tensor:
        """
        Normalize the embeddings to have unit norm.

        Returns:
            torch.Tensor: Normalized embeddings.
        """
        norm_embeddings = torch.nn.functional.normalize(self.embeddings, p=2, dim=1)
        return norm_embeddings

    def calculate_distance(self, norm_embeddings: torch.Tensor, distance_type: str = 'cosine') -> np.ndarray:
        """
        Calculate distances between normalized embeddings based on the specified distance type.

        Args:
            norm_embeddings (torch.Tensor): The normalized embeddings.
            distance_type (str): The type of distance metric to use ('cosine', 'euclidean', 'manhattan').

        Returns:
            numpy.ndarray: Matrix of distances.
        """
        # Convert embeddings to numpy array for use with sklearn
        norm_embeddings_np = norm_embeddings.cpu().numpy()

        if distance_type == 'cosine':
            dist = cosine_distances(norm_embeddings_np)
        elif distance_type == 'euclidean':
            dist = euclidean_distances(norm_embeddings_np)
        elif distance_type == 'manhattan':
            dist = manhattan_distances(norm_embeddings_np)
        else:
            raise ValueError(f"Unsupported distance type: {distance_type}")

        return dist
    
class Cluster:
    def __init__(self, dist: np.ndarray):
        """
        Initialize the Cluster with a distance matrix.

        Args:
            dist (numpy.ndarray): The distance matrix.
        """
        self.dist = dist
        cos_dist = sch.distance.squareform(dist, checks=False)
        self.linkage_matrix = sch.linkage(cos_dist, method='single')

    def draw_dendrogram(self):
        """
        Draw a dendrogram based on the linkage matrix.
        """
        plt.figure()
        dendrogram = sch.dendrogram(self.linkage_matrix)
        plt.title('Dendrogram')
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.show()
    
    def hierarchical_cluster(self, threshold: float) -> np.ndarray:
        """
        Perform hierarchical clustering and return cluster labels.

        Args:
            threshold (float): The threshold to apply when forming flat clusters.

        Returns:
            numpy.ndarray: Cluster labels for each sample.
        """
        clusters = sch.fcluster(self.linkage_matrix, threshold, criterion='distance')
        return clusters
