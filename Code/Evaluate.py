import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, adjusted_rand_score
from itertools import combinations

class ClusteringEvaluator:
    def __init__(self, true_labels_file: str, df: pd.DataFrame, clusters: np.ndarray):
        """
        Initialize the ClusteringEvaluator with true labels and clustering results.

        Args:
            true_labels_file (str): Path to the file containing true cluster labels.
            df (pd.DataFrame): The DataFrame containing sequences and their assigned clusters.
            clusters (np.ndarray): The array of predicted cluster labels.
        """
        self.true_labels_file = true_labels_file
        self.df = df
        self.clusters = clusters
        self.true_cluster_dict = self.load_true_labels()
        self.prepare_dataframe()

    def load_true_labels(self) -> dict:
        """
        Load true cluster labels from the file and create a mapping of sequence_id to true clusters.

        Returns:
            dict: Mapping of sequence_id to true cluster id.
        """
        with open(self.true_labels_file, 'r') as f:
            true_results = f.read().splitlines()

        true_cluster_dict = {}
        for cluster_id, line in enumerate(true_results):
            for seq in line.split():
                true_cluster_dict[seq] = cluster_id

        return true_cluster_dict

    def prepare_dataframe(self):
        """
        Add true cluster and predicted cluster results to the DataFrame and clean it up.
        """
        self.df['clustered_result'] = self.clusters[:len(self.df)]
        self.df['true_result'] = self.df['sequence_id'].map(self.true_cluster_dict)
        self.df.dropna(subset=['true_result'], inplace=True)
        self.df['true_result'] = self.df['true_result'].astype(int)

    def pairwise_precision_recall(self) -> tuple:
        """
        Calculate pairwise precision and recall between true and predicted clusters.

        Returns:
            tuple: Precision and recall values.
        """
        true_pairs = set()
        pred_pairs = set()

        for label in set(self.df['true_result']):
            indices = list(self.df[self.df['true_result'] == label].index)
            true_pairs.update(combinations(indices, 2))

        for label in set(self.df['clustered_result']):
            indices = list(self.df[self.df['clustered_result'] == label].index)
            pred_pairs.update(combinations(indices, 2))

        tp = len(true_pairs & pred_pairs)
        fp = len(pred_pairs - true_pairs)
        fn = len(true_pairs - pred_pairs)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return precision, recall

    def closeness_precision_recall(self) -> tuple:
        """
        Calculate closeness precision and recall between true and predicted clusters.

        Returns:
            tuple: Precision and recall values.
        """
        tp = 0
        fp = 0
        fn = 0

        for label in set(self.df['clustered_result']):
            pred_indices = set(self.df[self.df['clustered_result'] == label].index)
            best_match = None
            best_overlap = 0
            for t_label in set(self.df['true_result']):
                true_indices = set(self.df[self.df['true_result'] == t_label].index)
                overlap = len(pred_indices & true_indices)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = t_label

            if best_match is not None:
                tp += best_overlap
                fn += len(pred_indices - true_indices)
                fp += len(true_indices - pred_indices)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return precision, recall

    def singleton_metrics(self) -> tuple:
        """
        Calculate singleton retention and singleton fraction metrics.

        Returns:
            tuple: Singleton retention and singleton fraction values.
        """
        unique, counts = np.unique(self.clusters, return_counts=True)
        cluster_counts = dict(zip(unique, counts))

        singletons = [count for count in cluster_counts.values() if count == 1]
        total_sequences = len(self.clusters)
        total_clusters = max(self.clusters)

        singleton_retention = len(singletons) / total_sequences
        singleton_fraction = len(singletons) / total_clusters

        return singleton_retention, singleton_fraction

    def evaluate(self):
        """
        Evaluate clustering performance and print the results.
        """
        pairwise_precision, pairwise_recall = self.pairwise_precision_recall()
        print(f'Pairwise Precision: {pairwise_precision:.4f}')
        print(f'Pairwise Recall: {pairwise_recall:.4f}')

        closeness_precision, closeness_recall = self.closeness_precision_recall()
        print(f'Closeness Precision: {closeness_precision:.4f}')
        print(f'Closeness Recall: {closeness_recall:.4f}')

        singleton_retention, singleton_fraction = self.singleton_metrics()
        print(f'Singleton Retention: {singleton_retention:.4f}')
        print(f'Singleton Fraction: {singleton_fraction:.4f}')

if __name__ == "__main__":
    # Example usage
    file_path = '/root/Simulated_ground_truth_label/l0016_mono_true_cluster.txt'
    df = pd.read_csv('/root/Simulated_Data_input/I0026_mono_vquest_airr.tsv', sep='\t')
    clusters = np.load('/path/to/clusters.npy') 

    evaluator = ClusteringEvaluator(true_labels_file=file_path, df=df, clusters=clusters)
    evaluator.evaluate()
