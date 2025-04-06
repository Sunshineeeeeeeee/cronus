import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import SpectralClustering
import warnings
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import itertools
from sklearn.mixture import GaussianMixture
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components
from gudhi import SimplexTree
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from scipy.linalg import svd
from concurrent.futures import ThreadPoolExecutor
import functools

# Suppress warnings
warnings.filterwarnings('ignore')


class TopologicalDataAnalyzer:
    """
    Implements optimized topological data analysis for volatility regime detection.
    Specialized for market microstructure data (sequential tick data).

    Core Pipeline:
    1. Compute temporally weighted distance matrix with information theory enhancement
    2. Build filtration using sparse directed network representation
    3. Compute persistent homology using optimized simplex tree structure
    4. Apply zigzag persistence with sliding windows to preserve sequential nature
    5. Extract topological features for regime classification
    """

    def __init__(self, feature_array, feature_names, timestamp_array=None):
        """
        Initialize the topological data analyzer.

        Parameters:
        -----------
        feature_array : numpy.ndarray
            Array of shape (n_samples, n_features) containing features
        feature_names : list
            List of feature names
        timestamp_array : numpy.ndarray or None
            Array of timestamps for temporal ordering
        """
        self.features = feature_array
        self.feature_names = feature_names
        self.timestamps = timestamp_array

        self.n_samples, self.n_features = self.features.shape

        # If timestamps not provided, use sample indices
        if self.timestamps is None:
            self.timestamps = np.arange(self.n_samples)

        # Initialize attributes for later use
        self.distance_matrix = None
        self.temporal_distance_matrix = None
        self.regime_labels = None

        # Attributes for Persistent Homology
        self.persistence_diagrams = None
        self.betti_curves = None
        self.persistence_landscapes = None
        self.simplex_tree = None

        # Attributes for Zigzag Persistence
        self.zigzag_diagrams = None

    def compute_temporally_weighted_distance(
            self,
            alpha=0.5,
            beta=0.1,
            lambda_info=1.0,
            mi_matrix=None,
            transfer_entropy=None,
            chunk_size=1000):
        """
        Compute a distance matrix with temporal weighting and information theory enhancement.

        Parameters:
        -----------
        alpha : float
            Temporal weighting factor (0-1)
        beta : float
            Decay rate for temporal component
        lambda_info : float
            Weight for information theory enhancement (0-1)
        mi_matrix : numpy.ndarray or None
            Mutual information matrix
        transfer_entropy : numpy.ndarray or None
            Transfer entropy matrix
        chunk_size : int
            Size of chunks for processing large datasets

        Returns:
        --------
        numpy.ndarray
            Temporal weighted distance matrix
        """
        import numpy as np
        from scipy.spatial.distance import pdist, squareform

        # Initialize start_time for timing this method
        start_time = time.time()

        print("Computing temporally-weighted distance matrix...")

        # Print dimensions of the feature array
        print(f"Feature array dimensions for distance computation: {self.features.shape}")
        print(f"Number of feature names: {len(self.feature_names)}")

        # Print statistics about MI and TE matrices
        print("\n=== Information Theory Matrices Diagnostics ===")

        if mi_matrix is not None:
            print(f"Mutual Information Matrix:")
            print(f"  Shape: {mi_matrix.shape}")
            print(f"  Min: {np.min(mi_matrix):.6f}, Max: {np.max(mi_matrix):.6f}")
            print(f"  Mean: {np.mean(mi_matrix):.6f}, Median: {np.median(mi_matrix):.6f}")
            print(f"  Zeros: {np.sum(mi_matrix == 0)} out of {mi_matrix.size} ({np.sum(mi_matrix == 0)/mi_matrix.size:.2%})")
            print(f"  NaNs: {np.sum(np.isnan(mi_matrix))} out of {mi_matrix.size} ({np.sum(np.isnan(mi_matrix))/mi_matrix.size:.2%})")

            # Print the top 5 highest values and their feature pairs
            if not np.all(mi_matrix == 0) and not np.all(np.isnan(mi_matrix)):
                # Get flattened indices of top 5 values (excluding diagonal)
                np.fill_diagonal(mi_matrix, 0)  # Exclude self-information
                flat_indices = np.argsort(mi_matrix.flatten())[-5:][::-1]
                print(f"  Top 5 mutual information pairs:")
                for idx in flat_indices:
                    i, j = np.unravel_index(idx, mi_matrix.shape)
                    if i < len(self.feature_names) and j < len(
                            self.feature_names):
                        print(
                            f"    {self.feature_names[i]} ↔ {self.feature_names[j]}: {mi_matrix[i, j]:.6f}")
        else:
            print("Mutual Information Matrix: None")

        if transfer_entropy is not None:
            print(f"\nTransfer Entropy Matrix:")
            print(f"  Shape: {transfer_entropy.shape}")
            print(f"  Min: {np.min(transfer_entropy):.6f}, Max: {np.max(transfer_entropy):.6f}")
            print(f"  Mean: {np.mean(transfer_entropy):.6f}, Median: {np.median(transfer_entropy):.6f}")
            print(f"  Zeros: {np.sum(transfer_entropy == 0)} out of {transfer_entropy.size} ({np.sum(transfer_entropy == 0)/transfer_entropy.size:.2%})")
            print(f"  NaNs: {np.sum(np.isnan(transfer_entropy))} out of {transfer_entropy.size} ({np.sum(np.isnan(transfer_entropy))/transfer_entropy.size:.2%})")

            # Print the top 5 highest values and their feature pairs
            if not np.all(
                    transfer_entropy == 0) and not np.all(
                    np.isnan(transfer_entropy)):
                np.fill_diagonal(transfer_entropy, 0)
                flat_indices = np.argsort(
                    transfer_entropy.flatten())[-5:][::-1]
                print(f"  Top 5 transfer entropy pairs:")
                for idx in flat_indices:
                    i, j = np.unravel_index(idx, transfer_entropy.shape)
                    if i < len(
                            self.feature_names) and j < len(
                            self.feature_names):
                        print(
                            f"    {self.feature_names[i]} → {self.feature_names[j]}: {transfer_entropy[i, j]:.6f}")
            else:
                print("Transfer Entropy Matrix: None")

            print("===========================================")

        # Check for NaN values in the input and print diagnostics
        nan_counts = np.isnan(self.features).sum(axis=0)
        total_nans = np.isnan(self.features).sum()
        print(f"\nNaN value diagnostics:")
        print(f"Total NaN values: {total_nans} out of {self.features.size} ({total_nans/self.features.size:.2%})")
        print(f"NaN values per column:")
        for i, (name, count) in enumerate(zip(self.feature_names, nan_counts)):
            print(f"  {name}: {count} NaNs ({count / len(self.features):.2%})")

        if total_nans > 0:
            print("\nWARNING: NaN values found in feature array. Attempting to fix...")

            # Create a clean copy
            self.features = np.copy(self.features)

            # Handle NaN values by column (more robust than all at once)
            for i in range(self.features.shape[1]):
                col = self.features[:, i]
                nan_mask = np.isnan(col)
                if np.any(nan_mask):
                    # Calculate column mean excluding NaNs
                    col_mean = np.nanmean(col)
                    if np.isnan(col_mean):  # If entire column is NaN
                        col_mean = 0.0
                        print(
                            f"  Column {i}({
                                self.feature_names[i]}) is all NaN - filling with zeros")
                    # Replace NaNs with mean
                    self.features[nan_mask, i] = col_mean
                    print(
                        f"  Fixed column {i}({
                            self.feature_names[i]}) by replacing {
                            np.sum(nan_mask)} NaNs with mean: {
                            col_mean:.6f}")

        # Print statistics about features
        feature_means = np.mean(self.features, axis=0)
        feature_stds = np.std(self.features, axis=0)
        print(f"\nFeature statistics after preprocessing:")
        print(f"  Mean range: [{np.min(feature_means):.6f}, {np.max(feature_means):.6f}]")
        print(f"  Std range: [{np.min(feature_stds):.6f}, {np.max(feature_stds):.6f}]")

        # Convert timestamps to numerical values if available
        timestamps = None
        if self.timestamps is not None:
            if isinstance(self.timestamps[0], str):
                try:
                    from dateutil import parser
                    import pandas as pd
                    timestamps = np.array(
                        [parser.parse(ts).timestamp() for ts in self.timestamps])
                    print("Successfully parsed string timestamps to datetime objects")
                except BaseException:
                    print(
                        "Warning: Could not parse string timestamps, using indices instead")
                    timestamps = np.arange(len(self.timestamps))
            elif hasattr(self.timestamps[0], 'timestamp'):
                # Pandas/datetime timestamp objects
                timestamps = np.array([ts.timestamp()
                                      for ts in self.timestamps])
            else:
                # Assume already numerical
                timestamps = self.timestamps.astype(float)

            # Normalize timestamps to [0, 1] range
            min_ts = np.min(timestamps)
            max_ts = np.max(timestamps)
            if max_ts > min_ts:
                timestamps = (timestamps - min_ts) / (max_ts - min_ts)

        # Create feature weights based on mutual information and transfer
        # entropy
        feature_weights = np.ones(self.features.shape[1])

        # Apply mutual information weighting if available
        if mi_matrix is not None and mi_matrix.shape[0] == self.features.shape[1]:
            # Calculate global MI score for each feature (sum of MI with all
            # other features)
            mi_scores = np.sum(mi_matrix, axis=1)

            print("\nMutual Information scores before normalization:")
            for i, (name, score) in enumerate(
                    zip(self.feature_names, mi_scores)):
                print(f"  {name}: {score:.4f}")

            # Check if all MI scores are the same (which would result in
            # uniform weights)
            mi_range = np.max(mi_scores) - np.min(mi_scores)
            if mi_range < 1e-6:
                print(
                    "\nWARNING: All MI scores are nearly identical. Adding variance to scores.")
                # Add some variance to avoid uniform weights
                noise = np.random.uniform(0.1, 0.5, size=len(mi_scores))
                # Ensure volatility features get higher weights
                for i, name in enumerate(self.feature_names):
                    if 'volatil' in name.lower():
                        noise[i] = 0.8  # Higher weight for volatility
                mi_scores = mi_scores + noise

            # Normalize scores
            mi_scores = mi_scores / \
                np.max(mi_scores) if np.max(mi_scores) > 0 else mi_scores

            # Print MI weights for inspection
            print("\nMutual Information weights after normalization:")
            for i, (name, weight) in enumerate(
                    zip(self.feature_names, mi_scores)):
                print(f"  {name}: {weight:.4f}")

            # Apply MI weights to feature weights with stronger impact
            # Use a power factor to enhance differences
            power_factor = 2.0  # Square the weights to enhance differences
            feature_weights *= (1.0 + lambda_info *
                                np.power(mi_scores, power_factor))

            print("\nFeature weights after applying MI:")
            for i, (name, weight) in enumerate(
                    zip(self.feature_names, feature_weights)):
                print(f"  {name}: {weight:.4f}")

        # Apply transfer entropy weighting if available
        if transfer_entropy is not None and transfer_entropy.shape[0] == self.features.shape[1]:
            # Calculate global TE score for each feature
            te_scores = np.sum(transfer_entropy, axis=1)

            print("\nTransfer Entropy scores before normalization:")
            for i, (name, score) in enumerate(
                    zip(self.feature_names, te_scores)):
                print(f"  {name}: {score:.4f}")

            # Check if all TE scores are the same (which would result in
            # uniform weights)
            te_range = np.max(te_scores) - np.min(te_scores)
            if te_range < 1e-6:
                print(
                    "\nWARNING: All TE scores are nearly identical. Adding variance to scores.")
                # Add some variance to avoid uniform weights
                noise = np.random.uniform(0.1, 0.5, size=len(te_scores))
                # Ensure volatility features get higher weights
                for i, name in enumerate(self.feature_names):
                    if 'volatil' in name.lower():
                        noise[i] = 0.8  # Higher weight for volatility
                te_scores = te_scores + noise

            # Normalize scores
            te_scores = te_scores / \
                np.max(te_scores) if np.max(te_scores) > 0 else te_scores

            # Print TE weights for inspection
            print("\nTransfer Entropy weights after normalization:")
            for i, (name, weight) in enumerate(
                    zip(self.feature_names, te_scores)):
                print(f"  {name}: {weight:.4f}")

            # Apply TE weights to feature weights with stronger impact
            # Use a power factor to enhance differences
            power_factor = 2.0  # Square the weights to enhance differences
            feature_weights *= (1.0 + lambda_info *
                                np.power(te_scores, power_factor))

            print("\nFeature weights after applying TE:")
            for i, (name, weight) in enumerate(
                    zip(self.feature_names, feature_weights)):
                print(f"  {name}: {weight:.4f}")

        # If no information-theoretic weighting applied, add variance to
        # weights
        if (mi_matrix is None or mi_matrix.shape[0] != self.features.shape[1]) and (
                transfer_entropy is None or transfer_entropy.shape[0] != self.features.shape[1]):
            print(
                "\nNo valid MI or TE matrices provided. Using variance-based feature weights.")
            # Use feature variance as weights
            var_weights = np.std(self.features, axis=0)
            var_weights = var_weights / \
                np.max(var_weights) if np.max(var_weights) > 0 else var_weights

            # Ensure variance weights have enough spread
            if np.max(var_weights) - np.min(var_weights) < 0.3:
                print("Adding diversity to variance weights.")
                var_weights = (var_weights * 0.5) + \
                    (np.random.uniform(0, 1, size=len(var_weights)) * 0.5)

            # Apply variance-based weights
            feature_weights *= (0.5 + 1.5 * var_weights)

            # Extra emphasis on volatility features
            for i, name in enumerate(self.feature_names):
                if 'volatil' in name.lower():
                    # Double importance of volatility features
                    feature_weights[i] *= 2.0
                    print(
                        f"Extra emphasis added to volatility feature '{name}'")

            print("\nVariance-based feature weights:")
            for i, (name, weight) in enumerate(
                    zip(self.feature_names, feature_weights)):
                print(f"  {name}: {weight:.4f}")

        # Normalize weights to avoid overweighting
        if np.max(feature_weights) > 0:
            print(
                f"\nFeature weights before final normalization: min={
                    np.min(feature_weights):.4f}, max={
                    np.max(feature_weights):.4f}")

            # Ensure there's enough spread in weights
            weight_range = np.max(feature_weights) - np.min(feature_weights)
            if weight_range < 0.5:
                print(
                    "WARNING: Feature weights have low variability. Enhancing weight differences.")
                # Increase contrast in weights
                mean_weight = np.mean(feature_weights)
                feature_weights = mean_weight + \
                    (feature_weights - mean_weight) * 3.0

            # Normalize to [0.1, 1.0] range
            feature_weights = (feature_weights - np.min(feature_weights)) / \
                (np.max(feature_weights) - np.min(feature_weights))
            feature_weights = 0.1 + 0.9 * \
                feature_weights  # Scale to [0.1, 1.0]

        # Ensure no feature is completely ignored (min weight of 0.1)
        feature_weights = np.maximum(feature_weights, 0.1)

        # Final post-processing to ensure volatility features are emphasized
        for i, name in enumerate(self.feature_names):
            if 'volatil' in name.lower():
                # Ensure volatility has at least 0.8 weight
                feature_weights[i] = max(feature_weights[i], 0.8)
                print(
                    f"Ensuring volatility feature '{name}' has high weight: {
                        feature_weights[i]:.4f}")

        # Print final feature weights
        print("\nFinal feature weights after normalization:")
        for i, (name, weight) in enumerate(
                zip(self.feature_names, feature_weights)):
            print(f"  {name}: {weight:.4f}")

        print(
            f"Final feature weights: min={
                np.min(feature_weights):.4f}, max={
                np.max(feature_weights):.4f}, mean={
                np.mean(feature_weights):.4f}, std={
                    np.std(feature_weights):.4f}")

        # If the dataset is large, use chunking to compute distances
        n_samples = self.features.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))

        if n_samples > chunk_size:
            # Process in chunks for large datasets
            for i in range(0, n_samples, chunk_size):
                end_i = min(i + chunk_size, n_samples)
                for j in range(0, n_samples, chunk_size):
                    end_j = min(j + chunk_size, n_samples)

                    # Compute distance for this chunk
                    for k in range(i, end_i):
                        for l in range(j, end_j):
                            # Skip diagonal
                            if k == l:
                                continue

                            # Compute feature distance weighted by feature
                            # importance
                            feature_dist = np.abs(
                                self.features[k] - self.features[l])
                            weighted_dist = np.sum(
                                feature_dist * feature_weights)

                            # Add temporal component if timestamps are
                            # available
                            if timestamps is not None:
                                time_dist = np.abs(
                                    timestamps[k] - timestamps[l])
                                # Temporal penalty increases with time
                                # difference
                                temporal_penalty = 1.0 + alpha * \
                                    (1.0 - np.exp(-beta * time_dist))
                                weighted_dist *= temporal_penalty

                            distance_matrix[k, l] = weighted_dist
                            # Ensure symmetry
                            distance_matrix[l, k] = weighted_dist

        else:
            # Compute directly for small datasets
            for i in range(n_samples):
                for j in range(
                        i + 1, n_samples):  # Only compute upper triangle
                    # Compute feature distance weighted by feature importance
                    feature_dist = np.abs(self.features[i] - self.features[j])
                    weighted_dist = np.sum(feature_dist * feature_weights)

                    # Add temporal component if timestamps are available
                    if timestamps is not None:
                        time_dist = np.abs(timestamps[i] - timestamps[j])
                        # Temporal penalty increases with time difference
                        temporal_penalty = 1.0 + alpha * \
                            (1.0 - np.exp(-beta * time_dist))
                        weighted_dist *= temporal_penalty

                    distance_matrix[i, j] = weighted_dist
                    distance_matrix[j, i] = weighted_dist  # Ensure symmetry

        # Make diagonal zero
        np.fill_diagonal(distance_matrix, 0)

        # Check for zeros or NaNs in the distance matrix
        non_zeros = distance_matrix[~np.eye(n_samples, dtype=bool)]
        zero_count = np.sum(non_zeros == 0)
        nan_count = np.sum(np.isnan(non_zeros))

        if zero_count > 0 or nan_count > 0:
            print(
                f"Warning: Distance matrix contains {zero_count} zeros and {nan_count} NaNs(out of {
                    len(non_zeros)} off - diagonal elements)")

            # Replace NaNs with a small positive value if they exist
            if nan_count > 0:
                print(f"Replacing {nan_count} NaNs with 0.01")
                distance_matrix[np.isnan(distance_matrix)] = 0.01

        # Report distance matrix statistics
        print(
            f"Distance matrix has {
                np.sum(
                    distance_matrix > 0)} non - zero elements out of {
                n_samples *
                n_samples}")
        if np.sum(distance_matrix > 0) > 0:
            print(
                f"Non-zero distance range: [{np.min(distance_matrix[distance_matrix > 0]):.6f}, {np.max(distance_matrix):.6f}]")

        # Store the distance matrix
        self.temporal_distance_matrix = distance_matrix

        print(
            f"Distance matrix computation completed in {
                time.time() -
                start_time:.2f} seconds")
        print(f"Distance matrix shape: {distance_matrix.shape}")
        print(
            f"Final distance range: [{np.min(distance_matrix[distance_matrix > 0]):.6f}, {np.max(distance_matrix):.6f}]")

        return distance_matrix

    def _compute_weighted_chunk_distances(
            self,
            chunk_i,
            chunk_j,
            time_i,
            time_j,
            alpha,
            beta,
            feature_weights):
        """
        Helper method to compute distances between chunks of data using feature weights.

        Parameters:
        -----------
        chunk_i, chunk_j : numpy.ndarray
            Chunks of feature data
        time_i, time_j : numpy.ndarray
            Chunks of time indices
        alpha, beta : float
            Parameters for temporal weighting
        feature_weights : numpy.ndarray
            Weights for each feature dimension

        Returns:
        --------
        numpy.ndarray
            Distance matrix for the chunks
        """
        # Get dimensions
        ni = chunk_i.shape[0]
        nj = chunk_j.shape[0]

        # Initialize distances for this chunk
        dists = np.zeros((ni, nj))

        # Check if self-comparison (i == j)
        if np.array_equal(chunk_i, chunk_j):
            # Compute pairwise distances within the same chunk
            for i in range(ni):
                for j in range(i + 1, nj):
                    # Weighted Euclidean distance
                    diff = chunk_i[i] - chunk_j[j]
                    weighted_diff = diff * feature_weights
                    dist = np.sqrt(np.sum(weighted_diff * weighted_diff))

                    # Temporal weighting
                    time_diff = abs(time_i[i] - time_j[j])
                    temporal_weight = 1.0 + alpha * \
                        (1.0 - np.exp(-beta * time_diff))

                    # Store weighted distance
                    dists[i, j] = dist * temporal_weight
                    dists[j, i] = dists[i, j]  # Symmetric
        else:
            # Compute distances between different chunks
            for i in range(ni):
                for j in range(nj):
                    # Weighted Euclidean distance
                    diff = chunk_i[i] - chunk_j[j]
                    weighted_diff = diff * feature_weights
                    dist = np.sqrt(np.sum(weighted_diff * weighted_diff))

                    # Temporal weighting
                    time_diff = abs(time_i[i] - time_j[j])
                    temporal_weight = 1.0 + alpha * \
                        (1.0 - np.exp(-beta * time_diff))

                    # Store weighted distance
                    dists[i, j] = dist * temporal_weight

        return dists

    def construct_directed_network(self, epsilon=0.5, enforce_temporal=True):
        """
        Construct a directed weighted network based on the temporal distance matrix.
        This builds the proximity graph representing market microstructure relationships.

        Parameters:
        -----------
        epsilon : float
            Distance threshold for connecting points
        enforce_temporal : bool
            Whether to only create edges (X_i, X_j) if i < j (temporal ordering)

        Returns:
        --------
        networkx.DiGraph
            Directed graph representation of the data
        """
        print(f"Constructing directed network with epsilon={epsilon}...")

        if self.temporal_distance_matrix is None:
            raise ValueError("Temporal distance matrix must be computed first")

        # Create directed graph
        G = nx.DiGraph()

        # Add nodes
        for i in range(self.n_samples):
            G.add_node(
                i,
                features=self.features[i],
                timestamp=self.timestamps[i])

        # Create edges based on distance threshold and temporal ordering
        if enforce_temporal:
            # Only create edges that respect temporal ordering (i < j)
            # Convert matrices to float type to ensure compatibility
            distance_mask = (
                self.temporal_distance_matrix <= epsilon).astype(float)
            temporal_mask = np.triu(
                np.ones_like(
                    self.temporal_distance_matrix),
                k=1).astype(float)
            combined_mask = (distance_mask * temporal_mask) > 0
            rows, cols = np.where(combined_mask)
        else:
            # Create edges in both directions
            rows, cols = np.where(self.temporal_distance_matrix <= epsilon)

        # Add edges with weights
        for i, j in zip(rows, cols):
            if i != j:  # Avoid self-loops
                weight = 1.0 / max(self.temporal_distance_matrix[i, j], 1e-10)
                G.add_edge(i, j, weight=weight,
                           distance=self.temporal_distance_matrix[i, j])

        print(
            f"Constructed network with {
                G.number_of_nodes()} nodes and {
                G.number_of_edges()} edges")
        return G

    def extract_topological_features(self):
        """
        Extract topological features from the computed persistence diagrams and complex structures.
        Combines persistent homology features, zigzag persistence features, and network statistics.

        Returns:
        --------
        tuple
            (features_array, feature_names) where features_array contains the numerical values
            and feature_names contains descriptive names for each feature
        """
        features = {}

        # Extract zigzag persistence features if available
        if hasattr(
                self,
                'path_zigzag_diagrams') and self.path_zigzag_diagrams is not None:
            # Get transitions if available or create empty list
            transitions = self.path_zigzag_diagrams.get('transitions', [])

            # Use the compute_zigzag_features method to extract features
            features = self._compute_zigzag_features(
                self.path_zigzag_diagrams, transitions)
        else:
            # If zigzag features aren't available, extract basic network
            # features
            if hasattr(self, 'network'):
                # Basic network statistics
                G = self.network
                features['network_nodes'] = G.number_of_nodes()
                features['network_edges'] = G.number_of_edges()

                if G.number_of_nodes() > 1:
                    features['network_density'] = G.number_of_edges(
                    ) / (G.number_of_nodes() * (G.number_of_nodes() - 1))
                else:
                    features['network_density'] = 0.0

                # Compute average degree
                degrees = [d for _, d in G.degree()]
                if degrees:
                    features['network_degree_mean'] = sum(
                        degrees) / len(degrees)
                    features['network_degree_max'] = max(degrees)
                else:
                    features['network_degree_mean'] = 0.0
                    features['network_degree_max'] = 0.0

        # Convert to arrays
        feature_names = list(features.keys())
        feature_values = np.array([features[name] for name in feature_names])

        return feature_values, feature_names

    def compute_path_complex(self, G, max_path_length=2):
        """
        Generate a directed path complex from the graph.
        For market microstructure data, path homology is more appropriate than simplicial complexes.
        Prioritizes paths that contain information about volatility regimes.
        Uses vectorized operations for performance and avoids generating paths that don't contribute to homology.

        Parameters:
        -----------
        G : networkx.DiGraph
            Directed graph representation of the data
        max_path_length : int
            Maximum length of paths to consider

        Returns:
        --------
        dict
            Dictionary of paths by dimension
        """
        print(
            f"Computing path complex with max path length {max_path_length}...")
        start_time = time.time()

        # Initialize path complex
        path_complex = {k: [] for k in range(max_path_length + 1)}

        # Dictionary to store stats for reporting
        stats = {
            'total_candidates': 0,
            'volatility_paths': 0,
            'filtered_paths': 0,
            'homology_pruned': 0
        }

        # Use LRU cache for frequently accessed data
        import functools

        @functools.lru_cache(maxsize=10000)
        def get_successors(node):
            """Cached function to get successors of a node"""
            if node in G:
                return list(G.successors(node))
            return []

        @functools.lru_cache(maxsize=10000)
        def get_edge_data(u, v):
            """Cached function to get edge data"""
            if G.has_edge(u, v):
                return G[u][v]
            return {}

        # Add 0-simplices (vertices) - vectorized operation
        path_complex[0] = [(node,) for node in G.nodes()]

        # Create node-to-index mapping for vectorized operations
        import numpy as np
        from scipy import sparse
        from concurrent.futures import ThreadPoolExecutor

        # Create node-to-index mapping
        node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        idx_to_node = {i: node for node, i in node_to_idx.items()}
        n_nodes = len(node_to_idx)

        # Find volatility feature in graph nodes if it exists
        vol_feature = None
        for feature_name in [
            'Volatility',
            'volatility',
            'vol_adjusted_plus_di_100',
                'vol_adjusted_minus_di_100']:
            if G.nodes and feature_name in G.nodes[list(G.nodes())[0]]:
                vol_feature = feature_name
                print(f"Using {vol_feature} for volatility-based filtration")
                break

        # Extract all edges at once - vectorized operation
        edges = list(G.edges())
        if not edges:
            return path_complex

        # Create edge lookup for improved performance
        edge_lookup = {(u, v): G[u][v] for u, v in edges}

        # Vectorized processing for edges - parallelize large operations
        edge_scores = {}
        volatility_edges = []
        adj_matrix = np.zeros((n_nodes, n_nodes))

        # Precompute volatility values for all nodes if available
        vol_values = {}
        if vol_feature:
            vol_values = {
                node: G.nodes[node].get(
                    vol_feature,
                    0.0) for node in G.nodes()}

        # Process edges in parallel for large graphs
        def process_edge_batch(edge_batch):
            local_edge_scores = {}
            local_volatility_edges = []
            local_adj_updates = []

            for u, v in edge_batch:
                # Basic validity check - don't include self-loops
                if u == v:
                    continue

                # Get edge data
                edge_data = get_edge_data(u, v)
                edge_weight = edge_data.get('weight', 0.0)
                edge_distance = edge_data.get('distance', float('inf'))

                # Skip edges with very high distance
                if edge_distance > 1e10:
                    continue

                # Update adjacency matrix coordinates for later
                u_idx, v_idx = node_to_idx[u], node_to_idx[v]
                local_adj_updates.append((u_idx, v_idx))

                # Calculate volatility change if we have volatility data
                vol_change = 0.0
                is_vol_edge = False

                if vol_feature and u in vol_values and v in vol_values:
                    u_vol = vol_values[u]
                    v_vol = vol_values[v]

                    if u_vol is not None and v_vol is not None:
                        vol_change = abs(v_vol - u_vol)
                        # Normalize by average volatility to make it
                        # scale-invariant
                        avg_vol = max(0.5 * (abs(u_vol) + abs(v_vol)), 1e-10)
                        vol_change = vol_change / avg_vol

                        # Mark as volatility-significant edge if change is
                        # substantial
                        is_vol_edge = vol_change > 0.05  # threshold for significance

                # Edge score combines inverse distance, volatility change, and
                # edge weight
                edge_score = (1.0 / max(edge_distance, 1e-10)) * \
                    (1.0 + vol_change) * edge_weight
                local_edge_scores[(u, v)] = edge_score

                # Track volatility-significant edges
                if is_vol_edge:
                    local_volatility_edges.append((u, v))

            return local_edge_scores, local_volatility_edges, local_adj_updates

        # Process edges in parallel for large graphs
        if len(edges) > 1000:
            # Split edges into batches
            # Adjust batch size based on number of edges
            batch_size = min(1000, len(edges) // 4 + 1)
            edge_batches = [edges[i:i + batch_size]
                            for i in range(0, len(edges), batch_size)]

            with ThreadPoolExecutor(max_workers=min(8, len(edge_batches))) as executor:
                results = list(executor.map(process_edge_batch, edge_batches))

            # Combine results
            for batch_scores, batch_vol_edges, batch_adj_updates in results:
                edge_scores.update(batch_scores)
                volatility_edges.extend(batch_vol_edges)
                for u_idx, v_idx in batch_adj_updates:
                    adj_matrix[u_idx, v_idx] = 1
        else:
            # For smaller graphs, process sequentially
            batch_scores, batch_vol_edges, batch_adj_updates = process_edge_batch(
                edges)
            edge_scores.update(batch_scores)
            volatility_edges.extend(batch_vol_edges)
            for u_idx, v_idx in batch_adj_updates:
                adj_matrix[u_idx, v_idx] = 1

        # Add 1-simplices (edges) - this is just the list of valid edges
        path_complex[1] = [(u, v) for u, v in edges if (u, v) in edge_scores]
        stats['volatility_paths'] = len(volatility_edges)

        # Create sparse adjacency matrix for efficient path discovery
        sparse_adj = sparse.csr_matrix(adj_matrix)

        # Add higher-order paths up to max_path_length
        # Use efficient path generation with volatility-aware scoring
        for k in range(2, max_path_length + 1):
            print(f"Building dimension {k} paths...")
            dim_start_time = time.time()

            # Dynamic path limit based on dimension
            base_limit = 10000
            dim_factor = 3.0 if k == 2 else 1.0
            max_path_count = min(
                int(base_limit * dim_factor), 88000 // max_path_length)
            # Max number of volatility paths to collect
            max_volatility_paths = max_path_count // 2

            # Use arrays for faster path storage and processing
            valid_paths = []
            volatility_paths = []
            path_scores = {}
            total_candidates = 0

            # Early stopping flags
            is_early_stopping = False
            max_candidates = 1000000  # Process at most this many candidate paths

            # Process paths in parallel for larger dimensions
            def process_path_batch(batch_paths):
                local_valid_paths = []
                local_volatility_paths = []
                local_path_scores = {}
                local_total_candidates = 0

                for path in batch_paths:
                    # Skip processing if we've reached enough volatility paths
                    # and candidates
                    if len(
                            local_volatility_paths) >= max_volatility_paths and local_total_candidates > 100000:
                        break

                    last = path[-1]
                    last_idx = node_to_idx.get(last)
                    if last_idx is None:
                        continue

                    # Get potential successors efficiently using adjacency matrix
                    # Use cached function for smaller graphs
                    successors = get_successors(last)

                    # No successors means we can skip this path
                    if not successors:
                        continue

                    # Process paths
                    for neighbor in successors:
                        # Skip if neighbor is already in path (avoid cycles)
                        if neighbor in path:
                            continue

                        # Early stopping - limit total candidates
                        local_total_candidates += 1
                        if local_total_candidates > max_candidates // 4:  # Distribute the max across workers
                            break

                        # Create new path by extending current path
                        new_path = path + (neighbor,)

                        # Score path based on edges it contains
                        path_score = 0.0
                        edge_count = 0
                        is_volatility_path = False

                        # Calculate path score from edge scores - use cached
                        # lookups
                        edge_score_sum = 0.0

                        for i in range(len(new_path) - 1):
                            edge = (new_path[i], new_path[i + 1])
                            if edge in edge_scores:
                                edge_score_sum += edge_scores[edge]
                                edge_count += 1

                                # Check if this is a volatility-significant
                                # edge
                                if edge in volatility_edges:
                                    is_volatility_path = True
                                    # Early stopping if we found enough
                                    # volatility paths
                                    if len(
                                            local_volatility_paths) >= max_volatility_paths:
                                        break

                        # Normalize by path length
                        if edge_count > 0:
                            path_score = edge_score_sum / edge_count

                        # Store the path and its score
                        local_valid_paths.append(new_path)
                        local_path_scores[new_path] = path_score

                        # Track volatility-significant paths
                        if is_volatility_path:
                            local_volatility_paths.append(new_path)

                return local_valid_paths, local_volatility_paths, local_path_scores, local_total_candidates

            # Check if we need parallelization - more effective for larger path
            # counts
            if len(path_complex[k - 1]) > 100:
                # Split paths into batches for parallel processing
                batch_size = min(100, len(path_complex[k - 1]) // 4 + 1)
                path_batches = [path_complex[k - 1][i:i + batch_size]
                                for i in range(0, len(path_complex[k - 1]), batch_size)]

                with ThreadPoolExecutor(max_workers=min(8, len(path_batches))) as executor:
                    batch_results = list(
                        executor.map(
                            process_path_batch,
                            path_batches))

                # Combine results from all batches
                for batch_valid, batch_vol, batch_scores, batch_total in batch_results:
                    valid_paths.extend(batch_valid)
                    volatility_paths.extend(batch_vol)
                    path_scores.update(batch_scores)
                    total_candidates += batch_total

                    # Check for early stopping based on combined results
                    if len(
                            volatility_paths) >= max_path_count // 2 and total_candidates > max_candidates:
                        is_early_stopping = True
                        break
            else:
                # For smaller path counts, process sequentially
                valid_paths, volatility_paths, path_scores, total_candidates = process_path_batch(
                    path_complex[k - 1])

            stats['total_candidates'] = total_candidates

            # If early stopping occurred, report it
            if is_early_stopping:
                print(
                    f"Early stopping applied after processing {total_candidates} candidate paths")
                print(
                    f"  - Found {len(volatility_paths)} volatility paths, which exceeds the target {max_volatility_paths}")

            # If we have no valid paths at this point, we may need to create some
            # simple paths to ensure dimension k is not empty
            if not valid_paths and k == 2:
                print(
                    "No valid dimension 2 paths found through regular generation, creating some basic paths...")
                # Create some basic dimension 2 paths by extending edges
                edge_paths = path_complex[1][:min(1000, len(path_complex[1]))]

                # Process basic path generation in parallel for larger edge
                # sets
                def process_basic_edge_batch(edge_batch):
                    local_valid_paths = []
                    local_volatility_paths = []
                    local_path_scores = {}

                    for edge_path in edge_batch:
                        # Get the destination node
                        dest = edge_path[1]
                        # Find any successors
                        for succ in get_successors(dest):
                            if succ != edge_path[0]:  # Avoid cycles
                                new_path = edge_path + (succ,)
                                local_valid_paths.append(new_path)
                                # Use a default score
                                local_path_scores[new_path] = 1.0

                                # Check if this is a volatility path
                                is_vol_path = False
                                for i in range(len(new_path) - 1):
                                    edge = (new_path[i], new_path[i + 1])
                                    if edge in volatility_edges:
                                        is_vol_path = True
                                        break

                                if is_vol_path:
                                    local_volatility_paths.append(new_path)

                    return local_valid_paths, local_volatility_paths, local_path_scores

                # Process in parallel if we have enough edges
                if len(edge_paths) > 100:
                    batch_size = min(100, len(edge_paths) // 4 + 1)
                    edge_batches = [edge_paths[i:i + batch_size]
                                    for i in range(0, len(edge_paths), batch_size)]

                    with ThreadPoolExecutor(max_workers=min(8, len(edge_batches))) as executor:
                        basic_results = list(
                            executor.map(
                                process_basic_edge_batch,
                                edge_batches))

                    # Combine results
                    for batch_valid, batch_vol, batch_scores in basic_results:
                        valid_paths.extend(batch_valid)
                        volatility_paths.extend(batch_vol)
                        path_scores.update(batch_scores)

                        # Limit to a reasonable number
                        if len(valid_paths) >= 10000:
                            break
                else:
                    # For smaller edge sets, process sequentially
                    batch_valid, batch_vol, batch_scores = process_basic_edge_batch(
                        edge_paths)
                    valid_paths.extend(batch_valid)
                    volatility_paths.extend(batch_vol)
                    path_scores.update(batch_scores)

                print(
                    f"Created {
                        len(valid_paths)} basic dimension 2 paths as fallback")

            # Final filtering and prioritization with vectorized operations
            selected_paths = []

            # First prioritize volatility-significant paths
            if volatility_paths:
                # Use vectorized operations for sorting
                if len(volatility_paths) > 1:
                    # Use NumPy for faster sorting
                    vol_path_scores = np.array(
                        [path_scores.get(p, 0) for p in volatility_paths])
                    # Sort in descending order
                    vol_path_indices = np.argsort(-vol_path_scores)

                    # Take up to half of max_path_count volatility paths
                    vol_path_limit = min(
                        len(volatility_paths), max_path_count // 2)
                    for i in range(vol_path_limit):
                        selected_paths.append(
                            volatility_paths[vol_path_indices[i]])
                    stats['volatility_paths'] += vol_path_limit
                else:
                    selected_paths.extend(volatility_paths)
                    stats['volatility_paths'] += len(volatility_paths)

            # Fill remaining slots with highest-scoring regular paths
            remaining_slots = max_path_count - len(selected_paths)

            if remaining_slots > 0 and valid_paths:
                # Find paths that aren't already selected
                remaining_paths = [
                    p for p in valid_paths if p not in selected_paths]

                if remaining_paths:
                    # Use vectorized operations for sorting with NumPy
                    remaining_path_scores = np.array(
                        [path_scores.get(p, 0) for p in remaining_paths])
                    # Sort in descending order
                    remaining_path_indices = np.argsort(-remaining_path_scores)

                    # Add the top remaining paths
                    for i in range(min(remaining_slots, len(remaining_paths))):
                        selected_paths.append(
                            remaining_paths[remaining_path_indices[i]])

            # For dimension 2, ensure we have at least some paths
            if k == 2 and not selected_paths and valid_paths:
                # Take at least 1000 paths or all available if less
                num_paths = min(1000, len(valid_paths))
                selected_paths = valid_paths[:num_paths]
                print(
                    f"Ensuring dimension 2 has at least {
                        len(selected_paths)} paths")

            # Store selected paths in path complex
            path_complex[k] = selected_paths
            stats['filtered_paths'] += total_candidates - len(selected_paths)

            print(
                f"Created {
                    len(selected_paths)} paths in dimension {k}({
                        time.time() -
                        dim_start_time:.2f}s)")

            # Additional reporting
            if len(selected_paths) > 0:
                print(
                    f"  - Volatility-significant paths: {sum(1 for p in selected_paths if p in volatility_paths)}")
                print(
                    f"  - Path filtering rate: {stats['filtered_paths'] / max(1, total_candidates):.1%}")
                if stats['homology_pruned'] > 0:
                    print(
                        f"  - Paths pruned by homology check: {stats['homology_pruned']}")
                if is_early_stopping:
                    print(
                        f"  - Early stopping applied: Yes, stopped after {total_candidates} candidates")

        # Final statistics
        print(f"Path complex sizes: " +
              ", ".join([f"dim {d}: {len(paths)}" for d, paths in path_complex.items()]))
        print(
            f"Total path complex creation time: {
                time.time() -
                start_time:.2f}s")

        return path_complex

    def _compute_homology_from_boundary(self, boundary_matrix):
        """
        Compute homology from boundary matrix using efficient reduction.
        Uses sparse matrices for large boundary matrices to improve performance.
        Implements simplex tree approach for more efficient homology computation.

        Parameters:
        -----------
        boundary_matrix : numpy.ndarray
            Binary boundary matrix encoding boundary relationships

        Returns:
        --------
        int
            Betti number (dimension of homology group)
        """
        import numpy as np
        import scipy.sparse as sp
        from scipy.linalg import svd

        # Convert to sparse matrix if large enough
        n_rows, n_cols = boundary_matrix.shape
        is_sparse = n_rows * n_cols > 10000

        # Skip computation if the matrix is empty or trivial
        if n_rows == 0 or n_cols == 0:
            return 0

        # Check for degenerate matrices where SVD would fail
        if n_rows == 0 or n_cols == 0 or min(n_rows, n_cols) <= 0:
            return 0

        try:
            if is_sparse:
                # Convert to sparse format for large matrices
                sparse_matrix = sp.csr_matrix(boundary_matrix)

                # Check for empty matrix
                if sparse_matrix.nnz == 0:
                    return n_cols  # All columns are in the kernel

                # Use more robust rank estimation for sparse matrices
                if min(n_rows, n_cols) > 100:
                    # For very large matrices, use iterative methods
                    # with adaptive k parameter
                    k = min(min(n_rows, n_cols) - 1, 100)
                    if k <= 0:  # Safety check
                        return 0

                    try:
                        # Use ARPACK for large sparse matrices
                        singular_values = sp.linalg.svds(
                            sparse_matrix, k=k, return_singular_vectors=False)
                        # Determine numerical rank with adaptive threshold
                        tol = 1e-5 * max(singular_values)
                        rank = sum(s > tol for s in singular_values)
                    except Exception as e:
                        # Fall back to QR decomposition for numerical stability
                        print(f"SVD failed, using QR decomposition: {str(e)}")
                        if n_rows >= n_cols:
                            # Use QR decomposition for tall matrices
                            Q, R = sp.linalg.qr(sparse_matrix, mode='economic')
                            # Count non-zero diagonal elements in R
                            diag_elements = np.abs(R.diagonal())
                            tol = 1e-8 * \
                                max(diag_elements) if diag_elements.size > 0 else 1e-8
                            rank = np.sum(diag_elements > tol)
                        else:
                            # For fat matrices, transpose
                            Q, R = sp.linalg.qr(
                                sparse_matrix.T, mode='economic')
                            diag_elements = np.abs(R.diagonal())
                            tol = 1e-8 * \
                                max(diag_elements) if diag_elements.size > 0 else 1e-8
                            rank = np.sum(diag_elements > tol)
                else:
                    # For smaller sparse matrices, convert to dense
                    try:
                        _, s, _ = svd(
                            sparse_matrix.toarray(), full_matrices=False)
                        # Use adaptive tolerance for rank determination
                        tol = 1e-8 * max(s) if s.size > 0 else 1e-8
                        rank = np.sum(s > tol)
                    except Exception as e:
                        # Fall back to numpy's matrix rank
                        print(f"SVD failed, using numpy: {str(e)}")
                        rank = np.linalg.matrix_rank(
                            sparse_matrix.toarray(), tol=1e-8)
            else:
                # For smaller matrices, use regular SVD
                try:
                    # Use full SVD for accuracy
                    _, s, _ = svd(boundary_matrix, full_matrices=False)
                    # Adaptive tolerance based on matrix size and largest singular
                    # value
                    tol = 1e-8 * max(s) if s.size > 0 else 1e-8
                    rank = np.sum(s > tol)
                except Exception as e:
                    # Fall back to numpy's matrix rank with increased tolerance
                    print(f"SVD failed, using numpy: {str(e)}")
                    rank = np.linalg.matrix_rank(boundary_matrix, tol=1e-7)

            # Calculate Betti number
            # Betti number = nullity of boundary matrix = dim C - rank
            betti = n_cols - rank

            return betti

        except Exception as e:
            print(f"Error computing homology: {str(e)}")
            # Comprehensive fallback approach
            try:
                # Try simplex tree approach if available
                try:
                    import gudhi as gd
                    # Create a simplex tree from the boundary matrix
                    st = gd.SimplexTree()

                    # Extract simplices from the boundary matrix
                    # For each column in boundary matrix
                    simplices = []
                    weights = []

                    # Add simplices to the simplex tree
                    for i in range(n_cols):
                        # Find the faces of this simplex
                        faces = np.where(boundary_matrix[:, i] > 0)[0]
                        if len(faces) > 0:
                            simplex = tuple(int(f) for f in faces)
                            simplices.append(simplex)
                            weights.append(1.0)  # Unit weight for simplices

                    # Insert simplices into the tree
                    for i, simplex in enumerate(simplices):
                        st.insert(simplex, filtration=weights[i])

                    # Compute persistent homology
                    st.compute_persistence()

                    # Extract Betti numbers
                    betti_numbers = st.betti_numbers()

                    # Return appropriate Betti number (dimension 1 for most TDA
                    # cases)
                    if len(betti_numbers) >= 2:
                        return betti_numbers[1]  # Return Betti-1
                    elif len(betti_numbers) >= 1:
                        return betti_numbers[0]  # Return Betti-0
                    else:
                        return 0
                except ImportError:
                    # If gudhi not available, use numpy's matrix rank
                    rank = np.linalg.matrix_rank(
                        boundary_matrix.toarray() if is_sparse else boundary_matrix, tol=1e-6)
                return n_cols - rank
            except Exception as fallback_error:
                print(
                    f"Fallback computation failed: {
                        str(fallback_error)}. Returning 0.")
                return 0

    def identify_regimes(self, n_regimes=3, use_topological_features=True):
        """
        Identify volatility regimes using topological features and enhanced clustering.
        """
        print(f"Identifying {n_regimes} volatility regimes...")
        start_time = time.time()

        # Ensure we have a distance matrix
        if self.temporal_distance_matrix is None:
            self.compute_temporally_weighted_distance()

        # Base feature set with NaN handling
        scaler = RobustScaler(unit_variance=True)
        base_features = np.nan_to_num(self.features, nan=0.0)
        base_features = scaler.fit_transform(base_features)

        if use_topological_features:
            if not hasattr(self, 'topological_features'):
                print("\n=== Extracting Topological Features ===")
                topo_features, topo_feature_names = self.extract_topological_features()
                self.topological_features = topo_features
                self.topological_feature_names = topo_feature_names
                print(f"Extracted {len(topo_features)} topological features:")
                for i, name in enumerate(topo_feature_names):
                    print(f"  {i + 1}. {name}: {topo_features[i]:.4f}")
                print("======================================\n")

            # Create topological feature matrix per point
            n_samples = self.n_samples
            topo_feature_matrix = np.zeros(
                (n_samples, len(self.topological_features)))

            # For sequential data, use window-based assignment of topological
            # features
            if hasattr(
                    self,
                    'path_zigzag_diagrams') and self.path_zigzag_diagrams is not None:
                window_indices = self.path_zigzag_diagrams['window_indices']

                # Assign features to each point based on windows it belongs to
                point_window_count = np.zeros(n_samples)

                for i, window_idx in enumerate(window_indices):
                    for idx in window_idx:
                        topo_feature_matrix[idx] += np.nan_to_num(
                            self.topological_features, nan=0.0)
                        point_window_count[idx] += 1

                # Average features for points in multiple windows
                for i in range(n_samples):
                    if point_window_count[i] > 0:
                        topo_feature_matrix[i] /= point_window_count[i]
            else:
                # If no zigzag persistence, assign global topological features
                # to all points
                topo_feature_matrix = np.tile(np.nan_to_num(
                    self.topological_features, nan=0.0), (n_samples, 1))

            # Normalize topological features
            topo_feature_matrix = np.nan_to_num(topo_feature_matrix, nan=0.0)
            topo_feature_matrix = scaler.fit_transform(topo_feature_matrix)

            # Combine base features with topological features
            combined_features = np.column_stack(
                [base_features, topo_feature_matrix])
            print(f"Combined feature matrix shape: {combined_features.shape}")
        else:
            combined_features = base_features

        # Use specialized time series regime clustering
        regime_labels = self._cluster_time_series_regimes(
            n_regimes=n_regimes,
            features=combined_features,
            temporal_weight=0.3
        )

        # Store regime labels
        self.regime_labels = regime_labels

        # Print final distribution
        final_counts = np.bincount(regime_labels)
        print(f"Final regime distribution: {final_counts}")
        print(
            f"Regime balance ratio: {
                np.min(final_counts) /
                np.max(final_counts):.3f}")
        print(
            f"Regime identification completed in {
                time.time() -
                start_time:.2f} seconds")

        return regime_labels

    def _compute_zigzag_features(self, path_zigzag_diagrams, transitions):
        """
        Compute features from zigzag persistence diagrams for regime identification.
        Extracts statistical summaries of topological features across dimensions.

        Parameters:
        -----------
        path_zigzag_diagrams : dict
            Dictionary containing zigzag persistence diagrams by dimension
        transitions : list
            List of transitions between windows

        Returns:
        --------
        dict
            Dictionary of features extracted from zigzag persistence
        """
        import numpy as np

        # Initialize features dictionary
        features = {}

        # Extract features from Betti numbers by dimension
        for dim in range(3):  # 0, 1, 2 dimensions
            if dim in path_zigzag_diagrams and path_zigzag_diagrams[dim]:
                betti_series = path_zigzag_diagrams[dim]

                # Basic statistics of Betti numbers
                features[f'zigzag_dim{dim}_betti_mean'] = np.mean(betti_series)
                features[f'zigzag_dim{dim}_betti_max'] = np.max(betti_series)
                features[f'zigzag_dim{dim}_betti_min'] = np.min(betti_series)
                features[f'zigzag_dim{dim}_betti_std'] = np.std(betti_series)
                features[f'zigzag_dim{dim}_betti_range'] = np.max(
                    betti_series) - np.min(betti_series)

                # Stability features - change between consecutive windows
                if len(betti_series) > 1:
                    changes = np.diff(betti_series)
                    features[f'zigzag_dim{dim}_betti_changes_mean'] = np.mean(
                        np.abs(changes))
                    features[f'zigzag_dim{dim}_betti_changes_max'] = np.max(
                        np.abs(changes))
                    features[f'zigzag_dim{dim}_betti_stability'] = 1.0 - \
                        (np.count_nonzero(changes) / len(changes))

        # Extract features from transitions between windows
        dim_transitions = {}
        for t in transitions:
            dim = t['dimension']
            if dim not in dim_transitions:
                dim_transitions[dim] = []
            dim_transitions[dim].append(t['transition_strength'])

        # Compute transition statistics by dimension
        for dim in dim_transitions:
            if dim_transitions[dim]:
                trans_values = dim_transitions[dim]
                features[f'zigzag_dim{dim}_transition_mean'] = np.mean(
                    trans_values)
                features[f'zigzag_dim{dim}_transition_max'] = np.max(
                    trans_values)
                features[f'zigzag_dim{dim}_transition_total'] = np.sum(
                    trans_values)
                features[f'zigzag_dim{dim}_transition_std'] = np.std(
                    trans_values)

        # Add network statistics as additional features
        if hasattr(self, 'window_indices') and hasattr(self, 'path_complexes'):
            # Network size features
            n_nodes = [len(path_complex[0])
                       for path_complex in self.path_complexes]
            n_edges = [len(
                path_complex[1]) if 1 in path_complex else 0 for path_complex in self.path_complexes]

            features['network_nodes_mean'] = np.mean(n_nodes)
            features['network_edges_mean'] = np.mean(n_edges)

            # Network density
            densities = []
            for i, complex_size in enumerate(n_nodes):
                if complex_size > 1 and n_edges[i] > 0:
                    max_possible_edges = complex_size * (complex_size - 1)
                    densities.append(n_edges[i] / max_possible_edges)
                else:
                    densities.append(0)

            features['network_density_mean'] = np.mean(densities)

            # Average degree
            degrees = [2 * edges / max(1, nodes)
                       for nodes, edges in zip(n_nodes, n_edges)]
            features['network_degree_mean'] = np.mean(degrees)
            features['network_degree_max'] = np.max(degrees)

        print("\n=== Extracting Topological Features ===")
        print("Extracting topological features...")

        # Count feature types
        feature_counts = {
            'persistence_diagram': 0,
            'zigzag_persistence': 0,
            'network_statistics': 0
        }

        # Categorize features
        for name in features:
            if name.startswith('zigzag_'):
                feature_counts['zigzag_persistence'] += 1
            elif name.startswith('network_'):
                feature_counts['network_statistics'] += 1
            else:
                feature_counts['persistence_diagram'] += 1

        # Report feature extraction
        print(
            f"Extracted {
                len(features)} topological features in 0.00 seconds")
        print(
            f" - {feature_counts['persistence_diagram']} persistence diagram features")
        print(
            f" - {feature_counts['zigzag_persistence']} zigzag persistence features")
        print(
            f" - {feature_counts['network_statistics']} network statistics features")

        print("\nExtracted features:")
        for i, (name, value) in enumerate(features.items(), 1):
            print(f"  {i}. {name}: {value:.4f}")

        print("======================================\n")

        return features

    def optimize_epsilon_threshold(
            self,
            target_index=None,
            n_trials=10,
            min_percentile=5,
            max_percentile=90):
        """
        Optimize epsilon threshold for network construction based on mutual information gain.
        Uses a target feature (preferably volatility) to optimize against.

        Parameters:
        -----------
        target_index : int or None
            Index of target feature to optimize against, preferably volatility-related
        n_trials : int
            Number of epsilon values to try
        min_percentile : float
            Minimum percentile of distances to consider
        max_percentile : float
            Maximum percentile of distances to consider

        Returns:
        --------
        tuple
            (optimal_epsilon, min_epsilon, max_epsilon, info_gain_scores)
        """
        print("Optimizing epsilon threshold for network construction...")

        # Ensure distance matrix exists
        if self.temporal_distance_matrix is None:
            raise ValueError(
                "Distance matrix must be computed before optimizing epsilon")

        # If target index is not provided, try to find a volatility-related
        # feature
        if target_index is None:
            # Search for volatility-related features in feature names
            vol_keywords = ['volatil', 'vol_adjusted', 'vol_', 'Volatility']
            for keyword in vol_keywords:
                for i, name in enumerate(self.feature_names):
                    if keyword.lower() in name.lower():
                        target_index = i
                        print(
                            f"Using {name} as target for epsilon optimization")
                        break
                if target_index is not None:
                    break

            # If still not found, use a reasonable default
            if target_index is None:
                # Try to find any reasonable target from typical market
                # features
                for keyword in ['price', 'value', 'return',
                                'trade_volume', 'trade_frequency']:
                    for i, name in enumerate(self.feature_names):
                        if keyword.lower() in name.lower():
                            target_index = i
                            print(
                                f"Using {name} as target for epsilon optimization (no volatility feature found)")
                            break
                    if target_index is not None:
                        break

                # Last resort - use first feature
                if target_index is None:
                    target_index = 0
                    print(
                        f"Using {
                            self.feature_names[0]} as target for epsilon optimization(fallback)")

        # Get the target feature values
        target_feature = self.features[:, target_index]

        # Flatten the distance matrix and get distance distribution
        flat_dists = self.temporal_distance_matrix.flatten()
        # Exclude zeros (self-distances) and negative values
        flat_dists = flat_dists[(flat_dists > 0) & np.isfinite(flat_dists)]

        # Check if we have enough non-zero distances to compute percentiles
        if len(flat_dists) < 10:
            print(
                f"WARNING: Very few non-zero distances found ({len(flat_dists)})")
            print(f"Using default epsilon range instead of percentiles")
            # Use default range based on distance statistics
            if len(flat_dists) > 0:
                min_epsilon = np.min(flat_dists) * 0.8
                max_epsilon = np.max(flat_dists) * 1.2
            else:
                # Extreme fallback case - use arbitrary small values
                min_epsilon = 0.001
                max_epsilon = 0.1
            print(
                f"Fallback epsilon range: [{
                    min_epsilon:.4f}, {
                    max_epsilon:.4f}]")
        else:
            # Check if distances have very low variability
            dist_range = np.max(flat_dists) - np.min(flat_dists)
            if dist_range < 1e-5:
                print(
                    f"WARNING: Very low distance variability detected({
                        dist_range:.8f})")
                # Create artificial range around the mean
                mean_dist = np.mean(flat_dists)
                min_epsilon = max(0.001, mean_dist * 0.5)
                max_epsilon = mean_dist * 2.0
                print(
                    f"Using artificial epsilon range: [{
                        min_epsilon:.4f}, {
                        max_epsilon:.4f}]")
            else:
                # Normal case - compute percentiles
                try:
                    min_epsilon = np.percentile(flat_dists, min_percentile)
                    max_epsilon = np.percentile(flat_dists, max_percentile)
                except Exception as e:
                    print(f"Error computing percentiles: {str(e)}")
                    # Fallback to simple min/max
                    min_epsilon = np.min(flat_dists)
                    max_epsilon = np.max(flat_dists)

                # Ensure min_epsilon < max_epsilon with a minimum range
                if min_epsilon >= max_epsilon or (
                        max_epsilon - min_epsilon) < 1e-5:
                    print(f"WARNING: Epsilon range too small or invalid")
                    center = (min_epsilon + max_epsilon) / 2
                    min_epsilon = max(0.001, center * 0.5)
                    max_epsilon = center * 2.0
                    print(
                        f"Using expanded epsilon range: [{
                            min_epsilon:.4f}, {
                            max_epsilon:.4f}]")
                else:
                    print(
                        f"Searching epsilon range: [{
                            min_epsilon:.4f}, {
                            max_epsilon:.4f}]")

        # Try different epsilon values and measure information gain
        epsilon_values = np.linspace(min_epsilon, max_epsilon, n_trials)
        info_gain_scores = []

        for epsilon in epsilon_values:
            # Create adjacency matrix for this epsilon
            adjacency = self.temporal_distance_matrix <= epsilon

            # Remove self-loops
            np.fill_diagonal(adjacency, 0)

            # Check if the adjacency matrix is non-trivial (not all zeros or
            # all ones)
            if np.all(adjacency == 0) or np.all(
                    adjacency[~np.eye(len(adjacency), dtype=bool)]):
                # If trivial, assign a low score
                info_gain = 0.0
                print(
                    f" - Epsilon: {
                        epsilon:.4f}, Information gain: {
                        info_gain:.4f}(trivial adjacency)"
                )
            else:
                # Compute mutual information gain
                info_gain = self._estimate_mutual_information_gain(
                    target_feature, adjacency)
            print(
                f"  - Epsilon: {epsilon:.4f}, Information gain: {info_gain:.4f}")

            info_gain_scores.append(info_gain)

        # Find optimal epsilon (maximizing information gain)
        if len(info_gain_scores) == 0 or max(info_gain_scores) == 0:
            # If all scores are zero, use the smallest epsilon that gives a
            # non-trivial network
            for i, epsilon in enumerate(epsilon_values):
                adjacency = self.temporal_distance_matrix <= epsilon
                np.fill_diagonal(adjacency, 0)
                if np.sum(adjacency) > 0 and np.sum(
                        adjacency) < len(adjacency)**2 - len(adjacency):
                    optimal_idx = i
                    break
            else:
                # Default to middle epsilon
                optimal_idx = len(epsilon_values) // 2
        else:
            optimal_idx = np.argmax(info_gain_scores)

        optimal_epsilon = epsilon_values[optimal_idx]

        # Fix formatting for optimal epsilon printing
        print(f"Optimal epsilon: {optimal_epsilon:.4f} (information gain: {info_gain_scores[optimal_idx]:.4f})")
        
        # Return optimal value, range, and scores
        return optimal_epsilon, min_epsilon, max_epsilon, info_gain_scores

    def _estimate_mutual_information_gain(self, target, adjacency, bins=20):
        """
        Estimate mutual information gain between target feature and network structure.

        Parameters:
        -----------
        target : numpy.ndarray
            Target feature values
        adjacency : numpy.ndarray
            Adjacency matrix
        bins : int
            Number of bins for discretization

        Returns:
        --------
        float
            Mutual information gain
        """
        # Define a helper function to estimate entropy
        def estimate_entropy(data):
            # Discretize continuous data
            hist, _ = np.histogram(data, bins=bins, density=True)
            # Add small constant to avoid log(0)
            hist = hist + 1e-10
            # Normalize
            hist = hist / hist.sum()
            # Compute entropy
            entropy = -np.sum(hist * np.log2(hist))
            return entropy

        # Compute node degree as network feature
        node_degree = np.sum(adjacency, axis=1)

        # Compute individual entropies
        h_target = estimate_entropy(target)
        h_degree = estimate_entropy(node_degree)

        # Compute joint entropy
        joint_data = np.column_stack([target, node_degree])
        # Use 2D histogram for joint distribution
        hist, _ = np.histogramdd(joint_data, bins=bins)
        # Add small constant and normalize
        hist = hist.flatten() + 1e-10
        hist = hist / hist.sum()
        # Compute joint entropy
        h_joint = -np.sum(hist * np.log2(hist))

        # Compute mutual information: MI = H(X) + H(Y) - H(X,Y)
        mutual_info = h_target + h_degree - h_joint

        # Normalize by target entropy to get information gain ratio
        normalized_gain = mutual_info / max(h_target, 1e-10)

        return normalized_gain

    def compute_window_persistent_homology(
            self, window_data, window_idx, epsilon, max_dimension=2):
        """
        Compute persistent homology for a specific window of data.
        Used by the optimize_epsilon_threshold method to evaluate different epsilon values.

        Parameters:
        -----------
        window_data : numpy.ndarray
            Feature data for this window
        window_idx : int
            Window index for logging
        epsilon : float
            Distance threshold for network construction
        max_dimension : int
            Maximum homology dimension to compute

        Returns:
        --------
        dict
            Dictionary with homology results
        """
        import networkx as nx

        # Create distance matrix for this window
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(window_data)

        # Create network
        G = nx.DiGraph()

        # Add nodes
        for i in range(len(window_data)):
            G.add_node(i, index=i)

        # Add edges below epsilon threshold
        for i in range(len(window_data)):
            for j in range(len(window_data)):
                if i != j and distances[i, j] <= epsilon:
                    G.add_edge(i, j, weight=1.0 / max(distances[i, j], 1e-10))

        # Compute path complex
        path_complex = self.compute_path_complex(G, max_dimension)

        # Compute homology for each dimension
        homology_results = {}

        for dim in range(max_dimension + 1):
            if dim == 0:
                # For dimension 0, Betti number is the number of connected
                # components
                betti = nx.number_connected_components(G.to_undirected())
                homology_results[dim] = betti
            elif dim in path_complex and dim - 1 in path_complex:
                paths = path_complex[dim]
                boundaries = path_complex[dim - 1]

                if not paths or not boundaries:
                    homology_results[dim] = 0
                    continue

                # Create boundary matrix
                boundary_matrix = np.zeros((len(paths), len(boundaries)))

                for i, path in enumerate(paths):
                    for j, boundary in enumerate(boundaries):
                        # Check if boundary is a face of path
                        if len(boundary) == dim and all(
                                node in path for node in boundary):
                            boundary_matrix[i, j] = 1

                # Compute homology
                betti = self._compute_homology_from_boundary(boundary_matrix)
                homology_results[dim] = betti
            else:
                homology_results[dim] = 0

        return homology_results

    def compute_persistent_path_zigzag_homology(
            self,
            window_size=100,
            overlap=50,
            max_path_length=2,
            min_epsilon=0.1,
            max_epsilon=2.0,
            num_steps=10,
            output_dir=None,
            target_feature_name=None,
            information_gain_threshold=0.01):
        """
        Compute path zigzag persistence homology for market microstructure data.
        This approach is more suitable for directed time series data than
        traditional persistent homology.

        Parameters:
        -----------
        window_size : int
            Size of sliding window
        overlap : int
            Overlap between consecutive windows
        max_path_length : int
            Maximum length of paths to consider
        min_epsilon : float
            Minimum distance threshold for graph construction
        max_epsilon : float
            Maximum distance threshold for graph construction
        num_steps : int
            Number of steps in epsilon range
        output_dir : str or None
            Directory to save visualization output
        target_feature_name : str or None
            Name of target feature for epsilon optimization
        information_gain_threshold : float
            Threshold for information gain to consider a window informative

        Returns:
        --------
        dict
            Path zigzag persistence diagrams
        """
        import networkx as nx
        print("Computing persistent path zigzag homology for market microstructure...")
        start_time = time.time()

        n_samples = len(self.features)

        # Create sliding windows
        window_indices = []
        for start in range(0, n_samples - window_size + 1,
                           window_size - overlap):
            end = min(start + window_size, n_samples)
            window_indices.append(list(range(start, end)))

        # For each window, construct a directed network and compute path
        # complex
        networks = []
        path_complexes = []
        path_zigzag_diagrams = {k: [] for k in range(max_path_length + 1)}

        # Find volatility feature index for adaptive epsilon and path filtering
        vol_feature_idx = None
        vol_feature_name = None

        # First check for volatility in feature names
        for i, name in enumerate(self.feature_names):
            if 'volatil' in name.lower():
                vol_feature_idx = i
                vol_feature_name = name
                print(f"Found volatility feature: {name} (index {i})")
                break

        # If not found, check for related features
        if vol_feature_idx is None:
            for keyword in ['vol_adjusted', 'vol', 'Volatility']:
                for i, name in enumerate(self.feature_names):
                    if keyword.lower() in name.lower():
                        vol_feature_idx = i
                        vol_feature_name = name
                        print(
                            f"Using {name} as volatility feature for filtering (index {i})")
                        break
                if vol_feature_idx is not None:
                    break

        # Prepare computation stats for reporting
        computation_stats = {
            'window_count': len(window_indices),
            'total_paths_generated': 0,
            'volatility_paths': 0,
            'filtered_paths': 0,
            'homology_computation_time': 0.0,
            'selected_windows': [],
            'window_properties': []
        }

        # Analyze each window's volatility properties for adaptive filtering
        window_properties = []

        if vol_feature_idx is not None:
            for i, indices in enumerate(window_indices):
                window_data = self.features[indices]
                vol_values = window_data[:, vol_feature_idx]

                # Calculate volatility statistics for this window
                window_properties.append({
                    'idx': i,
                    'target_mean': np.mean(vol_values),
                    'target_std': np.std(vol_values),
                    'target_max': np.max(vol_values),
                    'target_min': np.min(vol_values),
                    'target_range': np.max(vol_values) - np.min(vol_values),
                    'information_gain': 0.0  # Will be computed below
                })

        # Construct networks for all windows
        print("Constructing directed networks for each window...")
        for i, indices in enumerate(window_indices):
            # For each window, create a subgraph
            window_data = self.features[indices]
            window_times = self.timestamps[indices] if self.timestamps is not None else None

            # Create a distance matrix for this window
            if hasattr(self, 'temporal_distance_matrix'):
                # Use precomputed distances if available
                window_dist = self.temporal_distance_matrix[np.ix_(
                    indices, indices)]
            else:
                # Otherwise compute distances for this window
                from sklearn.metrics import pairwise_distances
                window_dist = pairwise_distances(window_data)

            # Determine adaptive epsilon based on window properties
            window_epsilon = min_epsilon
            if window_properties:
                # Get this window's properties
                window_prop = window_properties[i]

                # Scale epsilon based on volatility - higher volatility = lower
                # epsilon (more edges)
                if vol_feature_idx is not None:
                    vol_values = window_data[:, vol_feature_idx]
                    vol_std = np.std(vol_values)
                    vol_range = np.max(vol_values) - np.min(vol_values)

                    # More volatility should result in a lower epsilon (more
                    # connections)
                    vol_scaling_factor = 1.0
                    if vol_range > 0:
                        # Normalize by the max range across all windows
                        max_range = max(wp['target_range']
                                        for wp in window_properties)
                        rel_range = vol_range / max_range if max_range > 0 else 0.5
                        vol_scaling_factor = 1.0 - 0.5 * rel_range  # Scale between 0.5 and 1.0

                    # Adjust epsilon by volatility scaling factor
                    window_epsilon = min_epsilon + \
                        (max_epsilon - min_epsilon) * vol_scaling_factor
                    print(
                        f"Window {
                            i + 1} / {
                            len(window_indices)}: Using adaptive epsilon range[{
                                window_epsilon:.4f}, {
                                max_epsilon:.4f}]")

            # Construct network with the adjusted epsilon
            G = nx.DiGraph()

            # Add nodes with features for volatility-aware path filtering
            for j, idx in enumerate(indices):
                node_attrs = {'index': idx, 'original_index': indices[j]}

                # Add feature values as node attributes
                for f_idx, f_name in enumerate(self.feature_names):
                    node_attrs[f_name] = window_data[j, f_idx]

                G.add_node(idx, **node_attrs)

            # Add edges below epsilon threshold
            for j in range(len(indices)):
                for k in range(len(indices)):
                    if j != k and window_dist[j, k] < window_epsilon:
                        weight = 1.0 / max(window_dist[j, k], 1e-10)
                        G.add_edge(
                            indices[j], indices[k], weight=weight, distance=window_dist[j, k])

            # Store the network
            networks.append(G)

        # Compute information gain for each window - prioritize windows with
        # high information content
        window_info_gain = []

        if vol_feature_idx is not None:
            from scipy.stats import entropy
            from sklearn.preprocessing import KBinsDiscretizer

            for i, indices in enumerate(window_indices):
                # Get volatility values for this window
                window_data = self.features[indices]
                vol_values = window_data[:, vol_feature_idx]

                # Discretize volatility values
                discretizer = KBinsDiscretizer(
                    n_bins=10, encode='ordinal', strategy='uniform')
                vol_discrete = discretizer.fit_transform(
                    vol_values.reshape(-1, 1)).flatten()

                # Calculate entropy as information content
                # Higher entropy = more varied volatility = more information
                hist, _ = np.histogram(vol_discrete, bins=10)
                # Add small constant to avoid log(0)
                info_gain = entropy(hist + 1e-10)

                # Store information gain
                window_info_gain.append((i, info_gain))
                if i < len(window_properties):
                    window_properties[i]['information_gain'] = info_gain

        # Select windows to process - if we have many windows, prioritize
        # informative ones
        windows_to_process = list(range(len(networks)))

        # If we have a very large number of windows, filter to most informative
        # ones
        if len(networks) > 10 and window_info_gain:
            # Sort windows by information gain
            window_info_gain.sort(key=lambda x: x[1], reverse=True)

            # Take top 7 windows (helps with computational efficiency)
            windows_to_process = [idx for idx, _ in window_info_gain[:7]]
            print(
                f"Selected {
                    len(windows_to_process)} most informative windows out of {
                    len(networks)}")

        # Process the selected windows
        computation_stats['selected_windows'] = windows_to_process

        for window_idx in windows_to_process:
            G = networks[window_idx]
            print(
                f"Computing path complex for window {window_idx + 1}/{len(networks)}...")

            # Compute path complex with volatility-aware filtering
            path_complex = self.compute_path_complex(G, max_path_length)
            path_complexes.append(path_complex)

            # Print path complex sizes
            print(f"Window {window_idx + 1} path complex sizes: " + ", ".join(
                [f"dim {d}: {len(paths)}" for d, paths in path_complex.items()]))

        # Calculate homology for dimension 0
        print("Processing homology in dimension 0...")
        dim0_start = time.time()

        for path_complex in path_complexes:
            # Dimension 0 is simply the number of connected components
            path_zigzag_diagrams[0].append(len(path_complex[0]))

        print(
            f"Dimension 0 processing completed in {
                time.time() -
                dim0_start:.2f} seconds")

        # Calculate homology for dimensions 1 and above
        for dim in range(1, max_path_length + 1):
            print(f"Processing homology in dimension {dim}...")
            dim_start = time.time()
            betti_series = []

            for i, path_complex in enumerate(path_complexes):
                if dim not in path_complex or not path_complex[dim]:
                    betti_series.append(0)
                    continue

                paths = path_complex[dim]

                if dim - 1 not in path_complex or not path_complex[dim - 1]:
                    betti_series.append(0)
                    continue

                # Create boundary matrix
                n_paths = len(paths)
                n_boundaries = len(path_complex[dim - 1])

                # Skip if we have empty boundaries
                if n_paths == 0 or n_boundaries == 0:
                    betti_series.append(0)
                    continue

                # Create boundary matrix - this defines the homology
                # calculation
                boundary_matrix = np.zeros((n_paths, n_boundaries))

                # Define boundary based on the type of complex
                for j, path in enumerate(paths):
                    for k, boundary_path in enumerate(path_complex[dim - 1]):
                        # For directed paths, boundary is contiguous sub-paths
                        if len(boundary_path) == dim:
                            # Check if boundary_path forms a contiguous subpath
                            # This is specific to directed path complexes
                            path_nodes = set(path)
                            boundary_nodes = set(boundary_path)

                            if boundary_nodes.issubset(path_nodes):
                                # Ensure subpath is contiguous
                                is_contiguous = False
                                for start_idx in range(len(path) - dim + 1):
                                    subpath = path[start_idx:start_idx + dim]
                                    if set(subpath) == boundary_nodes:
                                        is_contiguous = True
                                        break

                                if is_contiguous:
                                    boundary_matrix[j, k] = 1

                # Compute homology from boundary matrix
                homology_start = time.time()
                betti = self._compute_homology_from_boundary(boundary_matrix)
                computation_stats['homology_computation_time'] += time.time() - \
                    homology_start

                betti_series.append(betti)

            # Store Betti numbers for this dimension
            path_zigzag_diagrams[dim] = betti_series
            print(
                f"Dimension {dim} processing completed in {
                    time.time() -
                    dim_start:.2f} seconds")

        # Analyze persistence across windows for zigzag
        print("\n=== Persistence Analysis ===")
        persistence_stats = {}

        for dim in range(max_path_length + 1):
            if dim in path_zigzag_diagrams:
                betti_numbers = path_zigzag_diagrams[dim]

                if betti_numbers:
                    persistence_stats[dim] = {
                        'count': len(betti_numbers),
                        'mean': np.mean(betti_numbers),
                        'max': np.max(betti_numbers),
                        'min': np.min(betti_numbers),
                        'median': np.median(betti_numbers),
                        'sequence': betti_numbers
                    }

                    print(f"Dimension {dim}:")
                    print(
                        f" - Persistent features found in {
                            len(betti_numbers)} out of {
                            len(window_indices)} windows")
                    print(
                        f"  - Average Betti number: {np.mean(betti_numbers):.2f}")
                    print(f"  - Maximum Betti number: {np.max(betti_numbers)}")
                    print(f"  - Betti number sequence: {betti_numbers}")
                    print()

        # Compute stability of persistence features
        print("Persistence Stability Analysis:")
        stability_scores = {}
        self.regime_stability = {}

        for dim in range(max_path_length + 1):
            if dim in persistence_stats:
                stats = persistence_stats[dim]

                # Calculate stability as percentage of windows where Betti
                # number > 0
                non_zero_betti = sum(1 for b in stats['sequence'] if b > 0)
                stability = non_zero_betti / \
                    max(1, len(stats['sequence'])) * 100
                stability_scores[dim] = stability

                print(f"Dimension {dim} stability: {stability:.2f}%")

                # For each dimension, calculate stability by window
                for i, betti in enumerate(stats['sequence']):
                    window_idx = computation_stats['selected_windows'][i]
                    # Normalize to [0,1]
                    self.regime_stability[window_idx] = stability / 100.0

        print("===========================\n")

        # Compute zigzag transition strengths
        print("Computing transitions between windows...")
        transitions = []
        window_indices_array = [indices for i, indices in enumerate(
            window_indices) if i in computation_stats['selected_windows']]

        for i in range(len(path_complexes) - 1):
            # Find windows that correspond to these indices
            current_window = computation_stats['selected_windows'][i]
            next_window = computation_stats['selected_windows'][i + 1]

            # Find overlap indices between windows
            if current_window < len(
                    window_indices) and next_window < len(window_indices):
                current_indices = set(window_indices[current_window])
                next_indices = set(window_indices[next_window])
                overlap_indices = current_indices.intersection(next_indices)

                # Skip if no overlap
                if not overlap_indices:
                    continue

                # For each dimension, calculate transition strength
                for dim in range(1, max_path_length + 1):  # Skip dimension 0
                    if dim in path_zigzag_diagrams:
                        # Get Betti numbers
                        betti_current = path_zigzag_diagrams[dim][i]
                        betti_next = path_zigzag_diagrams[dim][i + 1]

                        # Simplified transition strength calculation
                        transition_strength = min(betti_current, betti_next)

                        transitions.append({
                            'window_pair': (current_window, next_window),
                            'dimension': dim,
                            'betti_current': betti_current,
                            'betti_next': betti_next,
                            'transition_strength': transition_strength
                        })

        # Store results
        self.path_zigzag_diagrams = path_zigzag_diagrams
        self.window_indices = window_indices
        self.path_complexes = path_complexes
        self.persistence_stats = persistence_stats
        self.transitions = transitions

        # Calculate zigzag features for regime identification
        self.zigzag_features = self._compute_zigzag_features(
            path_zigzag_diagrams, transitions)

        # Store computation statistics
        self.computation_stats = computation_stats

        print(
            f"Total path zigzag persistence computation completed in {
                time.time() -
                start_time:.2f} seconds"
        )

        return path_zigzag_diagrams
