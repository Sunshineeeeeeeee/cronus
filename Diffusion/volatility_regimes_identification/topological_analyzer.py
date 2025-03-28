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
        
    def compute_temporally_weighted_distance(self, alpha=0.5, beta=0.1, lambda_info=1.0, mi_matrix=None, transfer_entropy=None, chunk_size=5000):
        """
        Compute temporally-weighted distance matrix between data points with optimized memory usage.
        Uses chunking for large datasets and specialized distance weighting for market microstructure.
        
        Parameters:
        -----------
        alpha : float
            Weight for temporal component
        beta : float
            Decay rate for temporal distance
        lambda_info : float
            Weight for mutual information component
        mi_matrix : np.ndarray
            Pre-computed mutual information matrix (feature × feature)
        transfer_entropy : np.ndarray
            Pre-computed transfer entropy matrix (feature × feature)
        chunk_size : int
            Size of chunks for large dataset computation
        """
        start_time = time.time()
        print("Computing temporally-weighted distance matrix...")
        
        # Convert timestamps to numerical values (seconds since first timestamp)
        time_indices = np.zeros(len(self.timestamps))
        
        if np.issubdtype(self.timestamps.dtype, np.datetime64):
            # Convert numpy datetime64 to seconds
            t0 = self.timestamps[0]
            time_diffs = self.timestamps - t0
            time_indices = time_diffs.astype('timedelta64[ns]').astype(np.float64) / 1e9
        elif isinstance(self.timestamps[0], datetime):
            # Convert Python datetime objects to seconds
            t0 = self.timestamps[0]
            for i, t in enumerate(self.timestamps):
                time_indices[i] = (t - t0).total_seconds()
        else:
            # Try to convert numeric array
            time_indices = np.array([float(t) for t in self.timestamps])
        
        # Normalize time indices to [0,1] range
        if len(time_indices) > 1:
            min_time = np.min(time_indices)
            max_time = np.max(time_indices)
            if max_time > min_time:
                time_indices = (time_indices - min_time) / (max_time - min_time)
        
        n_samples = len(self.features)
        n_features = self.features.shape[1]
        
        # Create feature weights based on mutual information if provided
        feature_weights = np.ones(n_features)
        
        if mi_matrix is not None and lambda_info > 0:
            # Check if dimensions match our feature count
            if mi_matrix.shape[0] == n_features and mi_matrix.shape[1] == n_features:
                # Calculate importance of each feature based on average MI with other features
                mi_importance = np.mean(mi_matrix, axis=1)  # Changed from sum to mean
                mi_importance = np.nan_to_num(mi_importance, nan=0.0)  # Replace NaNs with 0
                
                # Transform MI importance to weights: higher MI means lower weight
                if np.max(mi_importance) > np.min(mi_importance):
                    # Normalize to [0,1] range first
                    normalized_mi = (mi_importance - np.min(mi_importance)) / (np.max(mi_importance) - np.min(mi_importance))
                    # Apply sigmoid-like transformation to get weights in [0.2, 1.0] range
                    feature_weights = 0.2 + 0.8 / (1.0 + lambda_info * normalized_mi)
                
                print(f"Applied MI-based feature weighting: min={feature_weights.min():.4f}, max={feature_weights.max():.4f}")
            else:
                print(f"Warning: MI matrix dimensions ({mi_matrix.shape}) don't match feature count ({n_features}). Using uniform weights.")
        
        # Also incorporate transfer entropy into feature weights if provided
        if transfer_entropy is not None and lambda_info > 0:
            # Check if dimensions match
            if transfer_entropy.shape[0] == n_features and transfer_entropy.shape[1] == n_features:
                # Use sum of outgoing transfer entropy to weight features
                # Higher TE means the feature has more causal influence
                te_importance = np.sum(transfer_entropy, axis=1)
                te_importance = np.nan_to_num(te_importance, nan=0.0)  # Replace NaNs with 0
                
                # Enhance weights for features with high causal influence
                if np.max(te_importance) > 0:
                    te_weights = 1.0 + lambda_info * te_importance / np.max(te_importance)
                    # Combine with MI-based weights
                    feature_weights *= te_weights
                
                print(f"Applied TE-based feature weighting: min={feature_weights.min():.4f}, max={feature_weights.max():.4f}")
            else:
                print(f"Warning: TE matrix dimensions ({transfer_entropy.shape}) don't match feature count ({n_features}). Ignoring TE.")
        
        # Normalize feature weights
        if np.max(feature_weights) > 0:
            feature_weights = feature_weights / np.max(feature_weights)
        
        # For large datasets, use chunking to avoid memory issues
        if n_samples > chunk_size:
            print(f"Using chunking for large dataset ({n_samples} samples)")
            dist_matrix = np.zeros((n_samples, n_samples))
            
            # Process by chunks to reduce memory usage
            for i in range(0, n_samples, chunk_size):
                i_end = min(i + chunk_size, n_samples)
                chunk_i = self.features[i:i_end]
                time_i = time_indices[i:i_end]
                
                for j in range(0, n_samples, chunk_size):
                    j_end = min(j + chunk_size, n_samples)
                    chunk_j = self.features[j:j_end]
                    time_j = time_indices[j:j_end]
                    
                    # Compute weighted distances for this chunk
                    dists_chunk = self._compute_weighted_chunk_distances(
                        chunk_i, chunk_j, time_i, time_j, alpha, beta, feature_weights
                    )
                    
                    # Store in the main distance matrix
                    dist_matrix[i:i_end, j:j_end] = dists_chunk
                    
                print(f"Processed chunks up to {i_end}/{n_samples}")
        else:
            # Standard computation for smaller datasets
            try:
                # Apply feature weighting to the input data
                weighted_features = self.features * feature_weights[np.newaxis, :]
                
                # Try vectorized computation with weighted features
                diff = weighted_features[:, np.newaxis, :] - weighted_features[np.newaxis, :, :]
                base_distances = np.sqrt(np.sum(diff * diff, axis=2))
                
                # Compute temporal weights matrix
                time_diffs = np.abs(time_indices[:, np.newaxis] - time_indices[np.newaxis, :])
                temporal_weights = 1.0 + alpha * (1.0 - np.exp(-beta * time_diffs))
                
                # Apply temporal weighting
                dist_matrix = base_distances * temporal_weights
                
            except MemoryError:
                # Fall back to loop-based computation
                print("Using loop-based distance computation due to memory constraints...")
                dist_matrix = np.zeros((n_samples, n_samples))
                
                for i in range(n_samples):
                    for j in range(i+1, n_samples):
                        # Compute weighted feature differences
                        diff = self.features[i] - self.features[j]
                        weighted_diff = diff * feature_weights
                        dist = np.sqrt(np.sum(weighted_diff * weighted_diff))
                        
                        # Apply temporal weighting
                        time_diff = abs(time_indices[i] - time_indices[j])
                        temporal_weight = 1.0 + alpha * (1.0 - np.exp(-beta * time_diff))
                        
                        # Store weighted distance
                        dist_matrix[i, j] = dist * temporal_weight
                        dist_matrix[j, i] = dist_matrix[i, j]  # Symmetric
        
        # Ensure the matrix is symmetric
        dist_matrix = 0.5 * (dist_matrix + dist_matrix.T)
        
        # Set diagonal to 0
        np.fill_diagonal(dist_matrix, 0)
        
        # Replace any NaN or inf values
        dist_matrix = np.nan_to_num(dist_matrix, nan=0.0, posinf=np.max(dist_matrix[~np.isinf(dist_matrix)]) if np.any(~np.isinf(dist_matrix)) else 1.0)
        
        # Store the result
        self.distance_matrix = dist_matrix
        self.temporal_distance_matrix = dist_matrix
        
        print(f"Distance matrix computation completed in {time.time() - start_time:.2f} seconds")
        print(f"Distance matrix shape: {dist_matrix.shape}")
        print(f"Distance range: [{dist_matrix.min():.4f}, {dist_matrix.max():.4f}]")
        
        return dist_matrix
    
    def _compute_weighted_chunk_distances(self, chunk_i, chunk_j, time_i, time_j, alpha, beta, feature_weights):
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
                for j in range(i+1, nj):
                    # Weighted Euclidean distance
                    diff = chunk_i[i] - chunk_j[j]
                    weighted_diff = diff * feature_weights
                    dist = np.sqrt(np.sum(weighted_diff * weighted_diff))
                    
                    # Temporal weighting
                    time_diff = abs(time_i[i] - time_j[j])
                    temporal_weight = 1.0 + alpha * (1.0 - np.exp(-beta * time_diff))
                    
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
                    temporal_weight = 1.0 + alpha * (1.0 - np.exp(-beta * time_diff))
                    
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
            G.add_node(i, features=self.features[i], timestamp=self.timestamps[i])
        
        # Create edges based on distance threshold and temporal ordering
        if enforce_temporal:
            # Only create edges that respect temporal ordering (i < j)
            # Convert matrices to float type to ensure compatibility
            distance_mask = (self.temporal_distance_matrix <= epsilon).astype(float)
            temporal_mask = np.triu(np.ones_like(self.temporal_distance_matrix), k=1).astype(float)
            combined_mask = (distance_mask * temporal_mask) > 0
            rows, cols = np.where(combined_mask)
        else:
            # Create edges in both directions
            rows, cols = np.where(self.temporal_distance_matrix <= epsilon)
            
        # Add edges with weights
        for i, j in zip(rows, cols):
            if i != j:  # Avoid self-loops
                weight = 1.0 / max(self.temporal_distance_matrix[i, j], 1e-10)
                G.add_edge(i, j, weight=weight, distance=self.temporal_distance_matrix[i, j])
        
        print(f"Constructed network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def compute_path_complex(self, G, max_path_length=2):
        """
        Generate a directed path complex from the graph.
        For market microstructure data, path homology is more appropriate than simplicial complexes.
        
        Parameters:
        -----------
        G : networkx.DiGraph
            Directed graph representation of the data
        max_path_length : int
            Maximum length of paths to consider (fixed to 2)
            
        Returns:
        --------
        dict
            Dictionary of paths by dimension
        """
        max_path_length = min(max_path_length, 2)  # Ensure max path length is at most 2
        print(f"Computing path complex with max path length {max_path_length}...")
        
        # Initialize path complex
        path_complex = {k: [] for k in range(max_path_length + 1)}
        
        # Add 0-simplices (vertices)
        for node in G.nodes():
            path_complex[0].append((node,))
            
        # Add 1-simplices (edges)
        for u, v in G.edges():
            path_complex[1].append((u, v))
            
        # Add higher-order paths - this is computationally expensive but builds the full complex
        for k in range(2, max_path_length + 1):
            print(f"Building dimension {k} paths...")
            start_time = time.time()
            path_count = 0
            
            for path in path_complex[k-1]:
                # Last vertex in the path
                last = path[-1]
                
                # Try to extend the path
                for neighbor in G.successors(last):
                    # Avoid cycles
                    if neighbor not in path:
                        new_path = path + (neighbor,)
                        path_complex[k].append(new_path)
                        path_count += 1
                        
                        # Print progress for large complexes
                        if path_count % 1000000 == 0:
                            print(f"  Created {path_count} paths in dimension {k} ({time.time() - start_time:.2f}s)")
            
            print(f"Created {path_count} paths in dimension {k} ({time.time() - start_time:.2f}s)")
                        
        return path_complex
    
    def _compute_homology_from_boundary(self, boundary_matrix):
        """
        Helper method to compute homology from boundary matrix.
        
        Parameters:
        -----------
        boundary_matrix : numpy.ndarray
            Boundary matrix of the complex
            
        Returns:
        --------
        dict
            Dictionary containing homology information
        """
        try:
            # Check if matrix is empty or trivial
            if boundary_matrix.size == 0 or boundary_matrix.shape[0] == 0 or boundary_matrix.shape[1] == 0:
                return {
                    'rank': 0,
                    'nullity': 0,
                    'betti': 0
                }
                
            # Compute rank of boundary matrix
            rank = np.linalg.matrix_rank(boundary_matrix)
            
            # Compute nullity
            nullity = boundary_matrix.shape[0] - rank
            
            # Compute Betti numbers
            betti = nullity - rank
            
            # Ensure non-negative Betti numbers
            betti = max(0, betti)
            
            return {
                'rank': rank,
                'nullity': nullity,
                'betti': betti
            }
            
        except Exception as e:
            print(f"Warning: Error in homology computation: {str(e)}")
            # Return safe fallback values
            return {
                'rank': 0,
                'nullity': boundary_matrix.shape[0] if hasattr(boundary_matrix, 'shape') else 0,
                'betti': 0
            }
    
    def compute_persistent_path_zigzag_homology(self, window_size=100, overlap=50, max_path_length=2, min_epsilon=0.1, max_epsilon=2.0, num_steps=10, output_dir=None):
        """
        Compute persistent path homology with zigzag persistence, specifically designed for sequential market microstructure data.
        This combines the directed path complex approach with zigzag persistence to capture temporal evolution of market states.
        
        Parameters:
        -----------
        window_size : int
            Size of sliding window for state computation
        overlap : int
            Number of points to overlap between windows
        max_path_length : int
            Maximum length of paths to consider
        min_epsilon : float
            Minimum distance threshold
        max_epsilon : float
            Maximum distance threshold
        num_steps : int
            Number of filtration steps
        output_dir : str
            Directory to save output files and plots
            
        Returns:
        --------
        dict
            Dictionary containing path zigzag persistence diagrams
        """
        start_time = time.time()
        print("Computing persistent path zigzag homology for market microstructure...")
        
        try:
            # Create sequence of directed networks for each window
            network_start = time.time()
            networks = []
            window_indices = []
            
            # Pre-compute temporal distance matrix if not already done
            if self.temporal_distance_matrix is None:
                self.compute_temporally_weighted_distance()
                self.temporal_distance_matrix = self.distance_matrix
                
            # Ensure distance matrix has no NaN values
            dist_matrix = np.nan_to_num(self.temporal_distance_matrix)
            
            for i in range(0, self.n_samples - window_size + 1, window_size - overlap):
                # Get window data indices
                window_idx = list(range(i, min(i + window_size, self.n_samples)))
                if len(window_idx) < 3:  # Skip too small windows
                    continue
                    
                window_indices.append(window_idx)
                
                # Create temporal distance submatrix for this window
                window_dist_matrix = dist_matrix[np.ix_(window_idx, window_idx)]
                
                # Create directed network for this window
                G = nx.DiGraph()
                
                # Add nodes
                for j, idx in enumerate(window_idx):
                    G.add_node(j, original_idx=idx, features=self.features[idx], timestamp=self.timestamps[idx])
                
                # Add edges (respecting temporal ordering)
                for j in range(len(window_idx)):
                    for k in range(j+1, len(window_idx)):  # Enforce j < k for temporal ordering
                        if window_dist_matrix[j, k] <= min_epsilon:
                            G.add_edge(j, k, weight=1.0/max(window_dist_matrix[j, k], 1e-10))
                
                networks.append(G)
            
            if not networks:
                print("No valid windows found. Try reducing window_size or increasing overlap.")
                return None
                
            print(f"Network creation completed in {time.time() - network_start:.2f} seconds")
            
            # For each network window, compute path complex
            print(f"Computing path complexes for {len(networks)} windows...")
            path_complexes = []
            for i, G in enumerate(networks):
                print(f"Computing path complex for window {i+1}/{len(networks)}...")
                path_complex = self.compute_path_complex(G, max_path_length)
                path_complexes.append(path_complex)
                print(f"Window {i+1} path complex sizes: " + ", ".join([f"dim {d}: {len(paths)}" for d, paths in path_complex.items()]))
            
            # Initialize zigzag path persistence
            path_zigzag_diagrams = {
                'window_diagrams': [],
                'transitions': [],
                'betti_numbers': {},
                'window_indices': window_indices
            }
            
            # Process each dimension
            dimension_start = time.time()
            persistence_summary = {0: [], 1: [], 2: []}  # Store persistence info for each dimension

            for dim in range(max_path_length + 1):
                if dim > 2:  # Skip dimensions higher than 2
                    continue
                print(f"Processing homology in dimension {dim}...")
                dim_start = time.time()
                betti_series = []
                
                # Compute homology for each window
                for i, path_complex in enumerate(path_complexes):
                    if dim not in path_complex or not path_complex[dim]:
                        betti_series.append(0)
                        continue
                    
                    paths = path_complex[dim]
                    
                    if dim == 0:
                        # For 0-dimension, Betti number is number of connected components
                        betti = nx.number_connected_components(networks[i].to_undirected())
                        betti_series.append(betti)
                        persistence_summary[dim].append(betti)
                        continue
                        
                    # Create boundary matrix
                    n_paths = len(paths)
                    
                    if dim-1 not in path_complex or not path_complex[dim-1]:
                        betti_series.append(0)
                        continue
                        
                    n_boundaries = len(path_complex[dim-1])
                    boundary_matrix = np.zeros((n_paths, n_boundaries))
                    
                    # Fill boundary matrix
                    for j, path in enumerate(paths):
                        # Get boundary paths
                        for k in range(len(path) - 1):
                            boundary_path = path[:k] + path[k+1:]
                            try:
                                boundary_idx = path_complex[dim-1].index(boundary_path)
                                boundary_matrix[j, boundary_idx] = 1
                            except ValueError:
                                continue
                    
                    # Compute homology
                    homology = self._compute_homology_from_boundary(boundary_matrix)
                    betti = homology['betti']
                    betti_series.append(betti)
                    persistence_summary[dim].append(betti)
                
                path_zigzag_diagrams[f'betti_{dim}'] = betti_series
                print(f"Dimension {dim} processing completed in {time.time() - dim_start:.2f} seconds")
            
            # Print persistence analysis
            print("\n=== Persistence Analysis ===")
            print("Found the following persistent features across windows:")
            for dim in range(3):
                persistent_count = sum(1 for b in persistence_summary[dim] if b > 0)
                avg_betti = np.mean(persistence_summary[dim]) if persistence_summary[dim] else 0
                max_betti = max(persistence_summary[dim]) if persistence_summary[dim] else 0
                
                print(f"\nDimension {dim}:")
                print(f"- Persistent features found in {persistent_count} out of {len(persistence_summary[dim])} windows")
                print(f"- Average Betti number: {avg_betti:.2f}")
                print(f"- Maximum Betti number: {max_betti}")
                print(f"- Betti number sequence: {persistence_summary[dim]}")

            print("\nPersistence Stability Analysis:")
            for dim in range(3):
                betti_series = path_zigzag_diagrams[f'betti_{dim}']
                if betti_series:
                    changes = np.diff(betti_series)
                    stability = 1.0 - (np.count_nonzero(changes) / len(changes)) if len(changes) > 0 else 1.0
                    print(f"Dimension {dim} stability: {stability:.2%}")
            print("===========================\n")
            
            # Compute zigzag persistence across windows
            print("Computing transitions between windows...")
            transition_features = []
            
            # For each pair of consecutive windows, analyze transitions
            for i in range(len(window_indices) - 1):
                current_window = set(window_indices[i])
                next_window = set(window_indices[i+1])
                
                # Find overlap indices
                overlap_indices = current_window.intersection(next_window)
                
                if not overlap_indices:
                    continue
                
                # Track transitions of paths
                for dim in range(1, max_path_length + 1):  # Skip 0-dim (just points)
                    if dim in path_complexes[i] and dim in path_complexes[i+1]:
                        # Find paths in current window that have elements in the overlap
                        persistent_paths = []
                        for path in path_complexes[i][dim]:
                            # Check if path nodes are in overlap (need original indices)
                            path_nodes_in_overlap = False
                            for node in path:
                                if node < len(networks[i].nodes) and 'original_idx' in networks[i].nodes[node]:
                                    original_idx = networks[i].nodes[node]['original_idx']
                                    if original_idx in overlap_indices:
                                        path_nodes_in_overlap = True
                                        break
                            
                            if path_nodes_in_overlap:
                                persistent_paths.append(path)
                        
                        transition_features.append({
                            'window_pair': (i, i+1),
                            'dimension': dim,
                            'persistent_paths': len(persistent_paths)
                        })
            
            path_zigzag_diagrams['transitions'] = transition_features
            
            # Generate epsilon stepping for filtration
            epsilon_values = np.linspace(min_epsilon, max_epsilon, num_steps)
            
            # Store results
            self.path_zigzag_diagrams = {
                'window_complexes': path_complexes,
                'window_networks': networks,
                'window_indices': window_indices,
                'betti_numbers': path_zigzag_diagrams,
                'epsilon_values': epsilon_values,
                'window_size': window_size,
                'overlap': overlap
            }
            
            # Generate visualizations
            if output_dir is not None:
                self._generate_zigzag_visualizations(output_dir)
            
            print(f"Total path zigzag persistence computation completed in {time.time() - start_time:.2f} seconds")
            return self.path_zigzag_diagrams
            
        except Exception as e:
            print(f"Error in path zigzag persistence computation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def _generate_zigzag_visualizations(self, output_dir):
        """Helper method to generate visualizations for zigzag persistence results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot Betti numbers across windows
        plt.figure(figsize=(12, 6))
        
        for dim in range(len(self.path_zigzag_diagrams['window_complexes'][0])):
            betti_key = f'betti_{dim}'
            if betti_key in self.path_zigzag_diagrams['betti_numbers']:
                betti_values = self.path_zigzag_diagrams['betti_numbers'][betti_key]
                if betti_values:  # Only plot if we have values
                    plt.plot(range(len(betti_values)), betti_values, 
                           label=f'H{dim}', linewidth=2, marker='o')
        
        plt.xlabel('Window Index')
        plt.ylabel('Betti Number')
        plt.title('Path Zigzag Persistence: Betti Numbers Across Windows')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'path_zigzag_betti.png'))
        plt.close()
        
        # Plot transition diagram
        transition_features = self.path_zigzag_diagrams['betti_numbers']['transitions']
        if transition_features:
            plt.figure(figsize=(12, 6))
            
            # Group by dimension
            max_dim = max(t['dimension'] for t in transition_features)
            for dim in range(1, max_dim + 1):
                dim_transitions = [t['persistent_paths'] for t in transition_features 
                                if t['dimension'] == dim]
                if dim_transitions:
                    plt.plot(range(len(dim_transitions)), dim_transitions, 
                           label=f'Dim {dim}', linewidth=2, marker='x')
            
            plt.xlabel('Window Transition')
            plt.ylabel('Persistent Paths')
            plt.title('Path Structure Persistence Across Window Transitions')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'path_zigzag_transitions.png'))
            plt.close()
    
    def _compute_zigzag_transition_strength(self):
        """
        Helper method to compute transition strength from zigzag persistence data.
        
        Returns:
        --------
        numpy.ndarray or None
            Transition strength matrix if zigzag data is available
        """
        try:
            window_indices = self.path_zigzag_diagrams['window_indices']
            transitions = self.path_zigzag_diagrams['betti_numbers']['transitions']
            
            transition_strength = np.zeros((self.n_samples, self.n_samples))
            
            for transition in transitions:
                window_pair = transition['window_pair']
                persistent_paths = transition['persistent_paths']
                dimension = transition['dimension']
                
                # Enhanced weighting scheme
                strength = persistent_paths * (dimension + 1) * np.exp(-dimension)
                
                # Get overlapping indices
                current_window = window_indices[window_pair[0]]
                next_window = window_indices[window_pair[1]]
                overlap_indices = list(set(current_window) & set(next_window))
                
                # Set transition strength with temporal decay
                for i in overlap_indices:
                    for j in overlap_indices:
                        if i < j:
                            temporal_dist = (j - i) / len(overlap_indices)
                            decay = np.exp(-temporal_dist)
                            transition_strength[i, j] += strength * decay
                            transition_strength[j, i] = transition_strength[i, j]
            
            # Normalize transition strength
            if np.max(transition_strength) > 0:
                transition_strength = transition_strength / np.max(transition_strength)
            
            return transition_strength
            
        except Exception as e:
            print(f"Warning: Error computing zigzag transition strength: {str(e)}")
            return None
    
    def identify_regimes(self, n_regimes=3, use_topological_features=True):
        """
        Identify volatility regimes using topological features and enhanced spectral clustering.
        
        Parameters:
        -----------
        n_regimes : int
            Number of regimes to identify
        use_topological_features : bool
            Whether to use extracted topological features for regime identification
            
        Returns:
        --------
        numpy.ndarray
            Array of regime labels for each data point
        """
        start_time = time.time()
        print(f"Identifying {n_regimes} volatility regimes...")
        
        # Ensure we have a distance matrix
        if self.temporal_distance_matrix is None:
            self.compute_temporally_weighted_distance()
        
        # Base feature set with NaN handling
        scaler = RobustScaler(unit_variance=True)
        base_features = np.nan_to_num(self.features, nan=0.0)  # Replace NaNs with 0
        base_features = scaler.fit_transform(base_features)
        
        if use_topological_features:
            # Extract topological features if they haven't been computed
            if not hasattr(self, 'topological_features'):
                topo_features, topo_feature_names = self.extract_topological_features()
                self.topological_features = topo_features
                self.topological_feature_names = topo_feature_names
            
            # Create topological feature matrix per point
            n_samples = self.n_samples
            topo_feature_matrix = np.zeros((n_samples, len(self.topological_features)))
            
            # For sequential data, use window-based assignment of topological features
            if hasattr(self, 'path_zigzag_diagrams') and self.path_zigzag_diagrams is not None:
                window_indices = self.path_zigzag_diagrams['window_indices']
                
                # Assign features to each point based on windows it belongs to
                point_window_count = np.zeros(n_samples)
                
                for i, window_idx in enumerate(window_indices):
                    for idx in window_idx:
                        topo_feature_matrix[idx] += np.nan_to_num(self.topological_features, nan=0.0)
                        point_window_count[idx] += 1
                
                # Average features for points in multiple windows
                for i in range(n_samples):
                    if point_window_count[i] > 0:
                        topo_feature_matrix[i] /= point_window_count[i]
            else:
                # If no zigzag persistence, assign global topological features to all points
                topo_feature_matrix = np.tile(np.nan_to_num(self.topological_features, nan=0.0), (n_samples, 1))
            
            # Normalize topological features
            topo_feature_matrix = np.nan_to_num(topo_feature_matrix, nan=0.0)
            topo_feature_matrix = scaler.fit_transform(topo_feature_matrix)
            
            # Combine base features with topological features
            combined_features = np.column_stack([base_features, topo_feature_matrix])
            print(f"Combined feature matrix shape: {combined_features.shape}")
        else:
            combined_features = base_features
        
        # Compute transition strength from zigzag persistence
        has_zigzag = hasattr(self, 'path_zigzag_diagrams') and self.path_zigzag_diagrams is not None
        
        # Build affinity matrix for clustering with NaN handling
        n_samples = self.n_samples
        affinity = np.zeros((n_samples, n_samples))
        
        # Compute adaptive bandwidth for each point
        distances = squareform(pdist(combined_features))
        distances = np.nan_to_num(distances, nan=np.nanmean(distances))  # Replace NaNs with mean distance
        k = min(15, n_samples - 1)  # k-nearest neighbors
        sorted_distances = np.sort(distances, axis=1)
        local_scales = sorted_distances[:, k].reshape(-1, 1)
        local_scales = np.maximum(local_scales, 1e-8)  # Ensure non-zero bandwidth
        
        # Enhanced temporal coherence with adaptive window
        temporal_coherence = np.zeros((n_samples, n_samples))
        adaptive_window = max(5, int(n_samples * 0.02))  # Adaptive window size
        
        for i in range(n_samples):
            lower_bound = max(0, i - adaptive_window)
            upper_bound = min(n_samples, i + adaptive_window + 1)
            for j in range(lower_bound, upper_bound):
                if i != j:
                    # Exponential decay for temporal coherence
                    temporal_weight = np.exp(-abs(i - j) / adaptive_window)
                    temporal_coherence[i, j] = temporal_weight
        
        # Get transition strength if zigzag persistence was computed
        if has_zigzag:
            transition_strength = self._compute_zigzag_transition_strength()
            if transition_strength is not None:
                transition_strength = np.nan_to_num(transition_strength, nan=0.0)
        else:
            transition_strength = None
        
        # Build the affinity matrix with robust computation
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                try:
                    # Compute feature similarity with adaptive bandwidth
                    feature_dist = distances[i, j]
                    bandwidth = np.mean([local_scales[i], local_scales[j]])
                    base_affinity = np.exp(-feature_dist**2 / (2 * bandwidth**2))
                    
                    # Enhanced temporal weighting
                    temporal_effect = 1.0 + temporal_coherence[i, j]
                    
                    # Incorporate zigzag information if available
                    if has_zigzag and transition_strength is not None:
                        zigzag_boost = 1.0 + transition_strength[i, j] * 2.0
                        affinity[i, j] = base_affinity * temporal_effect * zigzag_boost
                    else:
                        affinity[i, j] = base_affinity * temporal_effect
                    
                    affinity[j, i] = affinity[i, j]  # Symmetric matrix
                except Exception as e:
                    # If computation fails, use a fallback value
                    affinity[i, j] = affinity[j, i] = np.exp(-1.0)  # Conservative fallback
        
        # Clean and normalize affinity matrix
        affinity = np.nan_to_num(affinity, nan=0.0)  # Replace any remaining NaNs
        if np.max(affinity) > 0:
            affinity = affinity / np.max(affinity)
        np.fill_diagonal(affinity, 0)
        
        print(f"Affinity matrix range: [{np.min(affinity):.4f}, {np.max(affinity):.4f}]")
        
        # Try spectral clustering first
        try:
            clustering = SpectralClustering(
                n_clusters=n_regimes,
                affinity='precomputed',
                random_state=42,
                assign_labels='kmeans',
                n_init=20
            )
            regime_labels = clustering.fit_predict(affinity)
            
            # Check regime balance
            regime_counts = np.bincount(regime_labels)
            balance_ratio = np.min(regime_counts) / np.max(regime_counts)
            
            # If severely imbalanced, try alternative clustering
            if balance_ratio < 0.15:
                raise ValueError("Imbalanced clustering detected")
            
        except Exception as e:
            print(f"Spectral clustering failed: {str(e)}")
            print("Falling back to robust GMM clustering...")
            
            # Use PCA for dimensionality reduction
            from sklearn.decomposition import PCA
            
            # Determine number of components to keep 95% of variance
            pca = PCA(n_components=0.95, random_state=42)
            reduced_features = pca.fit_transform(combined_features)
            
            # Fit GMM with multiple initializations
            from sklearn.mixture import GaussianMixture
            
            best_gmm = None
            best_bic = np.inf
            
            for _ in range(5):  # Try multiple initializations
                gmm = GaussianMixture(
                    n_components=n_regimes,
                    covariance_type='full',
                    n_init=20,
                    random_state=np.random.randint(1000)
                )
                gmm.fit(reduced_features)
                
                bic = gmm.bic(reduced_features)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            
            regime_labels = best_gmm.predict(reduced_features)
        
        # Store regime labels
        self.regime_labels = regime_labels
        
        # Print final distribution
        final_counts = np.bincount(regime_labels)
        print(f"Final regime distribution: {final_counts}")
        print(f"Regime balance ratio: {np.min(final_counts) / np.max(final_counts):.3f}")
        print(f"Regime identification completed in {time.time() - start_time:.2f} seconds")
        
        return regime_labels
    
    def extract_topological_features(self):
        """
        Extract topological features from persistent homology and zigzag persistence.
        These features capture the structural characteristics of market microstructure.
        
        Returns:
        --------
        numpy.ndarray
            Array of topological features
        """
        start_time = time.time()
        print("Extracting topological features...")
        
        # Initialize features list
        topo_features = []
        feature_names = []
        
        # Check which computations have been done
        has_persistence = self.persistence_diagrams is not None
        has_zigzag = hasattr(self, 'path_zigzag_diagrams') and self.path_zigzag_diagrams is not None
        
        if not has_persistence and not has_zigzag:
            print("No persistence computations found. Computing persistent homology...")
            self.compute_persistent_homology()
            has_persistence = True
        elif has_zigzag:
            print("Using existing zigzag persistence computations...")
        
        # 1. Features from standard persistence
        if has_persistence:
            # Extract features from persistence diagrams
            persistence = self.persistence_diagrams
            
            # For each dimension, compute summary statistics
            for dim in range(3):  # Dimensions 0, 1, 2 only
                # Get persistence pairs for this dimension
                # GUDHI format: (dimension, (birth, death))
                pairs = [(p[1][0], p[1][1]) for p in persistence if p[0] == dim]
                
                if pairs:
                    # Convert to array for easier computation
                    pairs_array = np.array(pairs)
                    
                    # Filter out infinite death values
                    finite_pairs = pairs_array[np.isfinite(pairs_array[:, 1])]
                    
                    if len(finite_pairs) > 0:
                        # Compute persistence lifetimes
                        lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
                        
                        # Compute summary statistics
                        features = {
                            f'persistence_dim{dim}_count': len(pairs),
                            f'persistence_dim{dim}_max_lifetime': np.max(lifetimes) if len(lifetimes) > 0 else 0,
                            f'persistence_dim{dim}_mean_lifetime': np.mean(lifetimes) if len(lifetimes) > 0 else 0,
                            f'persistence_dim{dim}_std_lifetime': np.std(lifetimes) if len(lifetimes) > 0 else 0,
                            f'persistence_dim{dim}_sum_lifetime': np.sum(lifetimes) if len(lifetimes) > 0 else 0,
                            f'persistence_dim{dim}_birth_mean': np.mean(finite_pairs[:, 0]) if len(finite_pairs) > 0 else 0,
                            f'persistence_dim{dim}_death_mean': np.mean(finite_pairs[:, 1]) if len(finite_pairs) > 0 else 0
                        }
                    else:
                        features = {
                            f'persistence_dim{dim}_count': len(pairs),
                            f'persistence_dim{dim}_max_lifetime': 0,
                            f'persistence_dim{dim}_mean_lifetime': 0,
                            f'persistence_dim{dim}_std_lifetime': 0,
                            f'persistence_dim{dim}_sum_lifetime': 0,
                            f'persistence_dim{dim}_birth_mean': 0,
                            f'persistence_dim{dim}_death_mean': 0
                        }
                else:
                    features = {
                        f'persistence_dim{dim}_count': 0,
                        f'persistence_dim{dim}_max_lifetime': 0,
                        f'persistence_dim{dim}_mean_lifetime': 0,
                        f'persistence_dim{dim}_std_lifetime': 0,
                        f'persistence_dim{dim}_sum_lifetime': 0,
                        f'persistence_dim{dim}_birth_mean': 0,
                        f'persistence_dim{dim}_death_mean': 0
                    }
                
                # Add to feature lists
                for feat_name, feat_val in features.items():
                    topo_features.append(feat_val)
                    feature_names.append(feat_name)
            
            # Add Betti curve features if available
            if self.betti_curves is not None:
                for dim, curve in self.betti_curves['curves'].items():
                    # Compute statistics on the Betti curve
                    if len(curve) > 0:
                        features = {
                            f'betti_curve_dim{dim}_max': np.max(curve),
                            f'betti_curve_dim{dim}_mean': np.mean(curve),
                            f'betti_curve_dim{dim}_integral': np.trapz(curve, self.betti_curves['values'])
                        }
                    else:
                        features = {
                            f'betti_curve_dim{dim}_max': 0,
                            f'betti_curve_dim{dim}_mean': 0,
                            f'betti_curve_dim{dim}_integral': 0
                        }
                    
                    # Add to feature lists
                    for feat_name, feat_val in features.items():
                        topo_features.append(feat_val)
                        feature_names.append(feat_name)
        
        # 2. Features from zigzag persistence
        if has_zigzag:
            # Extract Betti number profiles from each dimension
            for dim in range(3):  # Dimensions 0, 1, 2 only
                betti_key = f'betti_{dim}'
                if betti_key in self.path_zigzag_diagrams['betti_numbers']:
                    betti_values = self.path_zigzag_diagrams['betti_numbers'][betti_key]
                    
                    if betti_values:
                        features = {
                            f'zigzag_dim{dim}_max_betti': np.max(betti_values),
                            f'zigzag_dim{dim}_mean_betti': np.mean(betti_values),
                            f'zigzag_dim{dim}_std_betti': np.std(betti_values),
                            f'zigzag_dim{dim}_changepoints': np.sum(np.abs(np.diff(betti_values)) > 0)
                        }
                    else:
                        features = {
                            f'zigzag_dim{dim}_max_betti': 0,
                            f'zigzag_dim{dim}_mean_betti': 0,
                            f'zigzag_dim{dim}_std_betti': 0,
                            f'zigzag_dim{dim}_changepoints': 0
                        }
                else:
                    features = {
                        f'zigzag_dim{dim}_max_betti': 0,
                        f'zigzag_dim{dim}_mean_betti': 0,
                        f'zigzag_dim{dim}_std_betti': 0,
                        f'zigzag_dim{dim}_changepoints': 0
                    }
                
                # Add to feature lists
                for feat_name, feat_val in features.items():
                    topo_features.append(feat_val)
                    feature_names.append(feat_name)
        
        # Convert to array
        topo_features_array = np.array(topo_features)
        
        print(f"Extracted {len(topo_features)} topological features in {time.time() - start_time:.2f} seconds")
        
        return topo_features_array, feature_names
    
    def analyze_regimes(self, output_dir=None):
        """
        Analyze identified regimes and generate visualizations.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save output files and plots
            
        Returns:
        --------
        dict
            Dictionary containing regime analysis results
        """
        if self.regime_labels is None:
            raise ValueError("Regimes must be identified first")
        
        print("Analyzing volatility regimes...")
        
        # Create output directory if specified
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results
        regime_analysis = {
            'regime_counts': np.bincount(self.regime_labels),
            'regime_stats': {},
            'transition_points': []
        }
        
        # Calculate statistics for each regime
        for regime in range(max(self.regime_labels) + 1):
            regime_mask = self.regime_labels == regime
            regime_features = self.features[regime_mask]
            
            # Basic statistics
            stats = {
                'count': np.sum(regime_mask),
                'mean': np.mean(regime_features, axis=0),
                'std': np.std(regime_features, axis=0),
                'min': np.min(regime_features, axis=0),
                'max': np.max(regime_features, axis=0)
            }
            
            regime_analysis['regime_stats'][regime] = stats
        
        # Find regime transition points
        transitions = []
        for i in range(1, len(self.regime_labels)):
            if self.regime_labels[i] != self.regime_labels[i-1]:
                transitions.append(i)
        
        regime_analysis['transition_points'] = transitions
        
        # Generate plots if output directory specified
        if output_dir is not None:
            # Plot regime distribution
            plt.figure(figsize=(12, 6))
            regime_counts = regime_analysis['regime_counts']
            plt.bar(range(len(regime_counts)), regime_counts)
            plt.xlabel('Regime')
            plt.ylabel('Count')
            plt.title('Regime Distribution')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'regime_distribution.png'))
            plt.close()
            
            # Plot regime timeline
            plt.figure(figsize=(15, 6))
            plt.plot(self.regime_labels, '-o', markersize=3, alpha=0.5)
            plt.xlabel('Time')
            plt.ylabel('Regime')
            plt.title('Regime Timeline')
            plt.grid(True, alpha=0.3)
            
            # Mark transition points
            for transition in transitions:
                plt.axvline(x=transition, color='r', linestyle='--', alpha=0.5)
            
            plt.savefig(os.path.join(output_dir, 'regime_timeline.png'))
            plt.close()
            
            # Plot topological features by regime if available
            if hasattr(self, 'topological_features') and hasattr(self, 'topological_feature_names'):
                # Select top features
                n_features = min(10, len(self.topological_feature_names))
                feature_importance = np.zeros(len(self.topological_feature_names))
                
                # Simple feature importance: variance between regimes
                for i, feature_name in enumerate(self.topological_feature_names):
                    values_by_regime = []
                    for regime in range(max(self.regime_labels) + 1):
                        regime_mask = self.regime_labels == regime
                        if np.sum(regime_mask) > 0:
                            # For simplicity, use global topological features
                            values_by_regime.append(self.topological_features[i])
                    
                    if values_by_regime:
                        feature_importance[i] = np.std(values_by_regime)
                
                # Get indices of top features
                top_indices = np.argsort(-feature_importance)[:n_features]
                
                # Plot top features by regime
                plt.figure(figsize=(15, 8))
                x = np.arange(max(self.regime_labels) + 1)
                width = 0.8 / n_features
                
                for i, idx in enumerate(top_indices):
                    feature_name = self.topological_feature_names[idx]
                    values_by_regime = []
                    
                    for regime in range(max(self.regime_labels) + 1):
                        values_by_regime.append(self.topological_features[idx])
                    
                    plt.bar(x + i * width - 0.4, values_by_regime, width, label=feature_name)
                
                plt.xlabel('Regime')
                plt.ylabel('Feature Value')
                plt.title('Top Topological Features by Regime')
                plt.xticks(x)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'topological_features_by_regime.png'))
                plt.close()
        
        return regime_analysis

    def build_sparse_directed_network(self, epsilon=0.5, k_neighbors=None, enforce_temporal=True):
        """
        Build an efficient sparse directed network from the distance matrix.
        For large datasets, uses k-nearest neighbors approach to limit edges.
        
        Parameters:
        -----------
        epsilon : float
            Distance threshold for connecting points
        k_neighbors : int or None
            If provided, use k-nearest neighbors instead of epsilon threshold
        enforce_temporal : bool
            If True, only create edges that respect temporal ordering (i < j)
            
        Returns:
        --------
        tuple
            (adjacency_matrix, edge_weights)
        """
        start_time = time.time()
        print(f"Building sparse directed network...")
        
        if self.temporal_distance_matrix is None:
            raise ValueError("Distance matrix must be computed first")
        
        n_samples = self.n_samples
        
        # Create sparse adjacency matrix
        adjacency = lil_matrix((n_samples, n_samples), dtype=float)
        
        if k_neighbors is not None and k_neighbors < n_samples:
            # K-nearest neighbors approach for sparse network
            print(f"Using {k_neighbors}-nearest neighbors approach")
            
            # Initialize nearest neighbors model
            nn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree')
            nn.fit(self.features)
            
            # Get k nearest neighbors for each point
            distances, indices = nn.kneighbors(self.features)
            
            # Create edges based on nearest neighbors
            for i in range(n_samples):
                for j_idx, j in enumerate(indices[i]):
                    # Skip self-loops
                    if i == j:
                        continue
                    
                    # Enforce temporal ordering if required
                    if enforce_temporal and i >= j:
                        continue
                    
                    # Add edge with weight
                    weight = 1.0 / max(distances[i][j_idx], 1e-10)
                    adjacency[i, j] = weight
        else:
            # Epsilon threshold approach
            if enforce_temporal:
                # Only create edges that respect temporal ordering (i < j)
                # Convert matrices to float type to ensure compatibility
                distance_mask = (self.temporal_distance_matrix <= epsilon).astype(float)
                temporal_mask = np.triu(np.ones_like(self.temporal_distance_matrix), k=1).astype(float)
                combined_mask = (distance_mask * temporal_mask) > 0
                rows, cols = np.where(combined_mask)
            else:
                # Create edges in both directions
                rows, cols = np.where(self.temporal_distance_matrix <= epsilon)
                
            # Add edges with weights
            for i, j in zip(rows, cols):
                if i != j:  # Avoid self-loops
                    weight = 1.0 / max(self.temporal_distance_matrix[i, j], 1e-10)
                    adjacency[i, j] = weight
        
        # Convert to CSR format for efficient operations
        adjacency = adjacency.tocsr()
        
        print(f"Built sparse directed network with {adjacency.getnnz()} edges in {time.time() - start_time:.2f} seconds")
        print(f"Sparsity: {adjacency.getnnz() / (n_samples * n_samples):.6f}")
        
        return adjacency
    
    def build_directed_filtration(self, max_dimension=2, max_simplex_size=None):
        """
        Build a directed filtration representing the sequential market microstructure.
        Uses GUDHI's SimplexTree structure for efficient computation.
        
        Parameters:
        -----------
        max_dimension : int
            Maximum homology dimension to compute
        max_simplex_size : int or None
            Maximum size of simplices to include in filtration
            
        Returns:
        --------
        gudhi.SimplexTree
            Simplex tree containing the filtration
        """
        start_time = time.time()
        print(f"Building directed filtration (max dim={max_dimension})...")
        
        # Build network if not already done
        if not hasattr(self, 'adjacency') or self.adjacency is None:
            self.adjacency = self.build_sparse_directed_network()
        
        # Create simplex tree
        st = SimplexTree()
        
        # Add vertices (0-simplices)
        for i in range(self.n_samples):
            st.insert([i], filtration=0.0)
        
        # Get edges and their weights
        cx, cy = self.adjacency.nonzero()
        edges = list(zip(cx, cy))
        weights = self.adjacency.data
        
        # Normalize weights to [0,1] range
        if len(weights) > 0:
            min_weight = np.min(weights)
            max_weight = np.max(weights)
            if max_weight > min_weight:
                weights = (weights - min_weight) / (max_weight - min_weight)
        
        # Add edges (1-simplices) with normalized filtration values
        edge_dict = {}  # Store edges and their filtration values for higher dimensions
        for (i, j), weight in zip(edges, weights):
            filtration_value = weight
            st.insert([i, j], filtration=filtration_value)
            edge_dict[(i, j)] = filtration_value
        
        # Add higher dimensional simplices
        if max_dimension >= 2:
            # Create graph for finding triangles and higher simplices
            G = nx.DiGraph()
            G.add_weighted_edges_from([(i, j, w) for (i, j), w in zip(edges, weights)])
            
            # Convert to undirected graph for triangle finding
            G_undirected = G.to_undirected()
            
            # Find triangles using networkx's find_cliques
            triangles = []
            for clique in nx.find_cliques(G_undirected):
                if len(clique) == 3:  # We only want triangles
                    i, j, k = sorted(clique)  # Sort to ensure consistent ordering
                    # Only include if all directed edges exist and respect ordering
                    if (G.has_edge(i, j) and G.has_edge(j, k) and G.has_edge(i, k)):
                        # Get maximum filtration value of constituent edges
                        filt_val = max(edge_dict.get((i, j), 1.0),
                                     edge_dict.get((j, k), 1.0),
                                     edge_dict.get((i, k), 1.0))
                        triangles.append(([i, j, k], filt_val))
            
            # Add triangles to filtration
            for simplex, filt_val in triangles:
                st.insert(simplex, filtration=filt_val)
        
        # Print simplex counts by dimension
        print("Simplex counts by dimension:")
        for dim in range(max_dimension + 1):
            count = st.num_simplices_by_dimension(dim)
            print(f"  Dimension {dim}: {count}")
        
        return st
    
    def compute_persistent_homology(self, max_dimension=2, min_persistence=0.01):
        """
        Compute persistent homology using the built filtration.
        
        Parameters:
        -----------
        max_dimension : int
            Maximum homology dimension to compute
        min_persistence : float
            Minimum persistence value to consider (noise threshold)
            
        Returns:
        --------
        list
            Persistence diagram (list of (dim, birth, death) tuples)
        """
        start_time = time.time()
        print(f"Computing persistent homology (max dim={max_dimension})...")
        
        # Build filtration if not already done
        if self.simplex_tree is None:
            self.build_directed_filtration(max_dimension=max_dimension)
        
        # Compute persistence
        self.simplex_tree.compute_persistence(min_persistence=min_persistence)
        
        # Extract persistence diagram
        persistence = self.simplex_tree.persistence()
        
        # Get persistence pairs
        persistence_pairs = self.simplex_tree.persistence_pairs()
        
        # Store persistence diagrams
        self.persistence_diagrams = persistence
        
        # Compute Betti curves
        self._compute_betti_curves()
        
        # Print persistence summary
        print(f"Computed persistent homology in {time.time() - start_time:.2f} seconds")
        print("Persistence diagram summary:")
        for dim in range(max_dimension + 1):
            dim_pairs = [p for p in persistence if p[0] == dim]
            print(f"  Dimension {dim}: {len(dim_pairs)} persistence pairs")
        
        return persistence
    
    def _compute_betti_curves(self, num_points=100):
        """
        Compute Betti curves from the persistence diagram.
        A Betti curve shows how Betti numbers change across filtration values.
        
        Parameters:
        -----------
        num_points : int
            Number of points in the Betti curve
            
        Returns:
        --------
        dict
            Dictionary of Betti curves by dimension
        """
        if self.persistence_diagrams is None:
            raise ValueError("Persistence diagrams must be computed first")
        
        # Get the filtration range
        min_val = float('inf')
        max_val = 0
        
        # GUDHI persistence format is [(dimension, (birth, death)), ...]
        for dim_pair in self.persistence_diagrams:
            dim = dim_pair[0]  # dimension
            birth = dim_pair[1][0]  # birth time
            death = dim_pair[1][1]  # death time
            
            if birth < min_val:
                min_val = birth
            if death > max_val and death != float('inf'):
                max_val = death
        
        # Handle the case of infinite max_val
        if max_val == 0:
            max_val = 1.0
        
        # Create filtration values
        filtration_values = np.linspace(min_val, max_val, num_points)
        
        # Initialize Betti curves
        max_dim = max([dim_pair[0] for dim_pair in self.persistence_diagrams]) if self.persistence_diagrams else 0
        betti_curves = {dim: np.zeros(num_points) for dim in range(max_dim + 1)}
        
        # Compute Betti numbers at each filtration value
        for i, filt_val in enumerate(filtration_values):
            for dim_pair in self.persistence_diagrams:
                dim = dim_pair[0]
                birth = dim_pair[1][0]
                death = dim_pair[1][1]
                
                if birth <= filt_val < death:
                    betti_curves[dim][i] += 1
        
        # Store Betti curves
        self.betti_curves = {
            'values': filtration_values,
            'curves': betti_curves
        }
        
        return self.betti_curves
    
    def compute_zigzag_persistence(self, window_size=100, overlap=50, max_dimension=2, min_persistence=0.01):
        """
        Compute zigzag persistence across sliding windows to preserve sequential structure.
        This is crucial for market microstructure data where temporal sequence matters.
        
        Parameters:
        -----------
        window_size : int
            Size of each window
        overlap : int
            Overlap between consecutive windows
        max_dimension : int
            Maximum homology dimension to compute
        min_persistence : float
            Minimum persistence value to consider
            
        Returns:
        --------
        dict
            Dictionary of zigzag persistence results
        """
        start_time = time.time()
        print(f"Computing zigzag persistence with window_size={window_size}, overlap={overlap}...")
        
        # Ensure distance matrix is computed
        if self.temporal_distance_matrix is None:
            self.compute_temporally_weighted_distance()
        
        # Create windows with overlap
        n_samples = self.n_samples
        window_indices = []
        
        for i in range(0, n_samples - window_size + 1, window_size - overlap):
            window_idx = list(range(i, min(i + window_size, n_samples)))
            if len(window_idx) < 3:  # Skip too small windows
                continue
                
            window_indices.append(window_idx)
        
        if not window_indices:
            print("No valid windows found. Try reducing window_size or increasing overlap.")
            return None
        
        print(f"Created {len(window_indices)} windows with size ≈{window_size} and overlap ≈{overlap}")
        
        # Initialize zigzag persistence results
        zigzag_results = {
            'windows': window_indices,
            'persistence_diagrams': [],
            'betti_numbers': {},
            'window_transitions': []
        }
        
        # For each window, compute persistent homology
        for i, window_idx in enumerate(window_indices):
            print(f"Processing window {i+1}/{len(window_indices)}...")
            
            # Get window data
            window_features = self.features[window_idx]
            window_times = self.timestamps[window_idx]
            
            # Create distance matrix for this window
            window_dist = self.temporal_distance_matrix[np.ix_(window_idx, window_idx)]
            
            # Build filtration for this window
            st = SimplexTree()
            
            # Add vertices
            for j in range(len(window_idx)):
                st.insert([j], filtration=0.0)
            
            # Add edges
            for j in range(len(window_idx)):
                for k in range(j+1, len(window_idx)):  # Enforce temporal ordering 
                    if window_dist[j, k] <= 1.0:  # Threshold
                        filtration_value = window_dist[j, k]
                        st.insert([j, k], filtration=filtration_value)
            
            # Expand to higher dimensions
            st.expansion(max_dimension)
            
            # Compute persistence
            st.compute_persistence(min_persistence=min_persistence)
            
            # Get persistence diagram
            persistence = st.persistence()
            
            # Get simplices with their filtration values
            simplices_with_filtration = list(st.get_filtration())
            
            # Create a mapping from simplex to its filtration value
            simplex_to_filtration = {tuple(simplex): filt for simplex, filt in simplices_with_filtration}
            
            # Store results with original indices
            window_persistence = []
            for dim_pair in persistence:
                dim = dim_pair[0]  # dimension
                birth = dim_pair[1][0]  # birth time
                death = dim_pair[1][1]  # death time
                
                # Find corresponding simplex for this persistence pair
                # We'll store the first simplex we find with matching dimension and filtration value
                matching_simplex = None
                for simplex, filt in simplices_with_filtration:
                    if len(simplex) == dim + 1 and abs(filt - birth) < 1e-10:
                        matching_simplex = simplex
                        break
                
                if matching_simplex is not None:
                    # Map local indices to global indices
                    global_simplex = [window_idx[s] for s in matching_simplex]
                    window_persistence.append((dim, birth, death, global_simplex))
                else:
                    # If no matching simplex found, store without simplex information
                    window_persistence.append((dim, birth, death, []))
            
            zigzag_results['persistence_diagrams'].append(window_persistence)
            
            # Compute Betti numbers for each dimension
            for dim in range(max_dimension + 1):
                betti_key = f'betti_{dim}'
                if betti_key not in zigzag_results['betti_numbers']:
                    zigzag_results['betti_numbers'][betti_key] = []
                
                # Count persistence pairs with birth <= 1.0 and death > 1.0
                betti = len([p for p in persistence if p[0] == dim and p[1][0] <= 1.0 and p[1][1] > 1.0])
                zigzag_results['betti_numbers'][betti_key].append(betti)
        
        # Compute window transitions
        for i in range(len(window_indices) - 1):
            current_window = set(window_indices[i])
            next_window = set(window_indices[i+1])
            
            # Find overlap indices
            overlap_indices = current_window.intersection(next_window)
            
            if not overlap_indices:
                continue
            
            # Store transition information
            transition = {
                'window_pair': (i, i+1),
                'overlap_size': len(overlap_indices),
                'overlap_indices': list(overlap_indices),
                'persistence_comparison': self._compare_window_persistence(
                    zigzag_results['persistence_diagrams'][i],
                    zigzag_results['persistence_diagrams'][i+1],
                    overlap_indices
                )
            }
            
            zigzag_results['window_transitions'].append(transition)
        
        # Store zigzag results
        self.path_zigzag_diagrams = zigzag_results
        
        print(f"Zigzag persistence computation completed in {time.time() - start_time:.2f} seconds")
        
        return zigzag_results
    
    def _compare_window_persistence(self, pers1, pers2, overlap_indices):
        """
        Compare persistence diagrams between consecutive windows.
        
        Parameters:
        -----------
        pers1, pers2 : list
            Persistence diagrams for consecutive windows
        overlap_indices : set
            Indices in the overlap region
            
        Returns:
        --------
        dict
            Comparison metrics
        """
        overlap_set = set(overlap_indices)
        
        # Count features in the overlap region
        overlap_features = {0: 0, 1: 0, 2: 0}
        
        # For the first window
        for entry in pers1:
            # Each entry is (dim, birth, death, simplex)
            dim = entry[0]
            simplex = entry[3]  # Get the simplex from the fourth element
            
            # Check if the simplex has any vertex in the overlap
            if any(idx in overlap_set for idx in simplex):
                if dim in overlap_features:
                    overlap_features[dim] += 1
        
        # Calculate stability metrics (simplified Wasserstein distance)
        # We only compare features in the overlap region
        stability = {}
        
        for dim in range(3):  # Dimensions 0, 1, 2
            # Extract persistence pairs for this dimension from both windows
            # Each entry is (dim, birth, death, simplex)
            pairs1 = [(entry[1], entry[2]) for entry in pers1 if entry[0] == dim]
            pairs2 = [(entry[1], entry[2]) for entry in pers2 if entry[0] == dim]
            
            # Simple stability metric: difference in number of features
            count_diff = abs(len(pairs1) - len(pairs2))
            
            stability[f'dim_{dim}_count_diff'] = count_diff
            stability[f'dim_{dim}_overlap_count'] = overlap_features.get(dim, 0)
        
        return stability 