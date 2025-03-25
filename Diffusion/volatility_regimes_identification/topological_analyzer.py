import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import warnings
import networkx as nx
import gudhi
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import itertools
import gudhi.point_cloud
import gudhi.representations
import gudhi.persistence_graphical_tools as gd_plot
from gudhi.rips_complex import RipsComplex
from gudhi.weighted_rips_complex import WeightedRipsComplex
from gudhi.persistence_graphical_tools import plot_persistence_barcode
from gudhi.persistence_graphical_tools import plot_persistence_diagram

# Suppress warnings
warnings.filterwarnings('ignore')

class TopologicalDataAnalyzer:
    """
    Implements the topological data analysis components for volatility regime detection.
    Uses GUDHI for proper persistent homology computation and witness complexes.
    
    Implements Sections 3-5 of the TDA Pipeline:
    - Temporally-Aware Simplicial Complex Construction
    - Persistent Path Homology Calculation
    - Regime Identification with Information-Theoretic Temporal Mapping
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
        self.persistence_diagrams = None
        self.regime_labels = None
        self.mapper_graph = None
        self.witness_complex = None
        self.multi_persistence_diagrams = None
        
        # New attributes for Zigzag and Path Homology
        self.zigzag_diagrams = None
        self.path_homology_diagrams = None
        self.flow_complex = None
        self.state_transitions = None
        
    def compute_temporally_weighted_distance(self, alpha=0.5, beta=0.1, lambda_info=1.0, mi_matrix=None):
        """
        Compute temporally-weighted distance matrix between data points.
        
        Parameters:
            alpha (float): Weight for temporal component
            beta (float): Decay rate for temporal distance
            lambda_info (float): Weight for mutual information component
            mi_matrix (np.ndarray): Pre-computed mutual information matrix
        """
        start_time = time.time()
        print("Computing temporally-weighted distance matrix...")
        
        # Convert timestamps to numerical values (seconds since first timestamp)
        if np.issubdtype(self.timestamps.dtype, np.datetime64):
            # Convert to nanoseconds, then to seconds
            t0 = self.timestamps[0]
            time_diffs = self.timestamps - t0
            time_indices = time_diffs.astype('timedelta64[ns]').astype(np.float64) / 1e9
        else:
            time_indices = np.array(self.timestamps)
        
        # Normalize time indices to [0,1] range
        if len(time_indices) > 1:
            time_indices = (time_indices - time_indices.min()) / (time_indices.max() - time_indices.min())
        
        n_samples = len(self.features)
        dist_matrix = np.zeros((n_samples, n_samples))
        
        # Compute Euclidean distances
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                # Base distance (Euclidean)
                dist = np.linalg.norm(self.features[i] - self.features[j])
                
                # Temporal weighting
                time_dist = abs(time_indices[i] - time_indices[j])
                temporal_weight = 1 + alpha * (1 - np.exp(-beta * time_dist))
                
                # Information weighting (if MI matrix provided)
                if mi_matrix is not None and lambda_info > 0:
                    # Ensure indices are within bounds
                    if i < mi_matrix.shape[0] and j < mi_matrix.shape[1]:
                        mi = mi_matrix[i, j]
                        info_weight = np.exp(-lambda_info * mi)
                    else:
                        info_weight = 1.0
                else:
                    info_weight = 1.0
                    
                # Combined distance
                dist_matrix[i, j] = dist * temporal_weight * info_weight
                dist_matrix[j, i] = dist_matrix[i, j]  # Symmetric matrix
        
        self.distance_matrix = dist_matrix
        print(f"Distance matrix computation completed in {time.time() - start_time:.2f} seconds")
        return dist_matrix
    
    def construct_directed_network(self, epsilon=0.5, enforce_temporal=True):
        """
        Construct a directed weighted network based on the temporal distance matrix.
        In other words, Construction of Proximity Graph
        
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
            
        # Add edges based on distance threshold and temporal ordering
        for i in range(self.n_samples):
            for j in range(self.n_samples):
                if i == j:
                    continue
                    
                # Enforce temporal ordering if requested
                if enforce_temporal and i >= j:
                    continue
                    
                # Check if distance is below threshold
                if self.temporal_distance_matrix[i, j] <= epsilon:
                    # Edge weight is inverse of distance
                    weight = 1.0 / max(self.temporal_distance_matrix[i, j], 1e-10)
                    G.add_edge(i, j, weight=weight, distance=self.temporal_distance_matrix[i, j])
        
        return G
    
    def compute_path_complex(self, G, max_path_length=2):
        """
        Generate a directed path complex from the graph.
        
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
        print(f"Computing path complex with max path length {max_path_length}...")
        
        # Initialize path complex
        path_complex = {k: [] for k in range(max_path_length + 1)}
        
        # Add 0-simplices (vertices)
        for node in G.nodes():
            path_complex[0].append((node,))
            
        # Add 1-simplices (edges)
        for u, v in G.edges():
            path_complex[1].append((u, v))
            
        # Add higher-order paths
        for k in range(2, max_path_length + 1):
            for path in path_complex[k-1]:
                # Last vertex in the path
                last = path[-1]
                
                # Try to extend the path
                for neighbor in G.successors(last):
                    # Avoid cycles
                    if neighbor not in path:
                        new_path = path + (neighbor,)
                        path_complex[k].append(new_path)
                        
        return path_complex
    
    def compute_persistent_homology(self, min_epsilon=0.1, max_epsilon=2.0, num_steps=10, use_weighted=False, output_dir=None):
        """
        Compute persistent homology using GUDHI's Rips complex with optimized computation.
        
        Parameters:
        -----------
        min_epsilon : float
            Minimum distance threshold
        max_epsilon : float
            Maximum distance threshold
        num_steps : int
            Number of filtration steps
        use_weighted : bool
            Whether to use weighted Rips complex (considers temporal weights)
        output_dir : str
            Directory to save output files and plots
            
        Returns:
        --------
        dict
            Dictionary containing persistence diagrams and related information
        """
        start_time = time.time()
        print("Computing persistent homology using GUDHI...")
        
        try:
            # Choose appropriate complex based on whether we want weighted computation
            if use_weighted and self.temporal_distance_matrix is not None:
                print("Using weighted Rips complex...")
                rips_complex = WeightedRipsComplex(
                    distance_matrix=self.temporal_distance_matrix,
                    max_edge_length=max_epsilon
                )
            else:
                print("Using standard Rips complex...")
                # Ensure features are properly scaled
                features_scaled = self.features.copy()
                if len(features_scaled) > 1:
                    features_scaled = (features_scaled - features_scaled.min(axis=0)) / (features_scaled.max(axis=0) - features_scaled.min(axis=0))
                
                rips_complex = RipsComplex(
                    points=features_scaled,
                    max_edge_length=max_epsilon
                )
            
            # Create simplex tree
            print("Creating simplex tree...")
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
            
            print(f"Simplex tree has {simplex_tree.num_simplices()} simplices")
            print(f"Dimension is {simplex_tree.dimension()}")
            
            # Compute persistence
            print("Computing persistence...")
            persistence = simplex_tree.persistence()
            
            if not persistence:
                raise ValueError("No persistence pairs found")
                
            print(f"Found {len(persistence)} persistence pairs")
            
            # Initialize persistence diagram storage
            persistence_diagrams = {}
            
            # Process each dimension
            for dim in range(3):
                # Initialize dimension results
                persistence_diagrams[dim] = {
                    'full_diagram': [],
                    'persistent_features': [],
                    'betti_numbers': [],
                    'birth_times': [],
                    'death_times': [],
                    'persistence_pairs': []
                }
                
                # Extract dimension-specific diagrams
                dim_pairs = []
                for (dimension, (birth, death)) in persistence:
                    if dimension == dim:
                        if death == float('inf'):
                            death = max_epsilon
                        dim_pairs.append((birth, death))
                
                if dim_pairs:
                    # Convert to numpy array for easier manipulation
                    dim_pairs = np.array(dim_pairs)
                    
                    # Sort by persistence (death - birth)
                    persistence_values = dim_pairs[:, 1] - dim_pairs[:, 0]
                    sort_idx = np.argsort(-persistence_values)  # Sort in descending order
                    dim_pairs = dim_pairs[sort_idx]
                    
                    # Store results
                    persistence_diagrams[dim]['full_diagram'] = dim_pairs
                    persistence_diagrams[dim]['birth_times'] = dim_pairs[:, 0]
                    persistence_diagrams[dim]['death_times'] = dim_pairs[:, 1]
                    persistence_diagrams[dim]['persistence_pairs'] = list(map(tuple, dim_pairs))
                    
                    # Compute Betti numbers across filtration
                    epsilon_values = np.linspace(min_epsilon, max_epsilon, num_steps)
                    betti_numbers = []
                    
                    for eps in epsilon_values:
                        alive = np.sum((dim_pairs[:, 0] <= eps) & (dim_pairs[:, 1] > eps))
                        betti_numbers.append(alive)
                        
                    persistence_diagrams[dim]['betti_numbers'] = np.array(betti_numbers)
                    
                    # Store infinite persistence features
                    persistence_diagrams[dim]['persistent_features'] = dim_pairs[dim_pairs[:, 1] == max_epsilon]
            
            # Store results
            self.persistence_diagrams = persistence_diagrams
            
            # Generate visualizations
            try:
                if output_dir is None:
                    output_dir = '.'
                    
                print("Generating visualizations...")
                
                # Plot persistence diagram
                plt.figure(figsize=(10, 10))
                colors = ['blue', 'red', 'green']
                for dim in range(3):
                    if dim in persistence_diagrams and len(persistence_diagrams[dim]['full_diagram']) > 0:
                        pairs = persistence_diagrams[dim]['full_diagram']
                        plt.scatter(pairs[:, 0], pairs[:, 1], c=colors[dim], 
                                  label=f'H{dim}', alpha=0.6)
                
                # Add diagonal line
                diag_min = min_epsilon
                diag_max = max_epsilon
                plt.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', alpha=0.5)
                
                plt.xlabel('Birth')
                plt.ylabel('Death')
                plt.title('Persistence Diagram')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, 'persistence_diagram.png'))
                plt.close()
                
                # Plot persistence barcode
                plt.figure(figsize=(12, 6))
                y_offset = 0
                for dim in range(3):
                    if dim in persistence_diagrams and len(persistence_diagrams[dim]['full_diagram']) > 0:
                        pairs = persistence_diagrams[dim]['full_diagram']
                        for i, (birth, death) in enumerate(pairs):
                            plt.plot([birth, death], [y_offset + i, y_offset + i], 
                                   c=colors[dim], linewidth=1.5, label=f'H{dim}' if i == 0 else "")
                        y_offset += len(pairs) + 1
                
                plt.xlabel('Epsilon')
                plt.ylabel('Homology Class')
                plt.title('Persistence Barcode')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, 'persistence_barcode.png'))
                plt.close()
                
            except Exception as viz_error:
                print(f"Warning: Could not generate visualizations: {str(viz_error)}")
            
            print(f"Persistent homology computation completed in {time.time() - start_time:.2f} seconds")
            return persistence_diagrams
            
        except Exception as e:
            print(f"Error in persistent homology computation: {str(e)}")
            print("Attempting to proceed with simplified computation...")
            
            try:
                # Fallback to simpler computation
                features_scaled = self.features.copy()
                if len(features_scaled) > 1:
                    features_scaled = (features_scaled - features_scaled.min(axis=0)) / (features_scaled.max(axis=0) - features_scaled.min(axis=0))
                
                rips_complex = RipsComplex(
                    points=features_scaled,
                    max_edge_length=max_epsilon
                )
                
                simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)  # Reduce dimension for simplicity
                persistence = simplex_tree.persistence()
                
                # Create basic persistence diagram
                persistence_diagrams = {
                    dim: {
                        'full_diagram': [],
                        'betti_numbers': []
                    }
                    for dim in range(2)  # Reduce to 2 dimensions
                }
                
                # Process persistence pairs
                for (dimension, (birth, death)) in persistence:
                    if dimension < 2:  # Only process dimensions 0 and 1
                        if death == float('inf'):
                            death = max_epsilon
                        persistence_diagrams[dimension]['full_diagram'].append((birth, death))
                
                # Convert to numpy arrays
                for dim in persistence_diagrams:
                    if persistence_diagrams[dim]['full_diagram']:
                        persistence_diagrams[dim]['full_diagram'] = np.array(persistence_diagrams[dim]['full_diagram'])
                    else:
                        persistence_diagrams[dim]['full_diagram'] = np.array([])
                
                self.persistence_diagrams = persistence_diagrams
                return persistence_diagrams
                
            except Exception as fallback_error:
                print(f"Error in simplified computation: {str(fallback_error)}")
                # Return empty persistence diagrams as last resort
                return {dim: {'full_diagram': np.array([]), 'betti_numbers': []} for dim in range(3)}

    def compute_multi_parameter_persistence(self, parameters=None, max_dimension=2):
        """
        Compute multi-parameter persistent homology using GUDHI's advanced capabilities.
        
        Parameters:
        -----------
        parameters : list of tuples
            List of (name, values) pairs for filtration parameters
        max_dimension : int
            Maximum homology dimension to compute
            
        Returns:
        --------
        dict
            Dictionary containing multi-parameter persistence results
        """
        print("Computing multi-parameter persistence...")
        
        if parameters is None:
            parameters = [
                ('time', self.timestamps),
                ('volatility', np.std(self.features, axis=1)),
                ('volume', np.sum(self.features, axis=1))
            ]
            
        # Normalize parameter values to [0,1]
        normalized_params = []
        for name, values in parameters:
            if len(values) > 1:
                values = (values - values.min()) / (values.max() - values.min())
            normalized_params.append((name, values))
            
        # Create multi-filtration
        multi_filtration = []
        n_samples = len(self.features)
        
        for i in range(n_samples):
            # Get parameter values for this point
            point_params = [values[i] for _, values in normalized_params]
            multi_filtration.append(point_params)
            
        # Convert to numpy array
        multi_filtration = np.array(multi_filtration)
        
        # Create multi-parameter Rips complex
        rips = gudhi.RipsComplex(
            points=self.features,
            max_edge_length=np.inf
        )
        
        # Create simplex tree with multi-parameter filtration
        st = rips.create_simplex_tree(max_dimension=max_dimension)
        
        # Assign multi-parameter filtration values
        for simplex, _ in st.get_simplices():
            if len(simplex) == 1:  # vertex
                idx = simplex[0]
                st.assign_filtration(simplex, multi_filtration[idx])
            else:  # higher dimensional simplex
                # Use maximum of vertex filtration values
                vertex_values = [multi_filtration[v] for v in simplex]
                filtration_value = np.max(vertex_values, axis=0)
                st.assign_filtration(simplex, filtration_value)
        
        # Compute multi-parameter persistence
        st.make_filtration_non_decreasing()
        multi_diagrams = st.persistence()
        
        # Process results
        results = {
            'diagrams': {},
            'parameter_names': [p[0] for p in parameters],
            'parameter_ranges': [(values.min(), values.max()) for _, values in parameters],
            'multi_filtration': multi_filtration
        }
        
        # Process persistence diagrams by dimension
        for dim in range(max_dimension + 1):
            dim_results = {
                'birth_values': [],
                'death_values': [],
                'persistence_pairs': [],
                'persistence_values': []
            }
            
            # Extract dimension-specific diagrams
            dim_diagrams = [(p[1], p[2]) for p in multi_diagrams if p[0] == dim]
            
            for birth, death in dim_diagrams:
                if death != float('inf'):
                    dim_results['birth_values'].append(birth)
                    dim_results['death_values'].append(death)
                    dim_results['persistence_pairs'].append((birth, death))
                    dim_results['persistence_values'].append(
                        np.array([d - b for b, d in zip(birth, death)])
                    )
            
            results['diagrams'][dim] = dim_results
        
        # Generate visualizations for 2D projections
        if not os.path.exists('multi_persistence_plots'):
            os.makedirs('multi_persistence_plots')
            
        # Plot 2D projections of the persistence diagram
        for i, j in itertools.combinations(range(len(parameters)), 2):
            param1, param2 = parameters[i][0], parameters[j][0]
            
            plt.figure(figsize=(10, 10))
            for dim in range(max_dimension + 1):
                if dim in results['diagrams']:
                    birth_values = results['diagrams'][dim]['birth_values']
                    death_values = results['diagrams'][dim]['death_values']
                    
                    if birth_values:  # Check if we have any values to plot
                        birth_proj = np.array(birth_values)[:, [i, j]]
                        death_proj = np.array(death_values)[:, [i, j]]
                        
                        plt.scatter(
                            birth_proj[:, 0],
                            death_proj[:, 0],
                            label=f'H{dim}',
                            alpha=0.6
                        )
            
            plt.xlabel(param1)
            plt.ylabel(param2)
            plt.title(f'Multi-parameter Persistence: {param1} vs {param2}')
            plt.legend()
            plt.savefig(f'multi_persistence_plots/persistence_{param1}_{param2}.png')
            plt.close()
        
        self.multi_persistence_diagrams = results
        return results

    def create_temporal_mapper(self, n_intervals=10, percent_overlap=50, filter_functions=None):
        """
        Create a mapper graph using temporal information.
        
        Parameters:
        -----------
        n_intervals : int
            Number of intervals for the filter function
        percent_overlap : float
            Percentage of overlap between intervals
        filter_functions : list
            List of additional filter functions to use
        """
        print("Creating temporal mapper graph...")
        
        # Convert timestamps to numerical values
        if np.issubdtype(self.timestamps.dtype, np.datetime64):
            # Convert to nanoseconds, then to seconds
            t0 = self.timestamps[0]
            time_diffs = self.timestamps - t0
            values = time_diffs.astype('timedelta64[ns]').astype(np.float64) / 1e9
        else:
            values = np.array(self.timestamps)
        
        # Normalize time values to [0,1]
        if len(values) > 1:
            values = (values - values.min()) / (values.max() - values.min())
        
        # Create filter function list
        if filter_functions is None:
            filter_functions = []
        
        # Add temporal filter
        filter_functions.append(values)
        
        # Create intervals
        overlap = percent_overlap / 100.0
        interval_size = 1.0 / (n_intervals - (n_intervals - 1) * overlap)
        
        # Create mapper graph
        self.mapper_graph = nx.Graph()
        
        # Create cover
        intervals = []
        
        for i in range(len(filter_functions)):
            # Create overlapping intervals
            function_intervals = []
            for j in range(n_intervals):
                start = j * interval_size - (j > 0) * overlap
                end = (j + 1) * interval_size + (j < n_intervals - 1) * overlap
                function_intervals.append((start, end))
                
            intervals.append(function_intervals)
            
        # Create cover elements (intersection of intervals from each filter function)
        cover_elements = {}
        cover_index = 0
        
        for interval_combo in itertools.product(*intervals):
            points_in_cover = []
            
            # Find points that fall in all intervals
            for i in range(self.n_samples):
                in_all_intervals = True
                
                for j, (start, end) in enumerate(interval_combo):
                    if not (start <= filter_functions[j][i] <= end):
                        in_all_intervals = False
                        break
                        
                if in_all_intervals:
                    points_in_cover.append(i)
                    
            if points_in_cover:
                cover_elements[cover_index] = points_in_cover
                cover_index += 1
                
        # Add nodes (cover elements)
        for node_id, points in cover_elements.items():
            # Node attributes: mean feature values, earliest timestamp
            mean_features = np.mean(self.features[points], axis=0)
            timestamps = self.timestamps[points]
            earliest_time = np.min(timestamps) if isinstance(timestamps[0], (int, float)) else min(timestamps)
            
            self.mapper_graph.add_node(
                node_id, 
                points=points, 
                size=len(points),
                features=mean_features,
                timestamp=earliest_time
            )
            
        # Add edges between nodes that share points
        for node1, node2 in itertools.combinations(cover_elements.keys(), 2):
            shared_points = set(cover_elements[node1]).intersection(set(cover_elements[node2]))
            
            if shared_points:
                self.mapper_graph.add_edge(
                    node1, 
                    node2, 
                    weight=len(shared_points),
                    shared_points=list(shared_points)
                )
                
        return self.mapper_graph
    
    def identify_regimes(self, n_regimes=3, affinity_sigma=1.0):
        """
        Identify volatility regimes using directed spectral clustering, 
        incorporating path zigzag persistence information when available.
        
        Parameters:
        -----------
        n_regimes : int
            Number of regimes to identify
        affinity_sigma : float
            Bandwidth parameter for affinity calculation
            
        Returns:
        --------
        numpy.ndarray
            Array of regime labels for each data point
        """
        start_time = time.time()
        print("Identifying volatility regimes...")
        
        if self.temporal_distance_matrix is None:
            raise ValueError("Temporal distance matrix must be computed first")
        
        # Normalize features to improve clustering
        normalized_features = StandardScaler().fit_transform(self.features)
        
        # Create affinity matrix from distance matrix
        affinity = np.zeros_like(self.temporal_distance_matrix)
        
        # Check if we have path zigzag persistence data
        has_zigzag_data = hasattr(self, 'path_zigzag_diagrams') and self.path_zigzag_diagrams is not None
        
        # Initialize temporal coherence matrix - encourages temporal consistency in regimes
        temporal_coherence = np.zeros_like(self.temporal_distance_matrix)
        half_window = 5  # Consider points within this window as temporally related
        
        # Build temporal coherence - higher affinity for nearby points in time
        for i in range(self.n_samples):
            lower_bound = max(0, i - half_window)
            upper_bound = min(self.n_samples, i + half_window + 1)
            for j in range(lower_bound, upper_bound):
                if i != j:
                    # Stronger coherence for closer points in time
                    temporal_coherence[i, j] = 1.0 - abs(i - j) / (2 * half_window)
        
        if has_zigzag_data:
            print("Using path zigzag persistence information for regime identification...")
            
            # Get window indices from zigzag computation
            window_indices = self.path_zigzag_diagrams['window_indices']
            
            # Get transition information
            transitions = self.path_zigzag_diagrams['betti_numbers']['transitions']
            
            # Create a transition strength matrix based on zigzag persistence
            transition_strength = np.zeros((self.n_samples, self.n_samples))
            
            # Fill transition strength matrix based on persistent paths
            for transition in transitions:
                window_pair = transition['window_pair']
                persistent_paths = transition['persistent_paths']
                dimension = transition['dimension']
                
                # More weight to higher-dimensional persistent paths
                strength = persistent_paths * (dimension + 1)  # Increase influence
                
                # Get indices for the overlapping part of consecutive windows
                current_window = window_indices[window_pair[0]]
                next_window = window_indices[window_pair[1]]
                overlap_indices = set(current_window).intersection(set(next_window))
                
                # Set transition strength for points in the overlap
                for i in overlap_indices:
                    for j in overlap_indices:
                        if i < j:  # Respect temporal ordering
                            transition_strength[i, j] += strength
            
            # Normalize transition strength
            if np.max(transition_strength) > 0:
                transition_strength = transition_strength / np.max(transition_strength)
            
            # Incorporate transition strength into affinity calculation
            for i in range(self.n_samples):
                for j in range(self.n_samples):
                    if i != j:
                        # Calculate feature distance using normalized features
                        feature_dist = np.sqrt(np.sum((normalized_features[i] - normalized_features[j]) ** 2))
                        
                        # Calculate modified affinity with zigzag information
                        base_affinity = np.exp(-feature_dist**2 / (2 * affinity_sigma**2))
                        temporal_effect = 1.0 + temporal_coherence[i, j]
                        
                        # Boost affinity for strong persistent path transitions
                        zigzag_boost = 1.0 + transition_strength[i, j] * 2.0
                        
                        # Calculate final affinity
                        affinity[i, j] = base_affinity * zigzag_boost * temporal_effect
                        
                        # Penalize large temporal gaps (non-consecutive regimes)
                        if abs(i - j) > half_window * 3:
                            affinity[i, j] *= 0.5
                            
        else:
            # Standard affinity computation (no zigzag information)
            for i in range(self.n_samples):
                for j in range(self.n_samples):
                    if i != j:
                        # Calculate feature distance using normalized features
                        feature_dist = np.sqrt(np.sum((normalized_features[i] - normalized_features[j]) ** 2))
                        
                        # Calculate modified affinity with temporal weighting
                        base_affinity = np.exp(-feature_dist**2 / (2 * affinity_sigma**2))
                        temporal_effect = 1.0 + temporal_coherence[i, j]
                        
                        # Calculate final affinity
                        affinity[i, j] = base_affinity * temporal_effect
                        
                        # Penalize large temporal gaps (non-consecutive regimes)
                        if abs(i - j) > half_window * 3:
                            affinity[i, j] *= 0.5
        
        # Make sure affinity matrix is valid for spectral clustering
        np.fill_diagonal(affinity, 0)  # Zero diagonal
        
        # Print some stats about the affinity matrix
        print(f"Affinity matrix shape: {affinity.shape}, min: {np.min(affinity)}, max: {np.max(affinity)}")
        
        # Perform spectral clustering with additional parameters for robustness
        from sklearn.cluster import SpectralClustering
        
        # Try with different algorithms for stability
        try:
            # First attempt with 'amg' solver which is faster and more accurate
            clustering = SpectralClustering(
                n_clusters=n_regimes,
                affinity='precomputed',
                random_state=42,
                assign_labels='kmeans',
                eigen_solver='amg'  # Algebraic multigrid solver
            )
            regime_labels = clustering.fit_predict(affinity)
        except:
            # Fall back to default solver if amg fails
            print("AMG solver failed, using default solver...")
            clustering = SpectralClustering(
                n_clusters=n_regimes,
                affinity='precomputed',
                random_state=42,
                assign_labels='kmeans'
            )
            regime_labels = clustering.fit_predict(affinity)
        
        # Check if regime distribution is reasonable
        regime_counts = np.bincount(regime_labels)
        min_regime_size = np.min(regime_counts)
        total_points = len(regime_labels)
        
        print(f"Initial regime distribution: {regime_counts}")
        
        # If any regime is too small, try to re-cluster
        if min_regime_size < total_points * 0.05 and total_points > 20:
            print(f"Regime imbalance detected, smallest regime has only {min_regime_size} points. Trying again with adjusted parameters...")
            
            # Adjust affinity to promote more balanced clusters
            for i in range(self.n_samples):
                for j in range(self.n_samples):
                    if i != j:
                        # Add more weight to temporal coherence
                        affinity[i, j] = affinity[i, j] * (1.0 + temporal_coherence[i, j] * 2.0)
            
            # Try again with slightly different parameters
            clustering = SpectralClustering(
                n_clusters=n_regimes,
                affinity='precomputed',
                random_state=42,
                n_init=10,  # Try multiple initializations
                assign_labels='discretize'  # Different label assignment
            )
            new_regime_labels = clustering.fit_predict(affinity)
            
            # Check if the new labels are better balanced
            new_regime_counts = np.bincount(new_regime_labels)
            new_min_regime_size = np.min(new_regime_counts)
            
            print(f"New regime distribution: {new_regime_counts}")
            
            # Use the new labels if they're better balanced
            if new_min_regime_size > min_regime_size:
                regime_labels = new_regime_labels
        
        # Store regime labels
        self.regime_labels = regime_labels
        
        # Print regime distribution
        final_regime_counts = np.bincount(regime_labels)
        print(f"Final regime distribution: {final_regime_counts}")
        
        # Print runtime
        print(f"Regime identification completed in {time.time() - start_time:.2f} seconds")
        return regime_labels
    
    def analyze_regimes(self):
        """
        Analyze the identified regimes to extract statistics and patterns.
        
        Returns:
        --------
        dict
            Dictionary containing regime analysis results
        """
        start_time = time.time()
        print("Analyzing volatility regimes...")
        
        if self.regime_labels is None:
            raise ValueError("Must detect regimes before analyzing them")
        
        # Get the unique regime labels, which may not be sequential from 0
        unique_regimes = np.unique(self.regime_labels)
        n_regimes = len(unique_regimes)
        
        # Create a mapping from regime labels to sequential indices
        regime_to_idx = {regime: idx for idx, regime in enumerate(unique_regimes)}
        idx_to_regime = {idx: regime for idx, regime in enumerate(unique_regimes)}
        
        regime_stats = []
        
        # Initialize transition probability matrix with the correct size
        transition_probs = np.zeros((n_regimes, n_regimes))
        
        # Compute transitions using the mapping
        for i in range(len(self.regime_labels) - 1):
            current_regime = self.regime_labels[i]
            next_regime = self.regime_labels[i + 1]
            current_idx = regime_to_idx[current_regime]
            next_idx = regime_to_idx[next_regime]
            transition_probs[current_idx, next_idx] += 1
        
        # Normalize transition probabilities
        row_sums = transition_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probs = transition_probs / row_sums
        
        # Analyze each regime
        for regime_idx, regime in enumerate(unique_regimes):
            regime_mask = self.regime_labels == regime
            regime_points = np.where(regime_mask)[0]
            regime_times = self.timestamps[regime_points] if hasattr(self, 'timestamps') and len(regime_points) > 0 else None
            
            if len(regime_points) > 0:
                # Convert numpy.timedelta64 to pandas Timedelta for duration calculation
                if regime_times is not None and np.issubdtype(regime_times.dtype, np.datetime64):
                    start_time_val = pd.Timestamp(min(regime_times))
                    end_time_val = pd.Timestamp(max(regime_times))
                    duration = (end_time_val - start_time_val).total_seconds()
                else:
                    # If not datetime, use number of points as duration
                    duration = len(regime_points)
                    start_time_val = regime_points[0] if len(regime_points) > 0 else 0
                    end_time_val = regime_points[-1] if len(regime_points) > 0 else 0
                
                # Calculate regime statistics
                stats = {
                    'regime_id': int(regime),  # Convert to int to avoid issues with numpy types
                    'size': len(regime_points),
                    'start_time': start_time_val,
                    'end_time': end_time_val,
                    'duration': duration,
                    'points': regime_points.tolist()
                }
                
                regime_stats.append(stats)
        
        result = {
            'regime_labels': self.regime_labels,
            'regime_stats': regime_stats,
            'transition_probs': transition_probs,
            'n_regimes': n_regimes,
            'unique_regimes': unique_regimes.tolist(),
            'regime_to_idx': regime_to_idx,
            'idx_to_regime': idx_to_regime
        }
        
        print(f"Regime analysis completed in {time.time() - start_time:.2f} seconds")
        return result

    def compute_zigzag_persistence(self, window_size=100, overlap=50):
        """
        Compute zigzag persistence to capture market state transitions.
        
        Parameters:
        -----------
        window_size : int
            Size of sliding window for state computation
        overlap : int
            Number of points to overlap between windows
            
        Returns:
        --------
        dict
            Dictionary containing zigzag persistence diagrams
        """
        print("Computing zigzag persistence for market states...")
        
        try:
            # Create sequence of complexes
            complexes = []
            for i in range(0, self.n_samples - window_size + 1, window_size - overlap):
                # Get window of data
                window_data = self.features[i:i + window_size]
                
                # Create Rips complex for this window
                rips = gudhi.RipsComplex(
                    points=window_data,
                    max_edge_length=np.inf
                )
                
                # Create simplex tree
                st = rips.create_simplex_tree(max_dimension=2)
                
                # Add to sequence
                complexes.append(st)
            
            # Create zigzag complex
            zz = gudhi.ZigzagPersistence()
            
            # Add complexes to zigzag
            for i, st in enumerate(complexes):
                zz.add_complex(st)
                
                # Add inclusion maps between overlapping windows
                if i > 0:
                    overlap_points = list(range(overlap))
                    zz.add_inclusion(complexes[i-1], st, overlap_points)
            
            # Compute zigzag persistence
            zz_diagrams = zz.compute_persistence()
            
            # Process results
            self.zigzag_diagrams = {
                'diagrams': zz_diagrams,
                'complexes': complexes,
                'window_size': window_size,
                'overlap': overlap
            }
            
            # Generate visualizations
            if not os.path.exists('zigzag_plots'):
                os.makedirs('zigzag_plots')
                
            plt.figure(figsize=(12, 6))
            for dim in range(3):
                if dim in zz_diagrams:
                    pairs = zz_diagrams[dim]
                    for birth, death in pairs:
                        plt.plot([birth, death], [dim, dim], 
                               c=['blue', 'red', 'green'][dim],
                               linewidth=2, label=f'H{dim}' if birth == pairs[0][0] else "")
            
            plt.xlabel('Time')
            plt.ylabel('Dimension')
            plt.title('Zigzag Persistence Diagram')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('zigzag_plots/zigzag_persistence.png')
            plt.close()
            
            return self.zigzag_diagrams
            
        except Exception as e:
            print(f"Error in zigzag persistence computation: {str(e)}")
            return None

    def compute_persistent_path_homology(self, max_path_length=3):
        """
        Compute persistent path homology to capture order flow dynamics.
        
        Parameters:
        -----------
        max_path_length : int
            Maximum length of paths to consider
            
        Returns:
        --------
        dict
            Dictionary containing path homology diagrams
        """
        print("Computing persistent path homology for order flow...")
        
        try:
            # Create directed network from temporal distance matrix
            G = self.construct_directed_network(epsilon=0.5)
            
            # Create path complex
            path_complex = self.compute_path_complex(G, max_path_length)
            
            # Create flow complex (directed simplicial complex)
            self.flow_complex = nx.DiGraph()
            
            # Add vertices
            for node in G.nodes():
                self.flow_complex.add_node(node)
            
            # Add edges with flow direction
            for u, v in G.edges():
                # Determine flow direction based on price movement
                price_diff = self.features[v, 0] - self.features[u, 0]  # Assuming first feature is price
                if price_diff > 0:
                    self.flow_complex.add_edge(u, v, direction='up')
                else:
                    self.flow_complex.add_edge(v, u, direction='down')
            
            # Compute path homology
            path_homology = {}
            
            for dim in range(max_path_length + 1):
                # Get paths of current dimension
                paths = path_complex[dim]
                
                # Create boundary matrix
                n_paths = len(paths)
                if n_paths == 0:
                    continue
                    
                boundary_matrix = np.zeros((n_paths, n_paths))
                
                # Fill boundary matrix
                for i, path in enumerate(paths):
                    # Get boundary paths
                    boundary_paths = []
                    for j in range(len(path) - 1):
                        boundary_path = path[:j] + path[j+1:]
                        if boundary_path in paths:
                            boundary_paths.append(paths.index(boundary_path))
                    
                    # Add to boundary matrix
                    for j in boundary_paths:
                        boundary_matrix[i, j] = 1
                
                # Compute homology
                homology = self._compute_homology_from_boundary(boundary_matrix)
                path_homology[dim] = homology
            
            # Store results
            self.path_homology_diagrams = {
                'homology': path_homology,
                'path_complex': path_complex,
                'flow_complex': self.flow_complex
            }
            
            # Generate visualizations
            if not os.path.exists('path_homology_plots'):
                os.makedirs('path_homology_plots')
                
            plt.figure(figsize=(12, 6))
            for dim in range(max_path_length + 1):
                if dim in path_homology:
                    betti = path_homology[dim]['betti']
                    plt.plot([0, self.n_samples], [betti, betti], 
                           c=['blue', 'red', 'green'][dim],
                           linewidth=2, label=f'H{dim}')
            
            plt.xlabel('Time')
            plt.ylabel('Betti Numbers')
            plt.title('Path Homology Betti Numbers')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('path_homology_plots/path_homology.png')
            plt.close()
            
            return self.path_homology_diagrams
            
        except Exception as e:
            print(f"Error in path homology computation: {str(e)}")
            return None
            
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
        # Compute rank of boundary matrix
        rank = np.linalg.matrix_rank(boundary_matrix)
        
        # Compute nullity
        nullity = boundary_matrix.shape[0] - rank
        
        
        # Compute Betti numbers
        betti = nullity - rank
        
        return {
            'rank': rank,
            'nullity': nullity,
            'betti': betti
        } 

    def compute_persistent_path_zigzag_homology(self, window_size=100, overlap=50, max_path_length=3, min_epsilon=0.1, max_epsilon=2.0, num_steps=10, output_dir=None):
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
            
            for i in range(0, self.n_samples - window_size + 1, window_size - overlap):
                # Get window data indices
                window_idx = list(range(i, min(i + window_size, self.n_samples)))
                window_indices.append(window_idx)
                
                # Create temporal distance submatrix for this window
                if self.temporal_distance_matrix is None:
                    # Compute it if not already done
                    self.compute_temporally_weighted_distance()
                    self.temporal_distance_matrix = self.distance_matrix
                
                window_dist_matrix = self.temporal_distance_matrix[np.ix_(window_idx, window_idx)]
                
                # Create directed network for this window
                epsilon = min_epsilon  # Start with minimum epsilon
                G = nx.DiGraph()
                
                # Add nodes
                for j, idx in enumerate(window_idx):
                    G.add_node(j, original_idx=idx, features=self.features[idx], timestamp=self.timestamps[idx])
                
                # Add edges (respecting temporal ordering)
                for j in range(len(window_idx)):
                    for k in range(j+1, len(window_idx)):  # Enforce j < k for temporal ordering
                        if window_dist_matrix[j, k] <= epsilon:
                            G.add_edge(j, k, weight=1.0/max(window_dist_matrix[j, k], 1e-10))
                
                networks.append(G)
            
            print(f"Network creation completed in {time.time() - network_start:.2f} seconds")
            
            # For each network window, compute path complex
            path_complexes = []
            for G in networks:
                path_complex = self.compute_path_complex(G, max_path_length)
                path_complexes.append(path_complex)
            
            # Initialize zigzag path persistence
            path_zigzag_diagrams = {
                'window_diagrams': [],
                'transitions': [],
                'betti_numbers': []
            }
            
            # Process each dimension
            dimension_start = time.time()
            for dim in range(max_path_length + 1):
                betti_series = []
                
                # Compute homology for each window
                for i, path_complex in enumerate(path_complexes):
                    if dim not in path_complex or not path_complex[dim]:
                        betti_series.append(0)
                        continue
                        
                    paths = path_complex[dim]
                    
                    # Create boundary matrix
                    n_paths = len(paths)
                    boundary_matrix = np.zeros((n_paths, n_paths))
                    
                    # Fill boundary matrix
                    for j, path in enumerate(paths):
                        # Get boundary paths
                        boundary_paths = []
                        for k in range(len(path) - 1):
                            boundary_path = path[:k] + path[k+1:]
                            if dim-1 in path_complex and boundary_path in path_complex[dim-1]:
                                boundary_paths.append(path_complex[dim-1].index(boundary_path))
                        
                        # Add to boundary matrix (if appropriate dimension exists)
                        if dim-1 in path_complex:
                            for k in boundary_paths:
                                if k < boundary_matrix.shape[1]:
                                    boundary_matrix[j, k] = 1
                    
                    # Compute homology
                    homology = self._compute_homology_from_boundary(boundary_matrix)
                    betti_series.append(homology['betti'])
                
                path_zigzag_diagrams[f'betti_{dim}'] = betti_series
            
            # Compute zigzag persistence across windows
            transition_features = []
            
            # For each pair of consecutive windows, analyze transitions
            for i in range(len(window_indices) - 1):
                current_window = window_indices[i]
                next_window = window_indices[i+1]
                
                # Find overlap indices
                overlap_indices = set(current_window).intersection(set(next_window))
                
                # Track transitions of paths
                for dim in range(1, max_path_length + 1):  # Skip 0-dim (just points)
                    if dim in path_complexes[i] and dim in path_complexes[i+1]:
                        # Find paths in current window that have elements in the overlap
                        persistent_paths = []
                        for path in path_complexes[i][dim]:
                            # Check if path nodes are in overlap (need original indices)
                            path_nodes = [networks[i].nodes[node]['original_idx'] for node in path]
                            if any(idx in overlap_indices for idx in path_nodes):
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
                os.makedirs(output_dir, exist_ok=True)
                
                # Plot Betti numbers across windows
                plt.figure(figsize=(12, 6))
                
                for dim in range(max_path_length + 1):
                    betti_key = f'betti_{dim}'
                    if betti_key in path_zigzag_diagrams:
                        betti_values = path_zigzag_diagrams[betti_key]
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
                if transition_features:
                    plt.figure(figsize=(12, 6))
                    
                    # Group by dimension
                    for dim in range(1, max_path_length + 1):
                        dim_transitions = [t['persistent_paths'] for t in transition_features if t['dimension'] == dim]
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
            
            print(f"Total path zigzag persistence computation completed in {time.time() - start_time:.2f} seconds")
            return self.path_zigzag_diagrams
            
        except Exception as e:
            print(f"Error in path zigzag persistence computation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None 