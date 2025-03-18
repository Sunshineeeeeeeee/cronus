import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
import os
import itertools
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class TopologicalDataAnalyzer:
    """
    Implements the topological data analysis components for volatility regime detection.
    
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
        
    def compute_temporally_weighted_distance(self, alpha=0.5, beta=0.1, lambda_info=1.0, mi_matrix=None):
        """
        Compute temporally-weighted distance matrix between data points.
        
        Parameters:
            alpha (float): Weight for temporal component
            beta (float): Decay rate for temporal distance
            lambda_info (float): Weight for mutual information component
            mi_matrix (np.ndarray): Pre-computed mutual information matrix
        """
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
        return dist_matrix
    
    def construct_directed_network(self, epsilon=0.5, enforce_temporal=True):
        """
        Construct a directed weighted network based on the temporal distance matrix.
        
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
    
    def compute_persistent_homology(self, min_epsilon=0.1, max_epsilon=2.0, num_steps=10):
        """
        Compute persistent homology using the path complex filtration.
        
        Parameters:
        -----------
        min_epsilon : float
            Minimum distance threshold
        max_epsilon : float
            Maximum distance threshold
        num_steps : int
            Number of thresholds to compute
            
        Returns:
        --------
        dict
            Dictionary of persistence diagrams by dimension
        """
        print("Computing persistent homology...")
        
        # Define filtration values
        epsilon_values = np.linspace(min_epsilon, max_epsilon, num_steps)
        
        # Initialize dictionaries to store results
        all_path_complexes = {}
        betti_numbers = {k: [] for k in range(3)}  # Track Betti numbers for dims 0, 1, 2
        
        # Compute path complex at each filtration value
        for eps in epsilon_values:
            # Construct network at this epsilon
            G = self.construct_directed_network(epsilon=eps)
            
            # Compute path complex
            path_complex = self.compute_path_complex(G)
            all_path_complexes[eps] = path_complex
            
            # Count connected components (Betti-0)
            components = list(nx.weakly_connected_components(G))
            betti_numbers[0].append(len(components))
            
            # Count cycles (Betti-1)
            # For directed graphs, use simple cycle finding
            try:
                cycles = list(nx.simple_cycles(G))
                betti_numbers[1].append(len(cycles))
            except nx.NetworkXNoCycle:
                betti_numbers[1].append(0)
                
            # Higher-order structures are more complex to compute
            # This is a simplified approach for Betti-2
            triangles = 0
            for path in path_complex[2]:
                # Check if it forms a "void" (simplification)
                if (path[2], path[0]) in G.edges():
                    triangles += 1
            betti_numbers[2].append(triangles)
            
        # Create simplified persistence diagrams
        # This is a basic approximation - production code would use more sophisticated libraries
        persistence_diagrams = {}
        
        for dim in range(3):
            diagram = []
            active_features = {}
            
            for i, eps in enumerate(epsilon_values):
                # Birth of new features
                if i > 0:
                    new_features = max(0, betti_numbers[dim][i] - betti_numbers[dim][i-1])
                    for _ in range(new_features):
                        birth_time = eps
                        active_features[len(active_features)] = birth_time
                        
                # Death of features
                if i > 0 and i < len(epsilon_values) - 1:
                    dying_features = max(0, betti_numbers[dim][i] - betti_numbers[dim][i+1])
                    keys_to_remove = list(active_features.keys())[:dying_features]
                    
                    for key in keys_to_remove:
                        birth_time = active_features.pop(key)
                        death_time = epsilon_values[i+1]
                        diagram.append((birth_time, death_time))
            
            # Add features that never die
            for birth_time in active_features.values():
                diagram.append((birth_time, np.inf))
                
            persistence_diagrams[dim] = np.array(diagram)
            
        self.persistence_diagrams = persistence_diagrams
        return persistence_diagrams
    
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
        
        # Rest of the method remains unchanged...
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
    
    def identify_regimes(self, n_regimes=None, affinity_sigma=1.0):
        """
        Identify volatility regimes using directed spectral clustering.
        
        Parameters:
        -----------
        n_regimes : int or None
            Number of regimes to identify (if None, determined automatically)
        affinity_sigma : float
            Bandwidth parameter for affinity calculation
            
        Returns:
        --------
        numpy.ndarray
            Array of regime labels for each data point
        """
        print("Identifying volatility regimes...")
        
        if self.temporal_distance_matrix is None:
            raise ValueError("Temporal distance matrix must be computed first")
            
        # Create affinity matrix from distance matrix
        affinity = np.zeros_like(self.temporal_distance_matrix)
        
        # Only consider i < j (temporal ordering)
        for i in range(self.n_samples):
            for j in range(i+1, self.n_samples):
                # Calculate modified affinity with information weighting
                if hasattr(self, 'mi_matrix'):
                    mi_term = self.mi_matrix[i, j]
                    affinity[i, j] = np.exp(-self.temporal_distance_matrix[i, j]**2 / (2 * affinity_sigma**2)) * (1 / (mi_term + 1e-10))
                else:
                    affinity[i, j] = np.exp(-self.temporal_distance_matrix[i, j]**2 / (2 * affinity_sigma**2))
                    
        # Make symmetric for spectral clustering
        affinity = affinity + affinity.T
        
        # Determine number of regimes if not specified
        if n_regimes is None:
            # Determine using eigengap heuristic
            from sklearn.manifold import SpectralEmbedding
            
            embedder = SpectralEmbedding(n_components=20, affinity='precomputed')
            embedding = embedder.fit_transform(affinity)
            
            # Calculate eigengap
            eigenvalues = embedder.eigenvalues_
            eigengaps = np.diff(eigenvalues)
            
            # Find elbow point (significant drop in eigenvalues)
            max_gap_idx = np.argmax(eigengaps)
            n_regimes = max_gap_idx + 2  # Add 2 because we're looking at gaps
            
            # Cap the number of regimes to a reasonable range
            n_regimes = max(2, min(10, n_regimes))
            
            print(f"Automatically determined {n_regimes} regimes based on eigengap heuristic")
            
        # Perform spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_regimes,
            affinity='precomputed',
            random_state=42
        )
        
        regime_labels = clustering.fit_predict(affinity)
        
        # Sort regimes by temporal order (first occurrence)
        regime_order = {}
        for i, label in enumerate(regime_labels):
            if label not in regime_order:
                regime_order[label] = i
                
        # Create mapping to sorted labels
        sorted_regimes = sorted(regime_order.items(), key=lambda x: x[1])
        label_mapping = {old_label: new_label for new_label, (old_label, _) in enumerate(sorted_regimes)}
        
        # Apply mapping
        regime_labels = np.array([label_mapping[label] for label in regime_labels])
        
        self.regime_labels = regime_labels
        return regime_labels
    
    def analyze_regimes(self):
        """
        Analyze identified regimes including transitions, characteristics, and significance.
        
        Returns:
        --------
        dict
            Dictionary containing regime analysis results
        """
        print("Analyzing volatility regimes...")
        
        if self.regime_labels is None:
            raise ValueError("Regimes must be identified first")
            
        n_regimes = len(np.unique(self.regime_labels))
        
        # 1. Compute transition probabilities between regimes
        transitions = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(self.regime_labels) - 1):
            from_regime = self.regime_labels[i]
            to_regime = self.regime_labels[i + 1]
            
            # Only count transitions between different regimes
            if from_regime != to_regime:
                transitions[from_regime, to_regime] += 1
                
        # Normalize to get probabilities
        transition_probs = np.zeros_like(transitions)
        for i in range(n_regimes):
            row_sum = transitions[i].sum()
            if row_sum > 0:
                transition_probs[i] = transitions[i] / row_sum
                
        # 2. Compute regime statistics
        regime_stats = []
        
        for regime in range(n_regimes):
            regime_points = np.where(self.regime_labels == regime)[0]
            
            # Skip if no points in this regime
            if len(regime_points) == 0:
                continue
                
            # Get regime timespan
            regime_times = self.timestamps[regime_points]
            if np.issubdtype(self.timestamps.dtype, np.datetime64):
                start_time = min(regime_times)
                end_time = max(regime_times)
                duration = (end_time - start_time).total_seconds()
            else:
                start_time = min(regime_times)
                end_time = max(regime_times)
                duration = end_time - start_time
                
            # Get regime volatility (if available)
            vol_features = [i for i, name in enumerate(self.feature_names) if 'volatility' in name.lower()]
            
            if vol_features:
                vol_idx = vol_features[0]
                vol_values = self.features[regime_points, vol_idx]
                mean_vol = np.mean(vol_values)
                max_vol = np.max(vol_values)
                min_vol = np.min(vol_values)
                vol_range = max_vol - min_vol
                vol_std = np.std(vol_values)
            else:
                mean_vol = max_vol = min_vol = vol_range = vol_std = np.nan
                
            # Compute feature importance within this regime
            if hasattr(self, 'mi_matrix'):
                # Use mutual information to rank features
                sub_mi = self.mi_matrix[regime_points][:, vol_features[0] if vol_features else 0]
                feature_importance = {
                    self.feature_names[i]: np.mean(sub_mi[i]) 
                    for i in range(len(self.feature_names))
                }
            else:
                feature_importance = {}
                
            # Store statistics
            regime_stats.append({
                'regime_id': regime,
                'size': len(regime_points),
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'mean_vol': mean_vol,
                'max_vol': max_vol,
                'min_vol': min_vol,
                'vol_range': vol_range,
                'vol_std': vol_std,
                'feature_importance': feature_importance
            })
            
        # 3. Compute information-theoretic measures between regimes
        regime_info = {}
        
        # Compute Jensen-Shannon divergence between regimes
        for r1 in range(n_regimes):
            for r2 in range(r1+1, n_regimes):
                r1_points = np.where(self.regime_labels == r1)[0]
                r2_points = np.where(self.regime_labels == r2)[0]
                
                if len(r1_points) == 0 or len(r2_points) == 0:
                    continue
                    
                # Compute divergence for each feature
                feature_divergences = {}
                
                for j, feature_name in enumerate(self.feature_names):
                    x1 = self.features[r1_points, j]
                    x2 = self.features[r2_points, j]
                    
                    # Compute histograms
                    hist1, bin_edges = np.histogram(x1, bins=20, density=True)
                    hist2, _ = np.histogram(x2, bins=bin_edges, density=True)
                    
                    # Add small constant to avoid zero probabilities
                    hist1 += 1e-10
                    hist2 += 1e-10
                    
                    # Normalize
                    hist1 /= hist1.sum()
                    hist2 /= hist2.sum()
                    
                    # Compute Jensen-Shannon divergence
                    m = 0.5 * (hist1 + hist2)
                    js_div = 0.5 * (np.sum(hist1 * np.log(hist1 / m)) + np.sum(hist2 * np.log(hist2 / m)))
                    
                    feature_divergences[feature_name] = js_div
                    
                regime_info[(r1, r2)] = {
                    'feature_divergences': feature_divergences,
                    'mean_divergence': np.mean(list(feature_divergences.values()))
                }
                
        result = {
            'regime_labels': self.regime_labels,
            'regime_stats': regime_stats,
            'transition_probs': transition_probs,
            'n_regimes': n_regimes
        }
        
        return result
    
    def visualize_regimes(self, output_dir, filename_prefix=''):
        """
        Visualize the identified regimes.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations
        filename_prefix : str
            Prefix for output filenames
        """
        print("Generating regime visualizations...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert timestamps to hours from start
        if np.issubdtype(self.timestamps.dtype, np.datetime64):
            t0 = self.timestamps[0]
            time_diffs = self.timestamps - t0
            # Convert to nanoseconds, then to hours
            time_values = time_diffs.astype('timedelta64[ns]').astype(np.float64) / (1e9 * 3600)  # Convert to hours
            x_label = "Time (hours from start)"
        else:
            time_values = np.array(self.timestamps)
            x_label = "Time"
        
        # Rest of the visualization code remains the same...
        # Plot regimes over time
        plt.figure(figsize=(15, 8))
        
        # Plot each regime with a different color
        unique_regimes = np.unique(self.regime_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regimes)))
        
        for regime, color in zip(unique_regimes, colors):
            mask = self.regime_labels == regime
            plt.scatter(time_values[mask], self.features[mask, 0],
                       c=[color], label=f'Regime {regime+1}',
                       alpha=0.6, s=30)
        
        plt.xlabel(x_label)
        plt.ylabel("Feature Value")
        plt.title("Volatility Regimes Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f"{filename_prefix}regimes_over_time.png"))
        plt.close()
        
        # Additional visualizations can be added here... 