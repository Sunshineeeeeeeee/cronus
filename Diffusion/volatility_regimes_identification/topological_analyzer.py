import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
import os
import itertools
import warnings
import gudhi
import gudhi.point_cloud
import gudhi.representations
import gudhi.persistence_graphical_tools as gd_plot
from gudhi.rips_complex import RipsComplex
from gudhi.weighted_rips_complex import WeightedRipsComplex
from gudhi.persistence_graphical_tools import plot_persistence_barcode
from gudhi.persistence_graphical_tools import plot_persistence_diagram
import pandas as pd

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
    
    def compute_persistent_homology(self, min_epsilon=0.1, max_epsilon=2.0, num_steps=10, use_weighted=False):
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
            
        Returns:
        --------
        dict
            Dictionary containing persistence diagrams and related information
        """
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
                if not os.path.exists('persistence_plots'):
                    os.makedirs('persistence_plots')
                    
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
                plt.savefig('persistence_plots/persistence_diagram.png')
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
                plt.savefig('persistence_plots/persistence_barcode.png')
                plt.close()
                
            except Exception as viz_error:
                print(f"Warning: Could not generate visualizations: {str(viz_error)}")
            
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
    
    def identify_regimes(self, n_regimes=3, affinity_sigma=1.0):
        """
        Identify volatility regimes using directed spectral clustering.
        
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
        
        # Perform spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_regimes,
            affinity='precomputed',
            random_state=42
        )
        
        regime_labels = clustering.fit_predict(affinity)
        
        self.regime_labels = regime_labels
        return regime_labels
    
    def analyze_regimes(self):
        """
        Analyze the identified regimes to extract statistics and patterns.
        
        Returns:
        --------
        dict
            Dictionary containing regime analysis results
        """
        print("Analyzing volatility regimes...")
        
        if self.regime_labels is None:
            raise ValueError("Must detect regimes before analyzing them")
        
        n_regimes = len(np.unique(self.regime_labels))
        regime_stats = []
        
        # Initialize transition probability matrix
        transition_probs = np.zeros((n_regimes, n_regimes))
        
        # Compute transitions
        for i in range(len(self.regime_labels) - 1):
            current_regime = self.regime_labels[i]
            next_regime = self.regime_labels[i + 1]
            transition_probs[current_regime, next_regime] += 1
        
        # Normalize transition probabilities
        row_sums = transition_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_probs = transition_probs / row_sums
        
        # Analyze each regime
        for regime in range(n_regimes):
            regime_mask = self.regime_labels == regime
            regime_points = np.where(regime_mask)[0]
            regime_times = self.timestamps[regime_points]
            
            if len(regime_points) > 0:
                # Convert numpy.timedelta64 to pandas Timedelta for duration calculation
                if np.issubdtype(regime_times.dtype, np.datetime64):
                    start_time = pd.Timestamp(min(regime_times))
                    end_time = pd.Timestamp(max(regime_times))
                    duration = (end_time - start_time).total_seconds()
                else:
                    # If not datetime, use number of points as duration
                    duration = len(regime_points)
                
                # Calculate regime statistics
                stats = {
                    'regime_id': regime,
                    'size': len(regime_points),
                    'start_time': start_time if np.issubdtype(regime_times.dtype, np.datetime64) else regime_points[0],
                    'end_time': end_time if np.issubdtype(regime_times.dtype, np.datetime64) else regime_points[-1],
                    'duration': duration,
                    'points': regime_points.tolist()
                }
                
                regime_stats.append(stats)
        
        result = {
            'regime_labels': self.regime_labels,
            'regime_stats': regime_stats,
            'transition_probs': transition_probs,
            'n_regimes': n_regimes
        }
        
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