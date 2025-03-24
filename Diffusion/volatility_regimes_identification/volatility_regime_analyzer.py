import numpy as np
import pandas as pd
import os
import pickle
from .microstructure_features import MicrostructureFeatureEngine
from .information_theory import InformationTheoryEnhancer
from .topological_analyzer import TopologicalDataAnalyzer
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class VolatilityRegimeAnalyzer:
    """
    Main class that combines the entire TDA pipeline for volatility regime detection.
    """
    
    def __init__(self, df=None, timestamp_col='Timestamp', price_col='Value', 
                 volume_col='Volume', volatility_col='Volatility'):
        """
        Initialize the volatility regime analyzer.
        
        Parameters:
        -----------
        df : pandas.DataFrame or None
            Input dataframe with tick data
        timestamp_col : str
            Column name for timestamps
        price_col : str
            Column name for price values
        volume_col : str
            Column name for volume values
        volatility_col : str
            Column name for volatility values
        """
        self.df = df
        self.timestamp_col = timestamp_col
        self.price_col = price_col
        self.volume_col = volume_col
        self.volatility_col = volatility_col
        
        # Initialize pipeline components
        self.feature_engine = None
        self.info_enhancer = None
        self.tda = None
        
        # Results
        self.features = None
        self.feature_names = None
        self.enhanced_features = None
        self.enhanced_feature_names = None
        self.regime_labels = None
        self.regime_analysis = None
        
    def compute_features(self, window_sizes=[10, 50, 100]):
        """
        Compute features using the microstructure feature engine.
        
        Parameters:
        -----------
        window_sizes : list
            List of window sizes for feature calculation
            
        Returns:
        --------
        self
        """
        if self.df is None:
            raise ValueError("DataFrame must be provided")
            
        print("\n--- STEP 1: Computing Microstructure Features ---")
        
        self.feature_engine = MicrostructureFeatureEngine(
            self.df, 
            timestamp_col=self.timestamp_col,
            price_col=self.price_col,
            volume_col=self.volume_col,
            volatility_col=self.volatility_col
        )
        
        self.features, self.feature_names, self.df = self.feature_engine.extract_all_features(window_sizes)
        
        return self
        
    def _compute_entropy(self):
        """
        Compute entropy for each feature.
        
        Returns:
        --------
        numpy.ndarray
            Array of entropy values for each feature
        """
        if self.features is None:
            raise ValueError("Features must be computed first")
            
        # Initialize information theory enhancer if not already done
        if not hasattr(self, 'info_enhancer') or self.info_enhancer is None:
            self.info_enhancer = InformationTheoryEnhancer(self.features, self.feature_names)
            
        # Calculate entropy
        self.info_enhancer.estimate_shannon_entropy()
            
        return self.info_enhancer.entropy
        
    def enhance_features(self, n_features=10, include_entropy=True, include_kl=True):
        """
        Apply information-theoretic feature enhancement.
        
        Parameters:
        -----------
        n_features : int
            Number of top features to include
        include_entropy : bool
            Whether to include entropy-weighted features
        include_kl : bool
            Whether to include KL divergence features
            
        Returns:
        --------
        self
        """
        if self.features is None:
            raise ValueError("Features must be computed first")
            
        print("\n--- STEP 2: Applying Information-Theoretic Feature Enhancement ---")
        
        self.info_enhancer = InformationTheoryEnhancer(self.features, self.feature_names)
        
        # Calculate information-theoretic measures
        self.info_enhancer.estimate_shannon_entropy()
        self.info_enhancer.compute_mutual_information_matrix()
        
        if include_kl:
            self.info_enhancer.compute_kl_divergence()
            
        # Rank features by importance
        self.info_enhancer.rank_features_by_importance()
        
        # Create enhanced feature set
        self.enhanced_features, self.enhanced_feature_names = self.info_enhancer.enhance_features(
            n_features=n_features,
            include_entropy=include_entropy,
            include_kl=include_kl
        )
        
        return self
        
    def detect_regimes(self, n_regimes=4, alpha=0.5, beta=0.1, lambda_info=1.0,
                    min_epsilon=0.1, max_epsilon=2.0, num_steps=10, window_size=100, overlap=50,
                    max_path_length=3, create_mapper=True, compute_homology=True, output_dir=None):
        """
        Detect volatility regimes using TDA and information theory, with path complex and zigzag persistence.
        This is specially adapted for sequential market microstructure data.
        
        Parameters:
            n_regimes (int): Number of regimes to detect
            alpha (float): Weight for temporal component
            beta (float): Decay rate for temporal distance
            lambda_info (float): Weight for mutual information
            min_epsilon (float): Minimum distance threshold
            max_epsilon (float): Maximum distance threshold
            num_steps (int): Number of steps for persistence
            window_size (int): Size of sliding window for zigzag persistence
            overlap (int): Number of points to overlap between windows
            max_path_length (int): Maximum length of paths to consider
            create_mapper (bool): Whether to create mapper graph
            compute_homology (bool): Whether to compute persistent homology
            output_dir (str): Directory to save outputs
        """
        print("\n=== Detecting Volatility Regimes ===")
        
        # Get timestamps from data
        timestamps = self.df[self.timestamp_col].values
        
        print("\n--- STEP 1: Initializing TDA Pipeline ---")
        self.tda = TopologicalDataAnalyzer(
            self.enhanced_features, 
            self.enhanced_feature_names,
            timestamps
        )
        
        # First compute the temporal distance matrix
        print("\n--- STEP 2: Computing Temporal Distance Matrix ---")
        self.tda.compute_temporally_weighted_distance(
            alpha=alpha, 
            beta=beta, 
            lambda_info=lambda_info,
            mi_matrix=self.info_enhancer.mi_matrix if hasattr(self.info_enhancer, 'mi_matrix') else None
        )
        
        # Store the temporal distance matrix
        self.tda.temporal_distance_matrix = self.tda.distance_matrix
        
        # Compute persistent homology using path complex and zigzag persistence if requested
        if compute_homology:
            print("\n--- STEP 3: Computing Path Zigzag Persistent Homology ---")
            self.tda.compute_persistent_path_zigzag_homology(
                window_size=window_size,
                overlap=overlap,
                max_path_length=max_path_length,
                min_epsilon=min_epsilon,
                max_epsilon=max_epsilon,
                num_steps=num_steps,
                output_dir=output_dir
            )
        
        # Create mapper graph if requested
        if create_mapper:
            print("\n--- STEP 4: Creating Mapper Graph ---")
            self.tda.create_temporal_mapper()
        
        # Identify regimes
        print("\n--- STEP 5: Identifying Regimes ---")
        self.regime_labels = self.tda.identify_regimes(n_regimes=n_regimes)
        
        print("\nRegime detection completed.")
        return self.regime_labels
        
    def visualize_results(self, output_dir='.', filename_prefix='volatility_regimes'):
        """
        Visualize the detected regimes.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations
        filename_prefix : str
            Prefix for filenames
            
        Returns:
        --------
        self
        """
        if self.regime_labels is None:
            raise ValueError("Regimes must be detected first")
            
        print("\n--- STEP 7: Visualizing Results ---")
        
        self.tda.visualize_regimes(output_dir=output_dir, filename_prefix=filename_prefix)
        
        return self
        
    def get_regime_labels(self):
        """
        Get the regime labels for each data point.
        
        Returns:
        --------
        numpy.ndarray
            Array of regime labels
        """
        return self.regime_labels
        
    def get_regime_analysis(self):
        """
        Get detailed analysis of the identified regimes.
        
        Returns:
        --------
        dict
            Dictionary of regime characteristics
        """
        return self.regime_analysis
        
    def label_new_data(self, new_df):
        """
        Label new data with the identified regimes.
        
        Parameters:
        -----------
        new_df : pandas.DataFrame
            New data to label with the same columns as the original data
            
        Returns:
        --------
        pandas.DataFrame
            New data with added 'regime' column
        """
        if self.regime_labels is None or self.tda is None:
            raise ValueError("Regime detection model must be trained first")
            
        print("Labeling new data with existing regimes...")
        
        # Process the new data through the same pipeline
        # 1. Extract features
        new_feature_engine = MicrostructureFeatureEngine(
            new_df.copy(), 
            timestamp_col=self.timestamp_col,
            price_col=self.price_col,
            volume_col=self.volume_col,
            volatility_col=self.volatility_col
        )
        
        new_features, _, new_df_processed = new_feature_engine.extract_all_features(
            window_sizes=[10, 50, 100] 
        )
        
        # 2. Get feature selection if available
        if self.enhanced_features is not None and self.info_enhancer is not None:
            # Select the same features that were used during training
            top_indices = self.info_enhancer.select_top_features()
            selected_features = new_features[:, top_indices]
            
            # Apply any entropy weighting or KL divergence enhancement if used during training
            if hasattr(self.info_enhancer, 'entropy') and len(self.enhanced_feature_names) > len(top_indices):
                entropy_weights = self.info_enhancer.entropy[top_indices] / np.sum(self.info_enhancer.entropy[top_indices])
                entropy_weighted = selected_features * entropy_weights.reshape(1, -1)
                new_enhanced_features = np.column_stack([selected_features, entropy_weighted])
            else:
                new_enhanced_features = selected_features
        else:
            # If no enhancement was done, use all features
            new_enhanced_features = new_features
            
        # 3. Calculate distances from each new point to each existing regime centroid
        regime_centroids = {}
        for regime in range(len(np.unique(self.regime_labels))):
            regime_points = np.where(self.regime_labels == regime)[0]
            regime_centroids[regime] = np.mean(self.enhanced_features[regime_points], axis=0)
            
        # Assign each new point to the nearest regime
        n_new_points = len(new_enhanced_features)
        new_regime_labels = np.zeros(n_new_points, dtype=int)
        
        for i in range(n_new_points):
            min_dist = float('inf')
            nearest_regime = 0
            
            for regime, centroid in regime_centroids.items():
                # Calculate Euclidean distance
                if len(new_enhanced_features[i]) == len(centroid):
                    dist = np.sqrt(np.sum((new_enhanced_features[i] - centroid) ** 2))
                else:
                    # Handle feature dimension mismatch (use common features)
                    min_dim = min(len(new_enhanced_features[i]), len(centroid))
                    dist = np.sqrt(np.sum((new_enhanced_features[i][:min_dim] - centroid[:min_dim]) ** 2))
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_regime = regime
                    
            new_regime_labels[i] = nearest_regime
            
        # Add regime labels to the processed dataframe
        new_df_processed['regime'] = new_regime_labels
        
        # Calculate confidence metrics (distance to assigned regime centroid vs. next closest)
        if len(regime_centroids) > 1:
            confidences = []
            
            for i in range(n_new_points):
                assigned_regime = new_regime_labels[i]
                assigned_centroid = regime_centroids[assigned_regime]
                
                # Distance to assigned centroid
                if len(new_enhanced_features[i]) == len(assigned_centroid):
                    assigned_dist = np.sqrt(np.sum((new_enhanced_features[i] - assigned_centroid) ** 2))
                else:
                    min_dim = min(len(new_enhanced_features[i]), len(assigned_centroid))
                    assigned_dist = np.sqrt(np.sum((new_enhanced_features[i][:min_dim] - assigned_centroid[:min_dim]) ** 2))
                
                # Find next closest centroid
                next_closest_dist = float('inf')
                
                for regime, centroid in regime_centroids.items():
                    if regime == assigned_regime:
                        continue
                        
                    if len(new_enhanced_features[i]) == len(centroid):
                        dist = np.sqrt(np.sum((new_enhanced_features[i] - centroid) ** 2))
                    else:
                        min_dim = min(len(new_enhanced_features[i]), len(centroid))
                        dist = np.sqrt(np.sum((new_enhanced_features[i][:min_dim] - centroid[:min_dim]) ** 2))
                        
                    if dist < next_closest_dist:
                        next_closest_dist = dist
                
                # Confidence metric: ratio of distances (higher is better)
                if assigned_dist > 0:
                    confidence = next_closest_dist / assigned_dist
                else:
                    confidence = float('inf')  # Perfect match to centroid
                    
                confidences.append(confidence)
                
            new_df_processed['regime_confidence'] = confidences
            
        print(f"Labeled {n_new_points} new data points")
        
        # Merge regime labels back to original dataframe (if desired)
        # This ensures any columns from the original dataframe are preserved
        if set(new_df_processed.index) == set(new_df.index):
            result_df = new_df.copy()
            result_df['regime'] = new_df_processed['regime']
            if 'regime_confidence' in new_df_processed.columns:
                result_df['regime_confidence'] = new_df_processed['regime_confidence']
        else:
            # Handle case where indices might not match
            result_df = new_df_processed
            
        return result_df
    
    def predict_regime_transitions(self, steps_ahead=10):
        """
        Predict future regime transitions based on transition probabilities.
        
        Parameters:
        -----------
        steps_ahead : int
            Number of steps to predict ahead
            
        Returns:
        --------
        list
            List of most likely regime sequences
        """
        if self.regime_analysis is None:
            raise ValueError("Regime analysis must be computed first")
            
        # Get transition probability matrix
        transition_probs = self.regime_analysis['transition_probs']
        n_regimes = self.regime_analysis['n_regimes']
        
        # Get current regime (last observed point)
        current_regime = self.regime_labels[-1]
        
        # Initialize prediction
        predictions = [current_regime]
        
        # Simple Markov chain prediction
        for _ in range(steps_ahead):
            # Get transition probabilities from current regime
            probs = transition_probs[current_regime]
            
            # If all zeros (no observed transitions), assume uniform distribution
            if np.sum(probs) == 0:
                probs = np.ones(n_regimes) / n_regimes
                
            # Sample next regime based on transition probabilities
            next_regime = np.random.choice(n_regimes, p=probs)
            predictions.append(next_regime)
            
            # Update current regime for next iteration
            current_regime = next_regime
            
        return predictions
    
    def save_model(self, filepath):
        """
        Save the regime detection model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
            
        Returns:
        --------
        None
        """
        import pickle
        
        model_data = {
            'feature_names': self.feature_names,
            'enhanced_feature_names': self.enhanced_feature_names,
            'regime_labels': self.regime_labels,
            'regime_analysis': self.regime_analysis
        }
        
        # Add feature importance if available
        if hasattr(self, 'info_enhancer') and self.info_enhancer is not None:
            if hasattr(self.info_enhancer, 'feature_importance'):
                model_data['feature_importance'] = self.info_enhancer.feature_importance
                
        # Save the model to a file
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved to {filepath}")
        
    @classmethod
    def load_model(cls, filepath, df=None):
        """
        Load a saved regime detection model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        df : pandas.DataFrame or None
            Data to apply the model to (optional)
            
        Returns:
        --------
        VolatilityRegimeAnalyzer
            Loaded model
        """
        import pickle
        
        # Load the model from a file
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        # Create a new instance of the analyzer
        analyzer = cls()
        
        # Restore model attributes
        analyzer.feature_names = model_data['feature_names']
        analyzer.enhanced_feature_names = model_data['enhanced_feature_names']
        analyzer.regime_labels = model_data['regime_labels']
        analyzer.regime_analysis = model_data['regime_analysis']
        
        # Restore feature importance if available
        if 'feature_importance' in model_data:
            if analyzer.info_enhancer is None:
                analyzer.info_enhancer = type('obj', (object,), {})
            analyzer.info_enhancer.feature_importance = model_data['feature_importance']
            
        print(f"Model loaded from {filepath}")
        
        # Apply the model to the provided data if any
        if df is not None:
            labeled_df = analyzer.label_new_data(df)
            return analyzer, labeled_df
            
        return analyzer 