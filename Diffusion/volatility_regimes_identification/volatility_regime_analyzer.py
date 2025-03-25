import numpy as np
import pandas as pd
import os
import pickle
from .microstructure_features import MicrostructureFeatureEngine
from .information_theory import InformationTheoryEnhancer
from .topological_analyzer import TopologicalDataAnalyzer
import warnings
from sklearn.preprocessing import StandardScaler

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
        tuple
            (features, feature_names, updated_df, self)
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
        
        # Return the computed features, feature names, and updated dataframe along with self
        return self.features, self.feature_names, self.df, self
        
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
        
    def enhance_features(self, n_features=10, include_entropy=True, include_kl=True, is_training=True):
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
        is_training : bool
            Whether this is the training phase (if False, skip MI computation)
            
        Returns:
        --------
        self
        """
        if self.features is None:
            raise ValueError("Features must be computed first")
            
        print("\n--- STEP 2: Applying Information-Theoretic Feature Enhancement ---")
        
        self.info_enhancer = InformationTheoryEnhancer(self.features, self.feature_names)
        
        # Calculate information-theoretic measures only during training
        if is_training:
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
            include_kl=include_kl,
            is_training=is_training
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
        
        # Extract features and get feature names from the return value
        # Use the same window sizes as in training
        window_sizes_to_use = getattr(self, 'window_sizes', [10, 50, 100])
        print(f"Using window sizes for feature extraction: {window_sizes_to_use}")
        new_features, new_feature_names, new_df_processed = new_feature_engine.extract_all_features(
            window_sizes=window_sizes_to_use
        )
        
        # 2. Get feature selection if available
        if self.enhanced_features is not None and self.info_enhancer is not None:
            # Create new information theory enhancer
            new_info_enhancer = InformationTheoryEnhancer(
                new_features,
                new_feature_names  # Use the feature names from the return value
            )
            
            # Select the same features that were used during training using stored importance
            if hasattr(self.info_enhancer, 'feature_importance'):
                # Copy feature importance to avoid recomputing
                new_info_enhancer.feature_importance = self.info_enhancer.feature_importance
                top_indices = new_info_enhancer.select_top_features()
            else:
                # If no feature importance, compute it using FFT approximation
                print("Computing mutual information using FFT approximation...")
                # Always use FFT-based MI computation
                n_samples = len(new_df)
                n_bins = min(256, max(64, int(np.sqrt(n_samples / 50))))
                new_info_enhancer.compute_mutual_information_matrix(
                    use_fft=True,  # Always use FFT
                    fft_bins=n_bins
                )
                
                # Rank features and select top ones
                new_info_enhancer.rank_features_by_importance()
                top_indices = new_info_enhancer.select_top_features()
            
            selected_features = new_features[:, top_indices]
            
            # Apply any entropy weighting if used during training
            if hasattr(self.info_enhancer, 'entropy') and len(self.enhanced_feature_names) > len(top_indices):
                # Copy entropy from trained model if available
                new_info_enhancer.entropy = self.info_enhancer.entropy
                entropy_weights = new_info_enhancer.entropy[top_indices] / np.sum(new_info_enhancer.entropy[top_indices])
                entropy_weighted = selected_features * entropy_weights.reshape(1, -1)
                new_enhanced_features = np.column_stack([selected_features, entropy_weighted])
            else:
                # If entropy is needed but not available from trained model
                if len(self.enhanced_feature_names) > len(top_indices):
                    # Compute entropy for new data
                    print("Computing entropy for new data...")
                    new_info_enhancer.estimate_shannon_entropy()
                    entropy_weights = new_info_enhancer.entropy[top_indices] / np.sum(new_info_enhancer.entropy[top_indices])
                    entropy_weighted = selected_features * entropy_weights.reshape(1, -1)
                    new_enhanced_features = np.column_stack([selected_features, entropy_weighted])
                else:
                    new_enhanced_features = selected_features
        else:
            # If no enhancement was done, use all features
            new_enhanced_features = new_features
            
        # 3. Calculate distances from each new point to each existing regime centroid
        # Get the unique regime labels which may not be sequential
        unique_regimes = np.unique(self.regime_labels)
        regime_centroids = {}
        
        # Print information about feature dimensions
        print(f"Training features shape: {self.enhanced_features.shape}, New features shape: {new_enhanced_features.shape}")
        
        # Handle feature dimension mismatch with safer approach
        # First normalize each set of features separately to ensure consistency
        from sklearn.preprocessing import StandardScaler
        
        # Scale training features
        training_scaler = StandardScaler()
        scaled_training_features = training_scaler.fit_transform(self.enhanced_features)
        
        # Scale new features independently
        new_scaler = StandardScaler()
        scaled_new_features = new_scaler.fit_transform(new_enhanced_features)
        
        # Compute centroids from scaled training data
        for regime in unique_regimes:
            regime_points = np.where(self.regime_labels == regime)[0]
            if len(regime_points) > 0:
                regime_centroids[regime] = np.mean(scaled_training_features[regime_points], axis=0)
                
        # Check and adjust dimensions if needed
        min_dim = min(scaled_training_features.shape[1], scaled_new_features.shape[1])
        if scaled_training_features.shape[1] != scaled_new_features.shape[1]:
            print(f"Dimension mismatch: Training has {scaled_training_features.shape[1]} features, new data has {scaled_new_features.shape[1]} features")
            print(f"Using only the first {min_dim} dimensions for centroid distance calculation")
            
            # Trim centroids to consistent dimension
            for regime in regime_centroids:
                regime_centroids[regime] = regime_centroids[regime][:min_dim]
                
            # Trim new features to consistent dimension
            scaled_new_features = scaled_new_features[:, :min_dim]
        
        # Assign each new point to the nearest regime
        n_new_points = len(new_enhanced_features)
        new_regime_labels = np.zeros(n_new_points, dtype=int)
        confidences = []
        
        # Check if we have centroids for all regimes
        if len(regime_centroids) == 0:
            print("Warning: No valid regime centroids found. Using fallback assignment.")
            # Fallback: assign all to the most common regime in the training data
            most_common_regime = np.argmax(np.bincount(self.regime_labels))
            new_regime_labels.fill(most_common_regime)
            confidences = [1.0] * n_new_points
        else:
            # Calculate distances using scaled features
            for i in range(n_new_points):
                min_dist = float('inf')
                nearest_regime = unique_regimes[0]  # Default to first regime
                
                # Calculate distances to all centroids
                distances = {}
                for regime, centroid in regime_centroids.items():
                    # Use Euclidean distance with proper dimensions
                    dist = np.sqrt(np.sum((scaled_new_features[i] - centroid) ** 2))
                    distances[regime] = dist
                    
                    if dist < min_dist:
                        min_dist = dist
                        nearest_regime = regime
                
                new_regime_labels[i] = nearest_regime
                
                # Calculate confidence as ratio of distances
                sorted_dists = sorted(distances.values())
                if len(sorted_dists) > 1:
                    confidence = sorted_dists[1] / (sorted_dists[0] + 1e-10)  # Second closest / closest
                else:
                    confidence = 2.0  # High confidence if only one regime
                confidences.append(confidence)
            
            # Debug information
            print(f"Regime assignments: {np.unique(new_regime_labels, return_counts=True)}")
        
        # Add regime labels and confidence scores to the processed dataframe
        new_df_processed['regime'] = new_regime_labels
        new_df_processed['regime_confidence'] = confidences
        
        print(f"Labeled {n_new_points} new data points")
        
        # Merge regime labels back to original dataframe
        if set(new_df_processed.index) == set(new_df.index):
            result_df = new_df.copy()
            result_df['regime'] = new_df_processed['regime']
            result_df['regime_confidence'] = new_df_processed['regime_confidence']
        else:
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
        
        # Get regime mappings if available
        regime_to_idx = self.regime_analysis.get('regime_to_idx', {})
        idx_to_regime = self.regime_analysis.get('idx_to_regime', {})
        
        # Use mappings if available, otherwise assume sequential regimes
        use_mapping = len(regime_to_idx) > 0 and len(idx_to_regime) > 0
        
        # Get current regime (last observed point)
        current_regime = self.regime_labels[-1]
        
        # Initialize prediction
        predictions = [current_regime]
        
        # Simple Markov chain prediction
        for _ in range(steps_ahead):
            # Get index for current regime
            if use_mapping:
                current_idx = regime_to_idx.get(current_regime, 0)
            else:
                current_idx = current_regime
                
            # Get transition probabilities from current regime
            probs = transition_probs[current_idx]
            
            # If all zeros (no observed transitions), assume uniform distribution
            if np.sum(probs) == 0:
                probs = np.ones(n_regimes) / n_regimes
                
            # Sample next regime index based on transition probabilities
            next_idx = np.random.choice(n_regimes, p=probs)
            
            # Convert back to actual regime value if using mapping
            if use_mapping:
                next_regime = idx_to_regime.get(next_idx, next_idx)
            else:
                next_regime = next_idx
                
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

    def _apply_patterns_to_full_dataset(self, df, timestamp_col, price_col, volume_col, volatility_col):
        """
        Apply learned patterns to the full dataset.
        """
        print("Applying learned patterns to full dataset...")
        
        # Process the full dataset through the same pipeline
        full_analyzer = VolatilityRegimeAnalyzer(
            df=df,
            timestamp_col=timestamp_col,
            price_col=price_col,
            volume_col=volume_col,
            volatility_col=volatility_col
        )
        
        # Copy window_sizes to the new analyzer
        full_analyzer.window_sizes = self.window_sizes
        print(f"Using window sizes: {self.window_sizes}")
        
        # Use the same feature computation but skip MI computation
        print("Computing microstructure features...")
        full_analyzer.features, full_analyzer.feature_names, _, _ = full_analyzer.compute_features(window_sizes=self.window_sizes)
        
        # Apply feature enhancement without computing MI
        if hasattr(self, 'info_enhancer') and self.info_enhancer is not None:
            print("Enhancing features using FFT approximation for mutual information...")
            # Copy necessary attributes from trained model
            n_features = len(self.enhanced_feature_names) if hasattr(self, 'enhanced_feature_names') else 10
            
            if hasattr(self, 'enhanced_feature_names') and len(self.enhanced_feature_names) > 0 and '_ent_weighted' in self.enhanced_feature_names[0]:
                # If feature names contain _ent_weighted, we need to calculate the actual number of base features
                n_features = len([f for f in self.enhanced_feature_names if not ('_ent_weighted' in f or '_kl' in f)])
            
            # Create a new info enhancer but copy over feature importance from trained model
            full_analyzer.info_enhancer = InformationTheoryEnhancer(
                full_analyzer.features, 
                full_analyzer.feature_names
            )
            
            # Copy feature importance from trained model to avoid recomputing
            if hasattr(self.info_enhancer, 'feature_importance'):
                full_analyzer.info_enhancer.feature_importance = self.info_enhancer.feature_importance
            
            # Copy entropy from trained model if available
            if hasattr(self.info_enhancer, 'entropy'):
                full_analyzer.info_enhancer.entropy = self.info_enhancer.entropy
            else:
                # If entropy is needed but not available from trained model, compute it
                print("Computing entropy using FFT approximation...")
                full_analyzer.info_enhancer.estimate_shannon_entropy()
                
            # Check if mutual information is needed but not available from trained model
            mi_needed = False
            if hasattr(self, 'enhanced_feature_names'):
                for feat in self.enhanced_feature_names:
                    if '_ent_weighted' in feat:
                        mi_needed = True
                        break
                    
            if mi_needed and not hasattr(self.info_enhancer, 'mi_matrix'):
                print("Computing mutual information using FFT approximation for the full dataset...")
                # Always use FFT-based MI computation for the full dataset
                n_samples = full_analyzer.info_enhancer.n_samples
                n_bins = min(256, max(64, int(np.sqrt(n_samples / 50))))  # Adjusted for better approximation
                
                # Force using FFT for mutual information computation
                full_analyzer.info_enhancer.compute_mutual_information_matrix(
                    use_fft=True,  # Always use FFT
                    fft_bins=n_bins
                )
                
                # Rank features after computing MI
                full_analyzer.info_enhancer.rank_features_by_importance()
                
            # Enhance features
            full_analyzer.enhanced_features, full_analyzer.enhanced_feature_names = full_analyzer.info_enhancer.enhance_features(
                n_features=n_features,
                include_entropy=True,
                include_kl=False,  # Skip KL divergence for speed
                is_training=False  # Skip MI recomputation if already done
            )
        else:
            # Fallback if no info enhancer is available
            full_analyzer.enhanced_features = full_analyzer.features
            full_analyzer.enhanced_feature_names = full_analyzer.feature_names
        
        # Copy trained model components
        full_analyzer.regime_labels = self.regime_labels  # Use regime_labels, not regimes
        full_analyzer.tda = self.tda
        full_analyzer.regime_analysis = self.regime_analysis
        
        # Use the learned patterns to label the full dataset - use the same window_sizes for new data
        df_with_regimes = full_analyzer.label_new_data(df)
        
        return df_with_regimes 