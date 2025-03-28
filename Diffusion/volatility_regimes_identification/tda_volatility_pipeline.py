import numpy as np
import pandas as pd
import os
import time
from datetime import datetime

from .microstructure_features import MicrostructureFeatureEngine
from .information_theory import InformationTheoryEnhancer
from .topological_analyzer import TopologicalDataAnalyzer

class TDAVolatilityPipeline:
    """
    Integrated pipeline that combines microstructure feature extraction,
    information theory enhancement, and topological data analysis to
    identify volatility regimes in financial time series data.
    
    This pipeline streamlines the process from raw data to regime identification
    by integrating the three core components:
    1. MicrostructureFeatureEngine: Extracts comprehensive microstructure features
    2. InformationTheoryEnhancer: Enhances features using information theory
    3. TopologicalDataAnalyzer: Performs topological analysis to identify regimes
    """
    
    def __init__(self):
        """Initialize the TDA volatility pipeline."""
        self.feature_engine = None
        self.info_enhancer = None
        self.tda = None
        
        # Store data and feature information
        self.input_df = None
        self.features = None
        self.feature_names = None
        self.enhanced_features = None
        self.enhanced_feature_names = None
        
        # Store regime information
        self.regimes = None
        
        # Configuration parameters
        self.window_sizes = None
        self.timestamp_col = None
        self.price_col = None
        self.volume_col = None
        self.volatility_col = None
    
    def process_data(self, df, timestamp_col='Timestamp', price_col='Value', 
                     volume_col='Volume', volatility_col='Volatility',
                     window_sizes=None, n_regimes=4, **kwargs):
        """
        Process data through the complete TDA pipeline to identify volatility regimes.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with tick data
        timestamp_col : str
            Column name for timestamps
        price_col : str
            Column name for price values
        volume_col : str
            Column name for volume values
        volatility_col : str
            Column name for volatility values
        window_sizes : list
            List of window sizes for feature calculation
        n_regimes : int
            Number of regimes to identify
        **kwargs : dict
            Additional parameters to pass to specific pipeline components
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with original data plus regime labels
        """
        print("\nStarting TDA volatility regime pipeline at", datetime.now().strftime("%H:%M:%S"))
        total_start_time = time.time()
        
        # Store parameters
        self.input_df = df.copy()
        self.timestamp_col = timestamp_col
        self.price_col = price_col
        self.volume_col = volume_col
        self.volatility_col = volatility_col
        self.window_sizes = window_sizes if window_sizes is not None else [10, 50, 100]
        
        # Extract parameters for specific pipeline steps
        feature_params = kwargs.get('feature_params', {})
        enhancement_params = kwargs.get('enhancement_params', {})
        tda_params = kwargs.get('tda_params', {})
        
        # Step 1: Extract microstructure features
        print("\n--- STEP 1: Extracting Microstructure Features ---")
        feature_start_time = time.time()
        self._extract_features(**feature_params)
        print(f"Feature extraction completed in {time.time() - feature_start_time:.2f} seconds")
        
        # Step 2: Enhance features using information theory
        print("\n--- STEP 2: Enhancing Features with Information Theory ---")
        enhancement_start_time = time.time()
        self._enhance_features(**enhancement_params)
        print(f"Feature enhancement completed in {time.time() - enhancement_start_time:.2f} seconds")
        
        # Step 3: Perform topological data analysis to identify regimes
        print("\n--- STEP 3: Performing Topological Data Analysis ---")
        tda_start_time = time.time()
        self._perform_tda_analysis(n_regimes=n_regimes, **tda_params)
        print(f"Topological analysis completed in {time.time() - tda_start_time:.2f} seconds")
        
        # Create output DataFrame with regime labels
        result_df = self.input_df.copy()
        result_df['regime'] = self.regimes
        
        total_time = time.time() - total_start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        return result_df
    
    def _extract_features(self, **kwargs):
        """
        Extract microstructure features from tick data.
        
        Parameters:
        -----------
        **kwargs : dict
            Additional parameters to pass to feature extraction
        """
        # Initialize feature engine
        self.feature_engine = MicrostructureFeatureEngine(
            self.input_df,
            timestamp_col=self.timestamp_col,
            price_col=self.price_col,
            volume_col=self.volume_col,
            volatility_col=self.volatility_col
        )
        
        # Extract features
        self.features, self.feature_names, self.input_df = self.feature_engine.extract_all_features(
            window_sizes=self.window_sizes
        )
        
        print(f"Extracted {len(self.feature_names)} microstructure features")
        return self
    
    def _enhance_features(self, n_features=10, include_entropy=True, include_kl=True,
                         include_transfer_entropy=True, include_high_signal=True,
                         min_snr=2.0, **kwargs):
        """
        Enhance features using information theory.
        
        Parameters:
        -----------
        n_features : int
            Number of top features to include
        include_entropy : bool
            Whether to include entropy-weighted features
        include_kl : bool
            Whether to include KL divergence features
        include_transfer_entropy : bool
            Whether to include transfer entropy features
        include_high_signal : bool
            Whether to include high signal-to-noise features
        min_snr : float
            Minimum signal-to-noise ratio for high signal features
        **kwargs : dict
            Additional parameters to pass to feature enhancement
        """
        if self.features is None:
            raise ValueError("Features must be extracted first")
        
        # Initialize information theory enhancer
        self.info_enhancer = InformationTheoryEnhancer(
            self.features,
            self.feature_names
        )
        
        # Estimate entropy
        self.info_enhancer.estimate_shannon_entropy()
        
        # Compute mutual information
        self.info_enhancer.compute_mutual_information_matrix()
        
        # Compute transfer entropy if needed
        if include_transfer_entropy:
            self.info_enhancer.compute_transfer_entropy()
        
        # Enhance features
        self.enhanced_features, self.enhanced_feature_names = self.info_enhancer.enhance_features(
            n_features=n_features,
            include_entropy=include_entropy,
            include_kl=include_kl,
            include_transfer_entropy=include_transfer_entropy,
            include_high_signal=include_high_signal,
            min_snr=min_snr,
            is_training=True
        )
        
        print(f"Enhanced features: {len(self.enhanced_feature_names)} features selected")
        return self
    
    def _perform_tda_analysis(self, n_regimes=4, alpha=0.5, beta=0.1, lambda_info=1.0,
                             min_epsilon=0.1, max_epsilon=2.0, num_steps=10,
                             window_size=100, overlap=50, max_path_length=3,
                             **kwargs):
        """
        Perform topological data analysis to identify regimes.
        
        Parameters:
        -----------
        n_regimes : int
            Number of regimes to identify
        alpha : float
            Weight for temporal component in distance calculation
        beta : float
            Decay rate for temporal distance
        lambda_info : float
            Weight for information-theoretic component
        min_epsilon : float
            Minimum distance threshold for network construction
        max_epsilon : float
            Maximum distance threshold for network construction
        num_steps : int
            Number of steps in epsilon range
        window_size : int
            Size of sliding window for zigzag persistence
        overlap : int
            Number of points to overlap between windows
        max_path_length : int
            Maximum path length for path complex
        **kwargs : dict
            Additional parameters to pass to TDA analysis
        """
        if self.enhanced_features is None:
            raise ValueError("Features must be enhanced first")
        
        # Extract timestamps for temporal weighting
        timestamps = self.input_df[self.timestamp_col].values
        
        # Initialize topological data analyzer
        self.tda = TopologicalDataAnalyzer(
            self.enhanced_features,
            self.enhanced_feature_names,
            timestamp_array=timestamps
        )
        
        # Transform MI and TE matrices to match enhanced feature dimensions if needed
        transformed_mi_matrix = None
        transformed_te_matrix = None
        
        if hasattr(self.info_enhancer, 'mi_matrix') and self.info_enhancer.mi_matrix is not None:
            # Get the original MI matrix and top feature indices
            original_mi = self.info_enhancer.mi_matrix
            
            # Check if we have feature importance information to map indices
            if hasattr(self.info_enhancer, 'feature_importance') and self.info_enhancer.feature_importance is not None:
                # Get indices of the top features used in the enhanced feature set
                top_indices = self.info_enhancer.feature_importance['ranked_indices'][:len(self.enhanced_feature_names)]
                
                # Extract the submatrix corresponding to these top features
                # This creates a matrix where rows and columns correspond to the enhanced features
                top_mi = original_mi[np.ix_(top_indices, top_indices)]
                
                # Create the transformed MI matrix matching enhanced feature dimensions
                n_enhanced = len(self.enhanced_feature_names)
                n_base = len(top_indices)
                
                # Only use the base features part (not the derived features)
                if n_base <= n_enhanced:
                    transformed_mi_matrix = top_mi
                    print(f"Transformed MI matrix to match enhanced features: {transformed_mi_matrix.shape}")
                else:
                    print("Warning: Cannot properly transform MI matrix. Using None.")
            else:
                print("Warning: No feature importance information available to transform MI matrix")
                
        if hasattr(self.info_enhancer, 'transfer_entropy') and self.info_enhancer.transfer_entropy is not None:
            # Apply the same transformation to the transfer entropy matrix
            original_te = self.info_enhancer.transfer_entropy
            
            if hasattr(self.info_enhancer, 'feature_importance') and self.info_enhancer.feature_importance is not None:
                top_indices = self.info_enhancer.feature_importance['ranked_indices'][:len(self.enhanced_feature_names)]
                top_te = original_te[np.ix_(top_indices, top_indices)]
                
                n_enhanced = len(self.enhanced_feature_names)
                n_base = len(top_indices)
                
                if n_base <= n_enhanced:
                    transformed_te_matrix = top_te
                    print(f"Transformed TE matrix to match enhanced features: {transformed_te_matrix.shape}")
                else:
                    print("Warning: Cannot properly transform TE matrix. Using None.")
            else:
                print("Warning: No feature importance information available to transform TE matrix")
        
        # Compute temporally weighted distance
        self.tda.compute_temporally_weighted_distance(
            alpha=alpha,
            beta=beta,
            lambda_info=lambda_info,
            mi_matrix=transformed_mi_matrix,
            transfer_entropy=transformed_te_matrix
        )
        
        # Compute persistent path zigzag homology
        self.tda.compute_persistent_path_zigzag_homology(
            window_size=window_size,
            overlap=overlap,
            max_path_length=max_path_length,
            min_epsilon=min_epsilon,
            max_epsilon=max_epsilon,
            num_steps=num_steps
        )
        
        # Identify regimes
        self.regimes = self.tda.identify_regimes(n_regimes=n_regimes)
        
        print(f"Identified {n_regimes} volatility regimes")
        return self
    
    def save_model(self, filepath):
        """
        Save the pipeline model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        import pickle
        
        # Create model object with all necessary components
        model_data = {
            'window_sizes': self.window_sizes,
            'timestamp_col': self.timestamp_col,
            'price_col': self.price_col,
            'volume_col': self.volume_col,
            'volatility_col': self.volatility_col,
            'features': self.features,
            'feature_names': self.feature_names,
            'enhanced_features': self.enhanced_features,
            'enhanced_feature_names': self.enhanced_feature_names,
            'regimes': self.regimes
        }
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath, df=None):
        """
        Load a saved pipeline model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        df : pandas.DataFrame, optional
            New data to label with the loaded model
            
        Returns:
        --------
        TDAVolatilityPipeline
            Loaded pipeline instance
        """
        import pickle
        
        # Load model data
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new pipeline instance
        pipeline = cls()
        
        # Restore model attributes
        pipeline.window_sizes = model_data['window_sizes']
        pipeline.timestamp_col = model_data['timestamp_col']
        pipeline.price_col = model_data['price_col']
        pipeline.volume_col = model_data['volume_col']
        pipeline.volatility_col = model_data['volatility_col']
        pipeline.features = model_data['features']
        pipeline.feature_names = model_data['feature_names']
        pipeline.enhanced_features = model_data['enhanced_features']
        pipeline.enhanced_feature_names = model_data['enhanced_feature_names']
        pipeline.regimes = model_data['regimes']
        
        # If new data is provided, label it
        if df is not None:
            pipeline.input_df = df.copy()
            labeled_df = pipeline.label_new_data(df)
            return pipeline, labeled_df
        
        return pipeline
    
    def label_new_data(self, new_df):
        """
        Label new data with the identified regimes.
        
        Parameters:
        -----------
        new_df : pandas.DataFrame
            New data to label with the identified regimes
            
        Returns:
        --------
        pandas.DataFrame
            New data with regime labels
        """
        if self.regimes is None or self.enhanced_features is None:
            raise ValueError("Pipeline must be trained before labeling new data")
        
        print("Labeling new data with identified regimes...")
        
        # Process new data through the same pipeline
        # Step 1: Extract features
        feature_engine = MicrostructureFeatureEngine(
            new_df,
            timestamp_col=self.timestamp_col,
            price_col=self.price_col,
            volume_col=self.volume_col,
            volatility_col=self.volatility_col
        )
        
        new_features, new_feature_names, new_df = feature_engine.extract_all_features(
            window_sizes=self.window_sizes
        )
        
        # Step 2: Enhance features using the same approach as training
        # Create a new info enhancer but copy over feature importance from trained model
        new_info_enhancer = InformationTheoryEnhancer(
            new_features, 
            new_feature_names
        )
        
        # Copy feature importance from trained model
        if hasattr(self, 'info_enhancer') and hasattr(self.info_enhancer, 'feature_importance'):
            new_info_enhancer.feature_importance = self.info_enhancer.feature_importance
        
        # Determine parameters from trained model
        include_entropy = any('_ent_weighted' in f for f in self.enhanced_feature_names)
        include_kl = any('_kl' in f for f in self.enhanced_feature_names)
        include_transfer_entropy = any('_te_weighted' in f for f in self.enhanced_feature_names)
        include_high_signal = any('_high_signal' in f for f in self.enhanced_feature_names)
        
        # Estimate entropy
        new_info_enhancer.estimate_shannon_entropy()
        
        # Compute mutual information using FFT for faster processing
        new_info_enhancer.compute_mutual_information_matrix(
            use_fft=True,
            fast_approximation=True
        )
        
        # Compute transfer entropy if needed
        if include_transfer_entropy:
            new_info_enhancer.compute_transfer_entropy()
        
        # Enhance features
        n_features = len(self.enhanced_feature_names)
        new_enhanced_features, _ = new_info_enhancer.enhance_features(
            n_features=n_features,
            include_entropy=include_entropy,
            include_kl=include_kl, 
            include_transfer_entropy=include_transfer_entropy,
            include_high_signal=include_high_signal,
            is_training=False
        )
        
        # Step 3: Use the trained model to label new data
        # We'll use k-nearest neighbors for assigning regimes to new points
        from sklearn.neighbors import KNeighborsClassifier
        
        # Train a k-NN classifier on the original data
        knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
        knn.fit(self.enhanced_features, self.regimes)
        
        # Predict regimes for new data
        new_regimes = knn.predict(new_enhanced_features)
        
        # Get prediction probabilities for confidence scores
        new_regime_probs = knn.predict_proba(new_enhanced_features)
        new_confidence = np.max(new_regime_probs, axis=1)
        
        # Add regime and confidence to new data
        new_df['regime'] = new_regimes
        new_df['regime_confidence'] = new_confidence
        
        print(f"Labeled {len(new_df)} data points with regimes")
        return new_df 