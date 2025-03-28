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
                     window_sizes=None, n_regimes=4, training_batch=500000,
                     batch_size=100000, **kwargs):
        """
        Process data through the complete TDA pipeline to identify volatility regimes.
        Uses first training_batch observations for TDA analysis, then extends to full dataset.
        
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
        training_batch : int
            Number of initial observations to use for TDA analysis
        batch_size : int
            Size of batches for processing remaining data
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
        self.timestamp_col = timestamp_col
        self.price_col = price_col
        self.volume_col = volume_col
        self.volatility_col = volatility_col
        self.window_sizes = window_sizes if window_sizes is not None else [10, 50, 100]
        
        # Split data into training and extension parts
        n_samples = len(df)
        training_batch = min(training_batch, n_samples)  # Ensure we don't exceed data size
        
        print(f"\nUsing first {training_batch:,} observations for TDA analysis")
        training_df = df.iloc[:training_batch].copy()
        extension_df = df.iloc[training_batch:].copy() if training_batch < n_samples else None
        
        # Extract parameters for specific pipeline steps
        feature_params = kwargs.get('feature_params', {})
        enhancement_params = kwargs.get('enhancement_params', {})
        tda_params = kwargs.get('tda_params', {})
        
        # Step 1: Run TDA on training batch
        self.input_df = training_df
        
        # Extract microstructure features
        print("\n--- STEP 1: Extracting Microstructure Features ---")
        feature_start_time = time.time()
        self._extract_features(**feature_params)
        print(f"Feature extraction completed in {time.time() - feature_start_time:.2f} seconds")
        
        # Enhance features using information theory
        print("\n--- STEP 2: Enhancing Features with Information Theory ---")
        enhancement_start_time = time.time()
        self._enhance_features(**enhancement_params)
        print(f"Feature enhancement completed in {time.time() - enhancement_start_time:.2f} seconds")
        
        # Perform topological data analysis
        print("\n--- STEP 3: Performing Topological Data Analysis ---")
        tda_start_time = time.time()
        self._perform_tda_analysis(n_regimes=n_regimes, **tda_params)
        print(f"Topological analysis completed in {time.time() - tda_start_time:.2f} seconds")
        
        # Train XGBoost model for extension
        print("\n--- STEP 4: Training Regime Extension Model ---")
        extension_start_time = time.time()
        self._train_regime_extension_model()
        print(f"Extension model training completed in {time.time() - extension_start_time:.2f} seconds")
        
        # Initialize result DataFrame with training results
        result_df = df.copy()
        result_df.loc[:training_batch-1, 'regime'] = self.regimes
        
        # Extend to remaining data if any
        if extension_df is not None and len(extension_df) > 0:
            print(f"\n--- STEP 5: Extending Regimes to Remaining {len(extension_df):,} Observations ---")
            extension_start = time.time()
            
            # Process remaining data in batches
            n_remaining = len(extension_df)
            n_batches = (n_remaining + batch_size - 1) // batch_size
            print(f"Processing in {n_batches} batches of size {batch_size}")
            
            for batch_idx, start_idx in enumerate(range(0, n_remaining, batch_size), 1):
                batch_start = time.time()
                end_idx = min(start_idx + batch_size, n_remaining)
                batch_df = extension_df.iloc[start_idx:end_idx]
                
                print(f"\nProcessing batch {batch_idx}/{n_batches}")
                print(f"Batch range: {start_idx:,} to {end_idx:,} ({end_idx-start_idx:,} observations)")
                
                # Process batch
                batch_result = self._process_extension_batch(batch_df)
                
                # Store results
                global_start = training_batch + start_idx
                global_end = training_batch + end_idx
                result_df.loc[global_start:global_end-1, 'regime'] = batch_result['regime']
                result_df.loc[global_start:global_end-1, 'regime_confidence'] = batch_result['regime_confidence']
                
                # Print batch statistics
                regime_counts = np.bincount(batch_result['regime'])
                print(f"Batch processing time: {time.time() - batch_start:.2f} seconds")
                print("Batch regime distribution:")
                for regime, count in enumerate(regime_counts):
                    print(f"  Regime {regime}: {count:,} ({count/(end_idx-start_idx)*100:.2f}%)")
                print(f"Average confidence: {np.mean(batch_result['regime_confidence']):.4f}")
                
                # Print progress
                progress = end_idx / n_remaining * 100
                print(f"Overall progress: {progress:.1f}%")
            
            extension_time = time.time() - extension_start
            print(f"\nExtension completed in {extension_time:.2f} seconds ({extension_time/60:.2f} minutes)")
            print(f"Average processing time per batch: {extension_time/n_batches:.2f} seconds")
        
        total_time = time.time() - total_start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        # Print final regime distribution
        regime_counts = result_df['regime'].value_counts().sort_index()
        print("\nFinal regime distribution:")
        for regime, count in regime_counts.items():
            print(f"Regime {regime}: {count:,} ({count/len(result_df)*100:.2f}%)")
        
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
                             min_epsilon=None, max_epsilon=None, num_steps=10,
                             window_size=150, overlap=75, max_path_length=2,
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
        min_epsilon, max_epsilon : float or None
            Distance thresholds for network construction. If None, will be set adaptively.
        num_steps : int
            Number of steps in epsilon range
        window_size : int
            Size of sliding window for zigzag persistence (increased for efficiency)
        overlap : int
            Number of points to overlap between windows (50% of window_size)
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
        
        if hasattr(self, 'mutual_info_matrix') and self.mutual_info_matrix is not None:
            if self.mutual_info_matrix.shape[0] != len(self.enhanced_feature_names):
                # Transform MI matrix to match enhanced features
                feature_indices = [i for i, name in enumerate(self.feature_names) 
                                 if name in self.enhanced_feature_names]
                transformed_mi_matrix = self.mutual_info_matrix[feature_indices][:, feature_indices]
                print(f"Transformed MI matrix to match enhanced features: {transformed_mi_matrix.shape}")
        
        if hasattr(self, 'transfer_entropy') and self.transfer_entropy is not None:
            if self.transfer_entropy.shape[0] != len(self.enhanced_feature_names):
                # Transform TE matrix to match enhanced features
                feature_indices = [i for i, name in enumerate(self.feature_names) 
                                 if name in self.enhanced_feature_names]
                transformed_te_matrix = self.transfer_entropy[feature_indices][:, feature_indices]
                print(f"Transformed TE matrix to match enhanced features: {transformed_te_matrix.shape}")
        
        # Compute distance matrix with information theory enhancement
        self.tda.compute_temporally_weighted_distance(
            alpha=alpha,
            beta=beta,
            lambda_info=lambda_info,
            mi_matrix=transformed_mi_matrix,
            transfer_entropy=transformed_te_matrix
        )
        
        # Set adaptive epsilon range based on distance distribution
        if min_epsilon is None or max_epsilon is None:
            distances = self.tda.temporal_distance_matrix.flatten()
            distances = distances[distances > 0]  # Exclude zeros
            
            # Use more conservative percentiles for epsilon range
            min_epsilon = np.percentile(distances, 5)  # 5th percentile
            max_epsilon = np.percentile(distances, 85)  # 85th percentile (reduced from 95)
            print(f"Using adaptive epsilon range: [{min_epsilon:.4f}, {max_epsilon:.4f}]")
        
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
    
    def _train_regime_extension_model(self, **kwargs):
        """
        Train XGBoost model for extending regimes to full dataset.
        Uses enhanced features from TDA analysis for training.
        
        Parameters:
        -----------
        **kwargs : dict
            Additional parameters for XGBoost model
        """
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        
        print("\n--- STEP 4: Training XGBoost Model for Regime Extension ---")
        print("Preparing training data...")
        
        # Use enhanced features for training
        X = self.enhanced_features
        y = self.regimes
        
        # Split data for validation, maintaining temporal order
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of regimes: {len(np.unique(y))}")
        
        # Set XGBoost parameters
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y)),
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',  # For faster training
            'eval_metric': ['mlogloss', 'merror'],
            'verbosity': 1,  # Show training progress
            **kwargs
        }
        
        print("\nXGBoost parameters:")
        for param, value in xgb_params.items():
            print(f"  {param}: {value}")
        
        # Create DMatrix for XGBoost
        print("\nCreating DMatrix objects...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        print("\nStarting XGBoost training...")
        # Train model with early stopping
        self.extension_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=20,
            verbose_eval=10  # Print evaluation every 10 rounds
        )
        
        # Save feature names for prediction
        self.extension_feature_names = self.enhanced_feature_names
        
        # Print final model performance
        print("\nFinal model performance:")
        train_pred = np.argmax(self.extension_model.predict(dtrain), axis=1)
        val_pred = np.argmax(self.extension_model.predict(dval), axis=1)
        train_acc = np.mean(train_pred == y_train)
        val_acc = np.mean(val_pred == y_val)
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        
        print("\nXGBoost model training completed")
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
        import os
        
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
            'regimes': self.regimes,
            'extension_feature_names': self.extension_feature_names if hasattr(self, 'extension_feature_names') else None
        }
        
        # Save main model data
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        # Save XGBoost model separately if it exists
        if hasattr(self, 'extension_model'):
            xgb_path = os.path.splitext(filepath)[0] + '_xgb.model'
            self.extension_model.save_model(xgb_path)
            print(f"XGBoost model saved to {xgb_path}")
            
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
        import os
        import xgboost as xgb
        
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
        
        if model_data['extension_feature_names'] is not None:
            pipeline.extension_feature_names = model_data['extension_feature_names']
            
            # Try to load XGBoost model
            xgb_path = os.path.splitext(filepath)[0] + '_xgb.model'
            if os.path.exists(xgb_path):
                pipeline.extension_model = xgb.Booster()
                pipeline.extension_model.load_model(xgb_path)
                print(f"XGBoost model loaded from {xgb_path}")
        
        # If new data is provided, label it
        if df is not None:
            pipeline.input_df = df.copy()
            labeled_df = pipeline._extend_regimes_to_full_dataset(df)
            return pipeline, labeled_df
        
        return pipeline
    
    def _process_extension_batch(self, batch_df):
        """
        Process a batch of data for regime extension.
        
        Parameters:
        -----------
        batch_df : pandas.DataFrame
            Batch of data to process
            
        Returns:
        --------
        dict
            Dictionary containing regime labels and confidence scores
        """
        import xgboost as xgb
        
        # Extract features for this batch
        feature_engine = MicrostructureFeatureEngine(
            batch_df,
            timestamp_col=self.timestamp_col,
            price_col=self.price_col,
            volume_col=self.volume_col,
            volatility_col=self.volatility_col
        )
        
        # Extract and enhance features using same parameters as training
        batch_features, _, _ = feature_engine.extract_all_features(
            window_sizes=self.window_sizes
        )
        
        # Create info enhancer for feature enhancement
        info_enhancer = InformationTheoryEnhancer(
            batch_features,
            self.feature_names
        )
        
        # Enhance features using same parameters as training
        # Use histogram approximation for faster computation in batch processing
        batch_enhanced_features, _ = info_enhancer.enhance_features(
            n_features=len(self.extension_feature_names),
            is_training=False,  # Don't recompute MI matrices
            include_entropy=True,
            include_kl=False,  # Skip KL divergence for batch processing
            include_transfer_entropy=False  # Skip transfer entropy for batch processing
        )
        
        # Create DMatrix for prediction
        dbatch = xgb.DMatrix(batch_enhanced_features)
        
        # Get predictions and probabilities
        pred_probs = self.extension_model.predict(dbatch)
        predictions = np.argmax(pred_probs, axis=1)
        confidences = np.max(pred_probs, axis=1)
        
        return {
            'regime': predictions,
            'regime_confidence': confidences
        }
    
    def _extend_regimes_to_full_dataset(self, df, batch_size=100000):
        """
        Extend regimes to full dataset using trained XGBoost model.
        Processes data in batches to handle large datasets efficiently.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Full dataset to label
        batch_size : int
            Size of batches for processing
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with regime labels
        """
        import xgboost as xgb
        
        if not hasattr(self, 'extension_model'):
            raise ValueError("Extension model must be trained first")
            
        print(f"Extending regimes to {len(df):,} data points...")
        result_df = df.copy()
        n_samples = len(df)
        
        # Initialize arrays for predictions and confidences
        all_regimes = np.zeros(n_samples, dtype=int)
        all_confidences = np.zeros(n_samples)
        
        # Process in batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            print(f"Processing batch {start_idx:,} to {end_idx:,}")
            
            # Get batch data
            batch_df = df.iloc[start_idx:end_idx]
            
            # Process batch
            batch_result = self._process_extension_batch(batch_df)
            
            # Store results
            all_regimes[start_idx:end_idx] = batch_result['regime']
            all_confidences[start_idx:end_idx] = batch_result['regime_confidence']
            
            print(f"Batch regime distribution: {np.bincount(batch_result['regime'])}")
        
        # Add predictions to result DataFrame
        result_df['regime'] = all_regimes
        result_df['regime_confidence'] = all_confidences
        
        print("Regime extension completed")
        print(f"Final regime distribution: {np.bincount(all_regimes)}")
        
        return result_df 