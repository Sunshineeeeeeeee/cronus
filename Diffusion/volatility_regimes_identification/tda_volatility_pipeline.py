import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from ripser import ripser
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from xgboost import DMatrix, XGBClassifier, train as xgb_train

from .information_theory import InformationTheoryEnhancer
from .topological_analyzer import TopologicalDataAnalyzer
from .microstructure_features import MicrostructureFeatureEngine

class TDAVolatilityPipeline:
    """
    Pipeline for identifying volatility regimes using topological data analysis.
    
    This pipeline combines microstructure feature extraction, information theory,
    and topological data analysis to identify market volatility regimes.
    """
    
    def __init__(self, input_df=None, timestamp_col='timestamp', price_col='price', 
                volume_col='volume', verbose=True):
        """
        Initialize the pipeline.
        
        Parameters:
        -----------
        input_df : pandas.DataFrame or None
            Input dataframe with market data
        timestamp_col : str
            Name of timestamp column
        price_col : str
            Name of price column
        volume_col : str
            Name of volume column
        verbose : bool
            Whether to print verbose output
        """
        self.input_df = input_df
        self.timestamp_col = timestamp_col
        self.price_col = price_col
        self.volume_col = volume_col
        self.verbose = verbose
        
        # Initialize components
        self.feature_engine = None
        self.info_enhancer = None
        self.tda = None
        
        # Data containers
        self.features = None
        self.feature_names = None
        self.enhanced_features = None
        self.enhanced_feature_names = None
        self.regimes = None
        self.extension_model = None
        
        # Initialize any necessary directories
        os.makedirs('output', exist_ok=True)
        
        if verbose:
            print("TDA Volatility Pipeline initialized")
        
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
        print("\nStarting TDA volatility regime pipeline at", time.strftime("%H:%M:%S"))
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
        
        # Add timestamps to TDA params if not already present
        if 'timestamps' not in tda_params:
            tda_params['timestamps'] = training_df[timestamp_col].values
        
        # Train the pipeline on the training batch
        self.train(
            input_df=training_df,
            timestamp_col=timestamp_col,
            price_col=price_col, 
            volume_col=volume_col,
            n_regimes=n_regimes,
            feature_params=feature_params,
            enhancement_params=enhancement_params,
            **tda_params
        )
        
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
            
            # Initialize arrays for aggregating results
            all_regimes = np.zeros(n_remaining, dtype=int)
            all_confidences = np.zeros(n_remaining)
            
            for batch_idx, start_idx in enumerate(range(0, n_remaining, batch_size), 1):
                batch_start = time.time()
                end_idx = min(start_idx + batch_size, n_remaining)
                batch_df = extension_df.iloc[start_idx:end_idx]
                
                print(f"\nProcessing batch {batch_idx}/{n_batches}")
                print(f"Batch range: {start_idx:,} to {end_idx:,} ({end_idx-start_idx:,} observations)")
                
                # Process batch
                batch_result = self._process_extension_batch(batch_df)
                
                # Store results in arrays
                all_regimes[start_idx:end_idx] = batch_result['regimes']
                all_confidences[start_idx:end_idx] = batch_result['confidences']
                
                # Print batch statistics
                batch_time = time.time() - batch_start
                samples_per_sec = (end_idx - start_idx) / batch_time
                print(f"Batch completed in {batch_time:.2f} seconds ({samples_per_sec:.1f} samples/sec)")
                print(f"Batch regime distribution: {np.bincount(batch_result['regimes'])}")
            
            # Update result DataFrame with batch results
            result_df.loc[training_batch:, 'regime'] = all_regimes
            result_df.loc[training_batch:, 'regime_confidence'] = all_confidences
            
            print(f"Extension completed in {time.time() - extension_start:.2f} seconds")
            print(f"Final regime distribution: {np.bincount(all_regimes)}")
        
        # Report total time
        total_time = time.time() - total_start_time
        print(f"\nTDA volatility regime pipeline completed in {total_time:.2f} seconds")
        print(f"Processed {len(df):,} total observations at {len(df)/total_time:.1f} samples/sec")
        
        return result_df
    
    def _extract_features(self, window_sizes=None, normalize=True, include_original=True,
                         min_periods=None, df=None, **kwargs):
        """
        Extract microstructure features from input data.
        
        Parameters:
        -----------
        window_sizes : list or None
            List of window sizes to use for feature extraction
        normalize : bool
            Whether to normalize features
        include_original : bool
            Whether to include original price and volume
        min_periods : int or None
            Minimum number of observations required for rolling calculations
        df : pandas.DataFrame or None
            Input dataframe with market data (if None, uses self.input_df)
        **kwargs : dict
            Additional parameters to pass to feature engine
        """
        # Use provided df or fall back to self.input_df
        input_df = df if df is not None else self.input_df
        
        if input_df is None:
            raise ValueError("Input dataframe must be provided either through df parameter or self.input_df")
            
        # Set default window sizes if not provided
        if window_sizes is None:
            window_sizes = self.window_sizes if self.window_sizes is not None else [10, 50, 100]
            
        # Set minimum periods if not provided
        if min_periods is None:
            min_periods = {size: max(3, size // 5) for size in window_sizes}
        
        # Initialize feature engine if needed
        if self.feature_engine is None:
            self.feature_engine = MicrostructureFeatureEngine(
                #data=input_df,
                timestamp_col=self.timestamp_col,
                price_col=self.price_col,
                volume_col=self.volume_col
            )
            
        print(f"Extracting microstructure features with window sizes: {window_sizes}")
        
        # Extract features
        start_time = time.time()
        feature_df = self.feature_engine.extract_features(
            input_df,
            window_sizes=window_sizes,
            normalize=normalize,
            include_original=include_original,
            min_periods=min_periods,
            **kwargs
        )
        
        # Store features for further analysis
        self.features = feature_df.values
        self.feature_names = feature_df.columns.tolist()
        
        print(f"Extracted {len(self.feature_names)} features from {len(input_df)} observations")
        print(f"Feature extraction time: {time.time() - start_time:.2f} seconds")
        
        # Return for convenient chaining
        return self
    
    def _enhance_features(self, n_components=10, use_clustering=True, use_log=True,
                         use_mi=True, use_te=True, use_entropy=True,
                         lambda_ent=0.5, bins=10, **kwargs):
        """
        Enhance features using information theory.
        
        Parameters:
        -----------
        n_components : int
            Number of components to extract
        use_clustering : bool
            Whether to use clustering for feature enhancement
        use_log : bool
            Whether to use log transformation
        use_mi : bool
            Whether to use mutual information
        use_te : bool
            Whether to use transfer entropy
        use_entropy : bool
            Whether to use entropy weighting
        lambda_ent : float
            Weight for entropy in weighting scheme
        bins : int
            Number of bins for entropy estimation
        """
        if self.features is None or self.feature_names is None:
            raise ValueError("Features and feature names must be set before enhancement")
        
        # Initialize information theory enhancer if needed
        if self.info_enhancer is None:
            # Create a new enhancer
            self.info_enhancer = InformationTheoryEnhancer(
                lambda_ent=lambda_ent,
                bins=bins
            )
            # Initialize the enhancer with our data
            self.info_enhancer.setup(
                self.features,
                self.feature_names
            )
            
        # Determine whether to use progressive feature selection
        use_progressive_selection = len(self.feature_names) > 30
        
        if use_progressive_selection:
            print(f"\n=== Using Progressive Feature Selection: {len(self.feature_names)} features ===")
            # Create a ProgressiveFeatureSelector instance
            from .information_theory import ProgressiveFeatureSelector
            feature_selector = ProgressiveFeatureSelector(
                features=self.features,
                feature_names=self.feature_names
            )
            
            # Perform progressive feature selection
            selected_indices = feature_selector.select_features(
                n_features=min(30, len(self.feature_names)), 
                include_redundancy=True
            )
            
            # Filter features and names
            selected_features = self.features[:, selected_indices]
            selected_feature_names = [self.feature_names[i] for i in selected_indices]
            
            print(f"Selected {len(selected_indices)} features from {len(self.feature_names)}")
            print(f"Selected features: {selected_feature_names}")
            
            # Set these as the new features
            self.features = selected_features
            self.feature_names = selected_feature_names
            
            # Re-initialize enhancer with selected features
            self.info_enhancer.setup(self.features, self.feature_names)
            
        # Enhance features with information theory
        start_time = time.time()
        
        print("Enhancing features with information theory...")
        self.enhanced_features, self.enhanced_feature_names = self.info_enhancer.enhance_features(
            self.features,
            self.feature_names,
            use_entropy=use_entropy,
            use_mi=use_mi,
            use_te=use_te,
            use_clustering=use_clustering,
            n_clusters=n_components,
            use_log=use_log
        )
        
        print(f"Feature enhancement took {time.time() - start_time:.2f} seconds")
        print(f"Enhanced features shape: {self.enhanced_features.shape}")
        print(f"Enhanced feature names: {self.enhanced_feature_names}")
        
        # Return for convenient chaining
        return self
    
    def _perform_tda_analysis(self, n_regimes=3, alpha=0.5, beta=0.1, lambda_info=1.0,
                             min_epsilon=None, max_epsilon=None, num_steps=10,
                             window_size=150, overlap=75, max_path_length=2,
                             optimize_epsilon=True, **kwargs):
        """
        Perform topological data analysis on the enhanced features.
        
        Parameters:
        -----------
        n_regimes : int
            Number of regimes to identify
        alpha : float
            Weight for temporal component in distance calculation
        beta : float
            Decay rate for temporal distance
        lambda_info : float
            Weight for information theory enhancement
        min_epsilon, max_epsilon : float or None
            Distance thresholds for network construction
        num_steps : int
            Number of steps in epsilon range
        window_size : int
            Size of sliding window for zigzag persistence
        overlap : int
            Number of points to overlap between windows
        max_path_length : int
            Maximum path length for path complex
        optimize_epsilon : bool
            Whether to optimize epsilon threshold using information theory
        """
        # Create timestamps if not provided
        timestamps = kwargs.get('timestamps', None)
        if timestamps is None:
            timestamps = np.arange(len(self.enhanced_features))
        
        # Convert pandas DatetimeIndex to numpy array if needed
        if hasattr(timestamps, 'to_numpy'):
            timestamps = timestamps.to_numpy()
        
        # Initialize topological data analyzer
        self.tda = TopologicalDataAnalyzer(
            feature_array=self.enhanced_features,
            feature_names=self.enhanced_feature_names,
            timestamp_array=timestamps
        )
        
        # Transform MI and TE matrices to match enhanced features
        # We need to create a mapping from original features to enhanced features
        transformed_mi_matrix = None
        transformed_te_matrix = None
        
        if hasattr(self, 'info_enhancer') and self.info_enhancer is not None:
            # Check if mutual information matrix exists
            if hasattr(self.info_enhancer, 'mi_matrix') and self.info_enhancer.mi_matrix is not None:
                # Original MI matrix shape
                original_mi = self.info_enhancer.mi_matrix
                print(f"Original MI matrix shape: {original_mi.shape}")
                print(f"Enhanced feature count: {len(self.enhanced_feature_names)}")
                
                # Check if the dimensions match (before enhancement)
                if original_mi.shape[0] == len(self.feature_names):
                    # Get indices of original features that correspond to enhanced features
                    matching_indices = []
                    for name in self.enhanced_feature_names:
                        # Skip entropy-weighted features (second half of enhanced features)
                        if '_ent_weighted' in name or '_te_weighted' in name:
                            continue
                        
                        # Find index in original features
                        try:
                            idx = self.feature_names.index(name)
                            matching_indices.append(idx)
                        except ValueError:
                            continue
                    
                    # Create transformed MI matrix with only matching features
                    if matching_indices:
                        # Select subset of original MI matrix
                        transformed_mi_matrix = original_mi[np.ix_(matching_indices, matching_indices)]
                        
                        # Duplicate rows/columns for entropy-weighted features
                        full_size = len(self.enhanced_feature_names)
                        full_matrix = np.zeros((full_size, full_size))
                        
                        # Fill in values from original matrix
                        n_base = len(matching_indices)
                        full_matrix[:n_base, :n_base] = transformed_mi_matrix
                        
                        # Copy values for derived features
                        if n_base < full_size // 2:
                            # For entropy-weighted features, copy the MI from original features
                            full_matrix[n_base:2*n_base, n_base:2*n_base] = transformed_mi_matrix
                            full_matrix[:n_base, n_base:2*n_base] = transformed_mi_matrix
                            full_matrix[n_base:2*n_base, :n_base] = transformed_mi_matrix
                        
                        transformed_mi_matrix = full_matrix
                        print(f"Found {len(matching_indices)} matching features in original matrix")
                        print(f"Created transformed MI matrix of shape {transformed_mi_matrix.shape}")
                        print(f"MI value range: [{transformed_mi_matrix.min():.4f}, {transformed_mi_matrix.max():.4f}]")
                    else:
                        print("Error: No matching features found for MI matrix transformation")
                else:
                    print(f"Error: MI matrix shape ({original_mi.shape}) doesn't match feature count ({len(self.feature_names)})")
            
            # Check if transfer entropy matrix exists
            if hasattr(self.info_enhancer, 'transfer_entropy') and self.info_enhancer.transfer_entropy is not None:
                # Original TE matrix shape
                original_te = self.info_enhancer.transfer_entropy
                print(f"Original TE matrix shape: {original_te.shape}")
                
                # Check if the dimensions match (before enhancement)
                if original_te.shape[0] == len(self.feature_names):
                    # Get indices of original features that correspond to enhanced features
                    matching_indices = []
                    for name in self.enhanced_feature_names:
                        # Skip entropy-weighted features (second half of enhanced features)
                        if '_ent_weighted' in name or '_te_weighted' in name:
                            continue
                        
                        # Find index in original features
                        try:
                            idx = self.feature_names.index(name)
                            matching_indices.append(idx)
                        except ValueError:
                            continue
                    
                    # Create transformed TE matrix with only matching features
                    if matching_indices:
                        # Select subset of original TE matrix
                        transformed_te_matrix = original_te[np.ix_(matching_indices, matching_indices)]
                        
                        # Duplicate rows/columns for entropy-weighted features
                        full_size = len(self.enhanced_feature_names)
                        full_matrix = np.zeros((full_size, full_size))
                        
                        # Fill in values from original matrix
                        n_base = len(matching_indices)
                        full_matrix[:n_base, :n_base] = transformed_te_matrix
                        
                        # Copy values for derived features
                        if n_base < full_size // 2:
                            # For entropy-weighted features, copy the TE from original features
                            full_matrix[n_base:2*n_base, n_base:2*n_base] = transformed_te_matrix
                            full_matrix[:n_base, n_base:2*n_base] = transformed_te_matrix
                            full_matrix[n_base:2*n_base, :n_base] = transformed_te_matrix
                        
                        transformed_te_matrix = full_matrix
                        print(f"Found {len(matching_indices)} matching features in original TE matrix")
                        print(f"Created transformed TE matrix of shape {transformed_te_matrix.shape}")
                        print(f"TE value range: [{transformed_te_matrix.min():.4f}, {transformed_te_matrix.max():.4f}]")
                    else:
                        print("Error: No matching features found for TE matrix transformation")
                else:
                    print(f"Error: TE matrix shape ({original_te.shape}) doesn't match feature count ({len(self.feature_names)})")
        
        # If we couldn't transform the matrices, print a message and continue without them
        if transformed_mi_matrix is None:
            print("No MI matrix transformation needed; using default values")
        if transformed_te_matrix is None:
            print("No TE matrix transformation needed; using default values")
        
        # Compute distance matrix with information theory enhancement
        self.tda.compute_temporally_weighted_distance(
            alpha=alpha,
            beta=beta,
            lambda_info=lambda_info,
            mi_matrix=transformed_mi_matrix,
            transfer_entropy=transformed_te_matrix
        )
        
        # Find volatility feature index if available
        target_idx = None
        
        # First check if we have a volatility feature in the enhanced features
        for i, name in enumerate(self.enhanced_feature_names):
            if 'volatil' in name.lower():
                target_idx = i
                print(f"Found volatility in enhanced features: {name}")
                break
        
        # If not found, check for other target candidates
        if target_idx is None:
            # Check for price, value, etc.
            for keyword in ['price', 'value', 'return', 'vol']:
                for i, name in enumerate(self.enhanced_feature_names):
                    if keyword.lower() in name.lower():
                        target_idx = i
                        print(f"Using {name} as target for epsilon optimization")
                        break
                if target_idx is not None:
                    break
            
            # If still not found, check if we have volatility column in original features
            if target_idx is None and hasattr(self, 'volatility_col') and self.volatility_col is not None:
                if self.volatility_col in self.feature_names:
                    # Find closest related feature in enhanced features
                    vol_idx = self.feature_names.index(self.volatility_col)
                    vol_name = self.feature_names[vol_idx]
                    print(f"Found volatility in original features: {vol_name}")
                    # Look for any enhanced feature that might be related
                    for i, name in enumerate(self.enhanced_feature_names):
                        if any(x in name.lower() for x in ['vol', 'price', 'value']):
                            target_idx = i
                            print(f"Using {name} as target for epsilon optimization (closest to {vol_name})")
                            break
        
        # Set epsilon thresholds based on optimization or default approach
        if min_epsilon is None or max_epsilon is None or optimize_epsilon:
            if optimize_epsilon:
                print("\n=== Using Information-Theoretic Epsilon Optimization ===")
                # Use the TDA optimizer directly
                _, min_epsilon, max_epsilon, _ = self.tda.optimize_epsilon_threshold(
                    target_index=target_idx,
                    n_trials=10,
                    min_percentile=5,
                    max_percentile=90
                )
                print("========================================================\n")
            else:
                # Use simple distance-based approach
                distances = self.tda.temporal_distance_matrix.flatten()
                distances = distances[distances > 0]  # Exclude zeros
                
                min_epsilon = np.percentile(distances, 5)  # 5th percentile
                max_epsilon = np.percentile(distances, 85)  # 85th percentile
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
    
    def _calculate_persistence_diagrams(self, max_dim=2):
        """Calculate persistence diagrams from the enhanced features."""
        self.persistence_diagrams = ripser(self.enhanced_features, maxdim=max_dim)["dgms"]
        return self.persistence_diagrams
    
    def _estimate_confidence(self, proba, regimes):
        """
        Estimate model confidence based on prediction probability.
        Combines with persistence stability estimates.
        """
        if self.tda and hasattr(self.tda, 'regime_stability'):
            # Combine model confidence with topological stability
            confidence = 0.7 * proba + 0.3 * self.tda.regime_stability[regimes]
        else:
            confidence = proba
        
        return confidence
    
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
            Batch processing results
        """
        import xgboost as xgb
        
        # Create feature engine with the batch data
        feature_engine = MicrostructureFeatureEngine(
            timestamp_col=self.timestamp_col,
            price_col=self.price_col,
            volume_col=self.volume_col,
            volatility_col=self.volatility_col if hasattr(self, 'volatility_col') else None
        )
        
        # Extract features directly using extract_features method
        print(f"Extracting features from batch with {len(batch_df)} observations...")
        feature_df = feature_engine.extract_features(
            tick_data=batch_df,  # Pass batch_df as tick_data parameter
            window_sizes=self.window_sizes,
            normalize=True,
            include_original=True
        )
        
        # Convert to numpy arrays
        batch_features = feature_df.values
        batch_feature_names = feature_df.columns.tolist()
        
        print(f"Extracted {batch_features.shape[1]} features from batch")
        
        # Determine if we need to use adaptive enhancement
        # Only use adaptive enhancement if the original training used it and the batch features are similar
        use_adaptive = False
        if (hasattr(self, 'info_enhancer') and self.info_enhancer is not None and 
            hasattr(self, 'extension_model') and self.extension_model is not None):
            # We have an enhancer and a model, but need to check if it's safe to use
            use_adaptive = set(self.feature_names).issubset(set(batch_feature_names))
        
        # Print info about processing method
        print(f"Feature count: {batch_features.shape[1]}")
        print(f"Using {'adaptive' if use_adaptive else 'standard'} enhancement")
        
        # Ensure we have model_features attribute set
        if not hasattr(self, 'model_features'):
            self.model_features = self.enhanced_features.shape[1] if hasattr(self, 'enhanced_features') else batch_features.shape[1]
            print(f"Setting model_features to {self.model_features}")
            
        # For processing batches, it's safer to just use the model feature count directly
        # This avoids issues with feature selection and entropy calculations
        if use_adaptive:
            try:
                # Try adaptive enhancement with safety checks
                print("Creating batch-optimized feature set...")
                
                # Match batch features to the original features used in training
                matching_indices = []
                original_feature_set = set(self.feature_names)
                
                # First, find indices that match the original feature set
                for i, name in enumerate(batch_feature_names):
                    if name in original_feature_set:
                        matching_indices.append(i)
                
                # If we don't have enough matching features, append some additional ones
                if len(matching_indices) < self.model_features // 2:
                    print(f"Warning: Only found {len(matching_indices)} matching features from training set")
                    # Add more features to reach at least half the model dimensions
                    remaining = list(set(range(len(batch_feature_names))) - set(matching_indices))
                    matching_indices.extend(remaining[:max(0, (self.model_features // 2) - len(matching_indices))])
                
                # Take only the first model_features indices (or fewer if not enough available)
                matching_indices = matching_indices[:min(len(matching_indices), self.model_features)]
                print(f"Selected {len(matching_indices)} features based on original training features")
                
                # Create a selected feature array
                batch_selected = batch_features[:, matching_indices]
                selected_names = [batch_feature_names[i] for i in matching_indices]
                
                # If we need additional features, we'll add zeros (padding)
                if batch_selected.shape[1] < self.model_features:
                    padding_size = self.model_features - batch_selected.shape[1]
                    print(f"Adding {padding_size} padding features to match model dimensions")
                    padding = np.zeros((batch_selected.shape[0], padding_size))
                    batch_enhanced = np.hstack([batch_selected, padding])
                    enhanced_names = selected_names + [f"padding_{i}" for i in range(padding_size)]
                else:
                    batch_enhanced = batch_selected
                    enhanced_names = selected_names
                
            except Exception as e:
                print(f"Error during adaptive enhancement: {e}")
                print("Falling back to standard feature selection")
                use_adaptive = False
                
        if not use_adaptive:
            # Standard approach: just select the top features up to model_features
            top_indices = list(range(min(self.model_features, batch_features.shape[1])))
            batch_enhanced = batch_features[:, top_indices]
            enhanced_names = [batch_feature_names[i] for i in top_indices]
            
            # If we need additional features, we'll add zeros (padding)
            if batch_enhanced.shape[1] < self.model_features:
                padding_size = self.model_features - batch_enhanced.shape[1]
                print(f"Adding {padding_size} padding features to match model dimensions")
                padding = np.zeros((batch_enhanced.shape[0], padding_size))
                batch_enhanced = np.hstack([batch_enhanced, padding])
                enhanced_names = enhanced_names + [f"padding_{i}" for i in range(padding_size)]
        
        # Final dimension check
        if batch_enhanced.shape[1] != self.model_features:
            print(f"Warning: Dimension mismatch. Expected {self.model_features}, got {batch_enhanced.shape[1]}")
            # Force the correct dimensions as a last resort
            if batch_enhanced.shape[1] > self.model_features:
                batch_enhanced = batch_enhanced[:, :self.model_features]
            else:
                padding = np.zeros((batch_enhanced.shape[0], self.model_features - batch_enhanced.shape[1]))
                batch_enhanced = np.hstack([batch_enhanced, padding])
        
        # Create DMatrix for prediction
        dmatrix = xgb.DMatrix(batch_enhanced)
        
        # Get predictions
        regime_probs = self.extension_model.predict(dmatrix)
        regimes = np.argmax(regime_probs, axis=1)
        confidences = np.max(regime_probs, axis=1)
        
        # Return results
        return {
            'regimes': regimes,
            'confidences': confidences,
            'probabilities': regime_probs
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
            all_regimes[start_idx:end_idx] = batch_result['regimes']
            all_confidences[start_idx:end_idx] = batch_result['confidences']
            
            print(f"Batch regime distribution: {np.bincount(batch_result['regimes'])}")
        
        # Add predictions to result DataFrame
        result_df['regime'] = all_regimes
        result_df['regime_confidence'] = all_confidences
        
        print("Regime extension completed")
        print(f"Final regime distribution: {np.bincount(all_regimes)}")
        
        return result_df
    
    def train(self, features=None, feature_names=None, 
             n_regimes=3, alpha=0.5, beta=0.1, lambda_info=1.0,
             min_epsilon=0.1, max_epsilon=5.0, num_steps=10,
             window_size=100, overlap=50, max_path_length=2,
             optimize_epsilon=True, model_type='xgboost', train_size=0.8,
             input_df=None, timestamp_col=None, price_col=None, volume_col=None,
             feature_params=None, enhancement_params=None, **kwargs):
        """
        Train the volatility regime pipeline with extracted features.
        
        Parameters:
        -----------
        features : numpy.ndarray or None
            Extracted features (will extract if None)
        feature_names : list or None
            Names of features (will generate if None)
        n_regimes : int
            Number of volatility regimes to identify
        alpha : float
            Weight for temporal component in distance calculation
        beta : float
            Decay rate for temporal component
        lambda_info : float
            Weight for information-theoretic enhancement
        min_epsilon : float
            Minimum epsilon for filtration
        max_epsilon : float
            Maximum epsilon for filtration
        num_steps : int
            Number of filtration steps
        window_size : int
            Size of sliding window for path persistence
        overlap : int
            Overlap between consecutive windows
        max_path_length : int
            Maximum length of paths to consider
        optimize_epsilon : bool
            Whether to use information-theoretic optimization for epsilon
        model_type : str
            Type of model for regime extension ('xgboost' or 'lightgbm')
        train_size : float
            Proportion of data to use for training
        input_df : pandas.DataFrame or None
            Input dataframe with market data
        timestamp_col, price_col, volume_col : str or None
            Column names for timestamp, price, and volume
        feature_params, enhancement_params : dict or None
            Parameters for feature extraction and enhancement
        """
        # Set timestamp, price, and volume columns if provided
        if timestamp_col is not None:
            self.timestamp_col = timestamp_col
        if price_col is not None:
            self.price_col = price_col
        if volume_col is not None:
            self.volume_col = volume_col
        
        # Extract features if needed
        if input_df is not None:
            feature_params = feature_params or {}
            self._extract_features(
                df=input_df,
                **feature_params
            )
        elif features is not None and feature_names is not None:
            self.features = features
            self.feature_names = feature_names
        else:
            raise ValueError("Either input_df or (features and feature_names) must be provided")
        
        # Set default enhancement parameters
        default_enhancement_params = {
            'n_components': kwargs.get('n_components', 10),
            'use_clustering': kwargs.get('use_clustering', True),
            'use_log': kwargs.get('use_log', True),
            'use_mi': kwargs.get('use_mi', True),
            'use_te': kwargs.get('use_te', True),
            'use_entropy': kwargs.get('use_entropy', True),
            'lambda_ent': kwargs.get('lambda_ent', 0.5),
            'bins': kwargs.get('bins', 10)
        }
        enhancement_params = {**default_enhancement_params, **(enhancement_params or {})}
        
        # Enhance features
        self._enhance_features(**enhancement_params)
        
        # Store the number of features for later batch processing
        self.model_features = self.enhanced_features.shape[1]
        self.extension_feature_names = self.enhanced_feature_names

        # Perform topological data analysis
        tda_params = {
            'n_regimes': n_regimes,
            'alpha': alpha,
            'beta': beta,
            'lambda_info': lambda_info,
            'min_epsilon': min_epsilon,
            'max_epsilon': max_epsilon,
            'num_steps': num_steps,
            'window_size': window_size,
            'overlap': overlap,
            'max_path_length': max_path_length,
            'optimize_epsilon': optimize_epsilon,
            'timestamps': None if input_df is None else input_df[self.timestamp_col].values
        }
        
        self._perform_tda_analysis(**tda_params)
        
        # Train regime extension model
        print("\nTraining regime extension model...")
        
        extension_params = {
            'model_type': model_type,
            'train_size': train_size
        }
        
        extension_params.update(kwargs)
        self._train_regime_extension_model(**extension_params)
        
        return self 