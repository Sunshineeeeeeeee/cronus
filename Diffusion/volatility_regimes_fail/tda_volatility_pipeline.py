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
        self.window_sizes = window_sizes if window_sizes is not None else [
            10, 50, 100]

        # Verify if the volatility column exists in the dataframe
        if volatility_col in df.columns:
            print(f"Verified volatility column '{volatility_col}' exists in input data")
        else:
            print(f"WARNING: Volatility column '{volatility_col}' not found in input data columns: {df.columns.tolist()}")
            # Try case-insensitive match
            for col in df.columns:
                if col.lower() == volatility_col.lower():
                    print(f"Found case-insensitive match: '{col}' will be used as volatility column")
                    self.volatility_col = col
                    volatility_col = col
                    break

        # Split data into training and extension parts
        n_samples = len(df)
        # Ensure we don't exceed data size
        training_batch = min(training_batch, n_samples)

        print(f"\nUsing first {training_batch:,} observations for TDA analysis")
        training_df = df.iloc[:training_batch].copy()
        extension_df = df.iloc[training_batch:].copy() if training_batch < n_samples else None

        # Print data summary for verification
        print("\nData Summary:")
        print(f"  - Total observations: {n_samples}")
        print(f"  - Training batch: {training_batch} observations")
        print(f"  - Extension data: {n_samples - training_batch if extension_df is not None else 0} observations")
        print(f"  - Timestamp column: '{timestamp_col}'")
        print(f"  - Price column: '{price_col}'")
        print(f"  - Volume column: '{volume_col}'")
        print(f"  - Volatility column: '{volatility_col}'")

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
            volatility_col=volatility_col,  # Explicitly pass the volatility column
            n_regimes=n_regimes,
            feature_params=feature_params,
            enhancement_params=enhancement_params,
            **tda_params
        )

        # Initialize result DataFrame with training results and add regime
        # column
        result_df = df.copy()

        # Create the regime column if it doesn't exist
        if 'regime' not in result_df.columns:
            result_df['regime'] = np.nan

        # Assign regimes to the training portion
        result_df.iloc[:training_batch,
                       result_df.columns.get_loc('regime')] = self.regimes

        # Add confidence column if not present
        if 'regime_confidence' not in result_df.columns:
            result_df['regime_confidence'] = np.nan

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

            for batch_idx, start_idx in enumerate(
                    range(0, n_remaining, batch_size), 1):
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
            result_df.iloc[training_batch:,
                           result_df.columns.get_loc('regime')] = all_regimes
            result_df.iloc[training_batch:, result_df.columns.get_loc(
                'regime_confidence')] = all_confidences

            print(f"Extension completed in {time.time() - extension_start:.2f} seconds")
            print(f"Final regime distribution: {np.bincount(all_regimes)}")

        # Report total time
        total_time = time.time() - total_start_time
        print(f"\nTDA volatility regime pipeline completed in {total_time:.2f} seconds")
        print(f"Processed {len(df):,} total observations at {len(df)/total_time:.1f} samples/sec")

        return result_df

    def _extract_features(self, window_sizes=None, normalize=True, include_original=True,
                          min_periods=None, df=None, volatility_col=None, **kwargs):
        """
        Extract microstructure features from input data.

        Parameters:
        -----------
        window_sizes : list or None
            List of window sizes to use for feature extraction
        normalize : bool
            Whether to normalize features
        include_original : bool
            Whether to include original price and volume columns
        min_periods : dict or None
            Minimum number of observations required for rolling calculations
        df : pandas.DataFrame or None
            Dataframe to extract features from (uses self.input_df if None)
        volatility_col : str or None
            Column name for volatility (if None, uses self.volatility_col)

        Returns:
        --------
        tuple
            Tuple containing (feature_array, feature_names)
        """
        start_time = time.time()
        print("Extracting features from input data...")

        if window_sizes is None:
            window_sizes = self.window_sizes if self.window_sizes else [
                10, 50, 100]

        if df is None:
            df = self.input_df

        if volatility_col is None:
            volatility_col = self.volatility_col

        # Initialize feature engine
        self.feature_engine = MicrostructureFeatureEngine(
            timestamp_col=self.timestamp_col,
            price_col=self.price_col,
            volume_col=self.volume_col,
            volatility_col=volatility_col
        )

        # Extract features using the new feature engine implementation
        print(f"Extracting microstructure features with window sizes: {window_sizes}")
        feature_df = self.feature_engine.extract_features(
            df,
            window_sizes=window_sizes,
            normalize=normalize,
            include_original=include_original
        )

        # The new feature engine handles NaNs internally, but we'll still check
        total_nan = feature_df.isna().sum().sum()
        if total_nan > 0:
            print(
                f"WARNING: Found {total_nan} NaN values in feature DataFrame after extraction")
            # Apply additional NaN handling if needed
            feature_df.fillna(0, inplace=True)
            print("Filled remaining NaNs with zeros")
        else:
            print("Feature DataFrame contains no NaN values - ready for analysis")

        # Get feature array and names
        feature_names = feature_df.columns.tolist()
        feature_array = feature_df.values

        # Report extracted features
        print(f"Extracted {len(feature_names)} features from {len(feature_array)} observations")
        print(f"Feature extraction time: {time.time() - start_time:.2f} seconds")

        # Also save the timestamps for use in TDA
        if self.timestamp_col in df.columns:
            self.timestamps = df[self.timestamp_col].values

        return feature_array, feature_names

    def _enhance_features(self, n_components=10, use_clustering=True, use_log=True,
                          use_mi=True, use_te=True, use_entropy=True):
        """
        Enhanced and more robust feature engineering that directly creates information-theory
        weighted features, bypassing the issues with NaN values from the InformationTheoryEnhancer.

        Parameters:
        -----------
        n_components : int
            Number of components to retain
        use_clustering : bool
            Whether to use clustering for enhancement
        use_log : bool
            Whether to use log transformation
        use_mi : bool
            Whether to use mutual information
        use_te : bool
            Whether to use transfer entropy
        use_entropy : bool
            Whether to use entropy

        Returns:
        --------
        self
        """
        print("Enhancing features with information theory...")
        self.start_time = time.time()  # Start timer for the whole process

        # Find the volatility column in the features
        volatility_indices = []
        volatility_name = None

        # Check if we have a volatility column
        for i, name in enumerate(self.feature_names):
            if name.lower() == self.volatility_col.lower() or 'volatil' in name.lower():
                volatility_indices.append(i)
                if volatility_name is None:
                    volatility_name = name

        if volatility_indices:
            print(f"Found volatility indices: {volatility_indices}")
            print(f"Using volatility column at index {volatility_indices[0]}: '{self.feature_names[volatility_indices[0]]}'")
        else:
            print("Warning: No volatility column found in features.")

        # Initialize the original information theory enhancer just for MI and TE calculation
        # We'll handle the feature enhancement directly to avoid NaN issues
        self.info_enhancer = InformationTheoryEnhancer()

        # Get the needed MI and TE calculations
        # This calls the enhancer but doesn't use its enhanced features
        if hasattr(self.info_enhancer, 'enhance_features'):
            # Call enhance_features but we'll only use the MI and TE matrices
            try:
                _, _ = self.info_enhancer.enhance_features(
                    self.features,
                    self.feature_names,
                    use_entropy=use_entropy,
                    use_mi=use_mi,
                    use_te=use_te,
                    use_clustering=use_clustering,
                    n_clusters=n_components,
                    use_log=use_log
                )
            except Exception as e:
                print(f"Error during information theory calculations: {e}")
                print("Will proceed with direct feature creation.")

        # ===== CREATE ENHANCED FEATURES DIRECTLY =====
        # Instead of relying on InformationTheoryEnhancer to create features,
        # we'll create them directly with full control over NaN handling

        # Step 1: Feature selection - Choose the most important features
        selected_feature_indices = []

        # Always include volatility features
        if volatility_indices:
            # Add the first volatility index
            selected_feature_indices.extend(volatility_indices[:1])

        # Get MI-based feature importance if available
        if hasattr(self.info_enhancer, 'mi_matrix') and use_mi:
            mi_matrix = self.info_enhancer.mi_matrix

            # Handle NaNs and negative values in MI matrix
            if np.isnan(mi_matrix).any():
                print("Handling NaNs in MI matrix")
                mi_matrix = np.nan_to_num(mi_matrix, nan=0.0)
            if (mi_matrix < 0).any():
                print("Handling negative values in MI matrix")
                mi_matrix = np.maximum(mi_matrix, 0.0)

            # Calculate feature importance based on MI
            if volatility_indices and volatility_indices[0] < mi_matrix.shape[0]:
                vol_idx = volatility_indices[0]
                # Get MI with volatility feature as importance score
                mi_importance = mi_matrix[vol_idx, :]
                # Select top features by MI importance
                top_mi_indices = np.argsort(-mi_importance)[:n_components // 2]
                selected_feature_indices.extend(
                    [i for i in top_mi_indices if i not in selected_feature_indices])

        # Get TE-based feature importance if available
        if hasattr(self.info_enhancer, 'te_matrix') and use_te:
            te_matrix = self.info_enhancer.te_matrix

            # Handle NaNs in TE matrix
            if np.isnan(te_matrix).any():
                print("Handling NaNs in TE matrix")
                te_matrix = np.nan_to_num(te_matrix, nan=0.0)

            # Calculate feature importance based on TE
            if volatility_indices and volatility_indices[0] < te_matrix.shape[0]:
                vol_idx = volatility_indices[0]
                # Get TE with volatility feature as importance score
                # Features that influence volatility
                te_importance = te_matrix[:, vol_idx]
                # Select top features by TE importance
                top_te_indices = np.argsort(-te_importance)[:n_components // 2]
                selected_feature_indices.extend(
                    [i for i in top_te_indices if i not in selected_feature_indices])

        # If we don't have enough features yet, add more based on variance
        if len(selected_feature_indices) < n_components:
            # Calculate feature variance
            feature_variance = np.var(self.features, axis=0)
            # Get indices sorted by variance (high to low)
            var_indices = np.argsort(-feature_variance)
            # Add top variance features that aren't already selected
            for idx in var_indices:
                if idx not in selected_feature_indices:
                    selected_feature_indices.append(idx)
                    if len(selected_feature_indices) >= n_components:
                        break

        # Ensure we don't exceed the desired number of components
        selected_feature_indices = selected_feature_indices[:n_components]

        # Step 2: Extract selected features
        selected_features = self.features[:, selected_feature_indices]
        selected_names = [self.feature_names[i]
                          for i in selected_feature_indices]

        # Step 3: Create additional information-weighted features
        # Start with selected features
        enhanced_features_list = [selected_features]
        enhanced_names_list = selected_names.copy()  # Start with selected names

        # Now add entropy-weighted versions of some key features
        # We'll do this carefully to avoid NaN values
        num_samples = self.features.shape[0]
        from sklearn.preprocessing import MinMaxScaler

        # Only select a few key features to weight
        key_feature_indices = selected_feature_indices[:min(
            10, len(selected_feature_indices))]

        if hasattr(self.info_enhancer, 'feature_entropy') and use_entropy:
            # Get entropy values
            entropy_values = self.info_enhancer.feature_entropy

            # Handle NaNs in entropy values
            if np.isnan(entropy_values).any():
                print("Handling NaNs in entropy values")
                entropy_values = np.nan_to_num(entropy_values, nan=0.0)

            # Normalize entropy to [0,1] range for weighting
            if len(entropy_values) > 0:
                entropy_scaler = MinMaxScaler()
                normalized_entropy = entropy_scaler.fit_transform(
                    entropy_values.reshape(-1, 1)).flatten()

                # Create entropy-weighted features
                for i, orig_idx in enumerate(key_feature_indices):
                    if i < len(
                            normalized_entropy) and orig_idx < self.features.shape[1]:
                        # Get the entropy weight
                        weight = normalized_entropy[orig_idx]
                        # Only create weighted feature if entropy is
                        # significant
                        if weight > 0.2:  # Only weight features with meaningful entropy
                            weighted_feature = self.features[:,
                                                             orig_idx] * weight
                            feature_name = f"{
                                self.feature_names[orig_idx]}_ent_weighted"

                            # Ensure no NaNs
                            if np.isnan(weighted_feature).any():
                                print(
                                    f"Fixing NaNs in weighted feature {feature_name}")
                                weighted_feature = np.nan_to_num(
                                    weighted_feature, nan=0.0)

                            # Add to our enhanced features and names
                            enhanced_features_list.append(
                                weighted_feature.reshape(-1, 1))
                            enhanced_names_list.append(feature_name)

        # Ensure the volatility feature is included
        vol_included = False
        for name in enhanced_names_list:
            if 'volatil' in name.lower() or name == self.volatility_col:
                vol_included = True
                break

        if not vol_included and volatility_indices:
            vol_idx = volatility_indices[0]
            vol_feature = self.features[:, vol_idx]
            vol_name = self.feature_names[vol_idx]

            # Add volatility feature
            enhanced_features_list.append(vol_feature.reshape(-1, 1))
            enhanced_names_list.append(vol_name)
            print(f"Added volatility column '{vol_name}' to enhanced features")

        # Step 4: Combine all features into final array
        # Combine all enhanced features (both selected and entropy-weighted)
        enhanced_features = np.hstack(enhanced_features_list)
        enhanced_names = enhanced_names_list

        # Final check for any remaining NaNs
        if np.isnan(enhanced_features).any():
            print(
                "\nWARNING: Found NaN values in enhanced features. Processing to fix...")
            nan_mask = np.isnan(enhanced_features)
            nan_counts = nan_mask.sum(axis=0)
            total_nans = nan_mask.sum()
            total_values = enhanced_features.size

            print(
                f"Total NaN values: {total_nans} out of {total_values} ({
                    total_nans / total_values:.2%})")

            # Identify columns with high NaN percentages
            nan_columns = []
            for i, count in enumerate(nan_counts):
                if count > 0:
                    nan_percent = count / enhanced_features.shape[0]
                    column_name = enhanced_names[i] if i < len(
                        enhanced_names) else f"Column {i}"
                    print(
                        f"Column {i} ({column_name}) has {count} NaNs ({
                            nan_percent:.2%})")

                    # If entire column is NaN, mark for potential removal
                    if count == enhanced_features.shape[0]:
                        nan_columns.append(i)
                        print(
                            f"Column {i} ({column_name}) is all NaN - will be dropped")

            # Remove columns that are all NaN
            if nan_columns:
                print(
                    f"\nDropping {
                        len(nan_columns)} columns that contain all NaN values")
                # Create mask of columns to keep
                keep_mask = np.ones(enhanced_features.shape[1], dtype=bool)
                keep_mask[nan_columns] = False

                # Filter features and names
                enhanced_features = enhanced_features[:, keep_mask]
                enhanced_names = [name for i, name in enumerate(
                    enhanced_names) if i not in nan_columns]

                print(
                    f"After dropping NaN columns: {
                        enhanced_features.shape[1]} features remain")

            # Replace any remaining NaNs with zeros
            enhanced_features = np.nan_to_num(enhanced_features, nan=0.0)
            print("Filled any remaining NaNs with zeros")

        # Store enhanced features and names
        self.enhanced_features = enhanced_features
        self.enhanced_feature_names = enhanced_names

        # Define n_enhanced here, before any conditional blocks
        n_enhanced = enhanced_features.shape[1]

        # Check if we can reuse the MI and TE matrices from the info_enhancer
        # to avoid redundant computation
        if hasattr(self.info_enhancer, 'mi_matrix') and hasattr(
                self.info_enhancer, 'te_matrix'):
            print(
                "\nReusing existing MI and TE matrices from InformationTheoryEnhancer...")

            # Get the original MI and TE matrices
            orig_mi_matrix = self.info_enhancer.mi_matrix
            orig_te_matrix = self.info_enhancer.te_matrix

            # We need to create proper-sized matrices for the enhanced features
            # Default medium correlation
            self.mi_matrix = np.ones((n_enhanced, n_enhanced)) * 0.5
            self.te_matrix = np.zeros((n_enhanced, n_enhanced))

            # If the original matrices have appropriate dimensions, we can use them
            # to populate values for the enhanced features
            if len(selected_feature_indices) > 0 and max(
                    selected_feature_indices) < orig_mi_matrix.shape[0]:
                print("Mapping original MI/TE values to enhanced feature matrices...")

                # Create mapping from original feature indices to enhanced
                # feature indices
                orig_to_enhanced = {}
                for i, orig_idx in enumerate(selected_feature_indices):
                    if i < len(selected_names):
                        orig_to_enhanced[orig_idx] = i

                # Transfer MI values from original to enhanced matrix
                for orig_i in orig_to_enhanced:
                    for orig_j in orig_to_enhanced:
                        enh_i = orig_to_enhanced[orig_i]
                        enh_j = orig_to_enhanced[orig_j]
                        if enh_i < self.mi_matrix.shape[0] and enh_j < self.mi_matrix.shape[1]:
                            self.mi_matrix[enh_i,
                                           enh_j] = orig_mi_matrix[orig_i,
                                                                   orig_j]

                # Transfer TE values from original to enhanced matrix
                for orig_i in orig_to_enhanced:
                    for orig_j in orig_to_enhanced:
                        enh_i = orig_to_enhanced[orig_i]
                        enh_j = orig_to_enhanced[orig_j]
                        if enh_i < self.te_matrix.shape[0] and enh_j < self.te_matrix.shape[1]:
                            self.te_matrix[enh_i,
                                           enh_j] = orig_te_matrix[orig_i,
                                                                   orig_j]
            else:
                print(
                    "Cannot reuse original matrices due to dimension mismatch - computing new ones...")
                # Fall through to compute new matrices
        else:
            print("\nComputing new MI and TE matrices for enhanced features...")

        # If we couldn't reuse the existing matrices, compute them from scratch
        if not hasattr(self, 'mi_matrix') or not hasattr(self, 'te_matrix') or \
           self.mi_matrix.shape[0] != n_enhanced or self.te_matrix.shape[0] != n_enhanced:

            # Create new matrices
            self.mi_matrix = np.zeros((n_enhanced, n_enhanced))
            self.te_matrix = np.zeros((n_enhanced, n_enhanced))

            try:
                # Try to import from sklearn.feature_selection instead of
                # sklearn.metrics
                from sklearn.feature_selection import mutual_info_regression
                from sklearn.preprocessing import StandardScaler

                # Scale features for better information theory calculations
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(enhanced_features)

                # Compute MI for each pair of features
                print("Computing mutual information matrix using sklearn...")
                for i in range(n_enhanced):
                    for j in range(n_enhanced):
                        if i == j:
                            # Self-information is maximum
                            self.mi_matrix[i, j] = 1.0
                        else:
                            # Calculate mutual information
                            mi = mutual_info_regression(
                                scaled_features[:, j].reshape(-1, 1),
                                scaled_features[:, i],
                                discrete_features=False
                            )[0]
                            self.mi_matrix[i, j] = mi

            except ImportError:
                # Fallback to a simplified correlation-based approach if
                # mutual_info_regression is not available
                print(
                    "mutual_info_regression not found, using correlation-based MI approximation...")
                from scipy.stats import pearsonr

                # Scale the features
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(enhanced_features)

                # Use squared correlation as a simple approximation for MI
                for i in range(n_enhanced):
                    for j in range(n_enhanced):
                        if i == j:
                            # Self-information is maximum
                            self.mi_matrix[i, j] = 1.0
                        else:
                            # Pearson correlation as MI proxy
                            try:
                                corr, _ = pearsonr(
                                    scaled_features[:, i], scaled_features[:, j])
                                # Square the correlation as a simple MI approximation
                                # MI ≈ -0.5 * log(1 - ρ²) for normal
                                # distributions
                                self.mi_matrix[i, j] = corr**2
                            except BaseException:
                                # Handle any calculation errors
                                self.mi_matrix[i, j] = 0.0

                print("Using correlation-based approximation for mutual information")

            # Normalize MI matrix
            if np.max(self.mi_matrix) > 0:
                self.mi_matrix /= np.max(self.mi_matrix)

            # Simple approximation of TE using lagged correlation
            print("Computing transfer entropy approximation...")
            for i in range(n_enhanced):
                for j in range(n_enhanced):
                    if i == j:
                        continue

                    # Use lagged correlation as TE proxy
                    x = scaled_features[:-1, i]
                    y = scaled_features[1:, j]

                    if len(x) > 2 and np.std(x) > 1e-10 and np.std(y) > 1e-10:
                        try:
                            corr = np.abs(np.corrcoef(x, y)[0, 1])
                            if not np.isnan(corr):
                                self.te_matrix[i, j] = corr**2
                        except Exception:
                            # Handle any calculation errors
                            pass

        # Normalize TE matrix
        if np.max(self.te_matrix) > 0:
            self.te_matrix /= np.max(self.te_matrix)

        # Emphasize volatility in MI and TE matrices
        for i, name in enumerate(enhanced_names):
            if 'volatil' in name.lower():
                # Increase weights for volatility features
                self.mi_matrix[i, :] *= 1.5
                self.mi_matrix[:, i] *= 1.5
                self.te_matrix[i, :] *= 1.5
                self.te_matrix[:, i] *= 1.5

                # Confirm volatility emphasis
                print(
                    f"Confirmed volatility column '{name}' is present in enhanced features at index {i}")
                break

        # Ensure volatility column is preserved
        vol_in_enhanced = False
        for name in self.enhanced_feature_names:
            if name.lower() == self.volatility_col.lower() or 'volatil' in name.lower():
                vol_in_enhanced = True
                break

        if vol_in_enhanced:
            print(
                f"Original volatility column '{
                    self.volatility_col}' preserved in enhanced features")
        else:
            print(
                f"WARNING: Volatility column '{
                    self.volatility_col}' not found in enhanced features")

        # Final information about enhanced features
        print(
            f"Feature enhancement took {
                time.time() -
                self.start_time:.2f} seconds")
        print(f"Enhanced features shape: {self.enhanced_features.shape}")

        return self

    def _perform_tda_analysis(self, n_regimes=3, window_size=100, overlap=50,
                              max_dimension=2, epsilon=None, optimize_epsilon=True,
                              compute_persistence=True, **kwargs):
        """
        Perform topological data analysis on features to identify volatility regimes.

        Parameters:
        -----------
        n_regimes : int
            Number of volatility regimes to identify
        window_size : int
            Size of sliding window for zigzag persistence
        overlap : int
            Overlap between consecutive windows
        max_dimension : int
            Maximum homology dimension to compute
        epsilon : float or None
            Distance threshold for network construction (if None, will be optimized)
        optimize_epsilon : bool
            Whether to optimize epsilon threshold
        compute_persistence : bool
            Whether to compute persistent homology

        Returns:
        --------
        self
        """
        print("\n===== Performing Topological Data Analysis =====")

        # Get original or enhanced features
        if hasattr(
                self, 'enhanced_features') and self.enhanced_features is not None:
            print("Using enhanced features for TDA")
            feature_array = self.enhanced_features
            feature_names = self.enhanced_feature_names
        else:
            print("No enhanced features available. Using original features for TDA.")
            # Use clean extracted features to avoid NaN issues
            feature_array, feature_names = self._extract_features()

        print(
            f"TDA input: {
                feature_array.shape[0]} samples, {
                feature_array.shape[1]} features")

        # Initialize topological analyzer with clean features
        self.tda = TopologicalDataAnalyzer(
            feature_array=feature_array,
            feature_names=feature_names,
            timestamp_array=self.timestamps if hasattr(
                self, 'timestamps') else None
        )

        # Find volatility feature index for optimization
        vol_index = None
        for i, name in enumerate(feature_names):
            if 'volatil' in name.lower() or name == self.volatility_col:
                vol_index = i
                print(f"Using {name} as target for epsilon optimization")
                break

        # Validate MI and TE matrices
        mi_matrix = None
        transfer_entropy = None

        if hasattr(self, 'mi_matrix') and self.mi_matrix is not None:
            # Check if MI matrix has correct dimensions
            if self.mi_matrix.shape == (
                    feature_array.shape[1], feature_array.shape[1]):
                mi_matrix = self.mi_matrix
                print(f"Using MI matrix with shape {mi_matrix.shape}")
            else:
                print(
                    f"WARNING: MI matrix shape {
                        self.mi_matrix.shape} doesnt match features shape {
                        feature_array.shape[1]}")
                print("MI matrix will not be used for distance computation")

        if hasattr(self, 'te_matrix') and self.te_matrix is not None:
            # Check if TE matrix has correct dimensions
            if self.te_matrix.shape == (
                    feature_array.shape[1], feature_array.shape[1]):
                transfer_entropy = self.te_matrix
                print(f"Using TE matrix with shape {transfer_entropy.shape}")
            else:
                print(
                    f"WARNING: TE matrix shape {
                        self.te_matrix.shape} doesn't match features shape {
                        feature_array.shape[1]}")
                print("TE matrix will not be used for distance computation")

        # Compute temporally-weighted distance matrix with validated matrices
        self.tda.compute_temporally_weighted_distance(
            alpha=0.5,  # weight for temporal component
            beta=0.1,   # decay rate for temporal component
            lambda_info=1.0,  # weight for information-theoretic enhancement
            mi_matrix=mi_matrix,
            transfer_entropy=transfer_entropy
        )

        # Optimize epsilon if requested
        if optimize_epsilon and epsilon is None:
            self.optimal_epsilon, self.min_epsilon, self.max_epsilon, self.info_gain_scores = self.tda.optimize_epsilon_threshold(
                target_index=vol_index,
                n_trials=10,
                min_percentile=5,
                max_percentile=90
            )
            epsilon = self.optimal_epsilon
        elif epsilon is None:
            # Default epsilon if not optimizing and not provided
            print("Using default epsilon value")
            epsilon = 0.5

        # Construct directed network
        print(f"Constructing directed network with epsilon={epsilon}")
        self.network = self.tda.construct_directed_network(
            epsilon=epsilon,
            enforce_temporal=True
        )

        # Compute zigzag persistence homology if requested
        if compute_persistence:
            print("Computing zigzag persistence homology...")
            self.zigzag_diagrams = self.tda.compute_persistent_path_zigzag_homology(
                window_size=window_size,
                overlap=overlap,
                max_path_length=max_dimension,
                min_epsilon=self.min_epsilon if hasattr(
                    self, 'min_epsilon') else 0.1,
                max_epsilon=self.max_epsilon if hasattr(
                    self, 'max_epsilon') else 1.0
            )

        # Identify regimes
        print(f"Identifying {n_regimes} volatility regimes...")
        self.regime_labels = self.tda.identify_regimes(
            n_regimes=n_regimes,
            use_topological_features=True
        )

        # Create regime summaries
        self._create_regime_summaries()

        print("===== TDA Analysis Completed =====\n")
        return self

    def _calculate_persistence_diagrams(self, max_dim=2):
        """Calculate persistence diagrams from the enhanced features."""
        self.persistence_diagrams = ripser(
            self.enhanced_features, maxdim=max_dim)["dgms"]
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
        print(
            f"Training set size: {
                len(X_train)}, Validation set size: {
                len(X_val)}")
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
            volatility_col=self.volatility_col if hasattr(
                self, 'volatility_col') else None
        )

        # Extract features directly using extract_features method
        print(
            f"Extracting features from batch with {
                len(batch_df)} observations...")
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
        # Only use adaptive enhancement if the original training used it and
        # the batch features are similar
        use_adaptive = False
        if (hasattr(self, 'info_enhancer') and self.info_enhancer is not None and
                hasattr(self, 'extension_model') and self.extension_model is not None):
            # We have an enhancer and a model, but need to check if it's safe
            # to use
            use_adaptive = set(
                self.feature_names).issubset(
                set(batch_feature_names))

        # Print info about processing method
        print(f"Feature count: {batch_features.shape[1]}")
        print(f"Using {'adaptive' if use_adaptive else 'standard'} enhancement")

        # Ensure we have model_features attribute set
        if not hasattr(self, 'model_features'):
            self.model_features = self.enhanced_features.shape[1] if hasattr(
                self, 'enhanced_features') else batch_features.shape[1]
            print(f"Setting model_features to {self.model_features}")

        # For processing batches, it's safer to just use the model feature count directly
        # This avoids issues with feature selection and entropy calculations
        if use_adaptive:
            try:
                # Try adaptive enhancement with safety checks
                print("Creating batch-optimized feature set...")

                # Match batch features to the original features used in
                # training
                matching_indices = []
                original_feature_set = set(self.feature_names)

                # First, find indices that match the original feature set
                for i, name in enumerate(batch_feature_names):
                    if name in original_feature_set:
                        matching_indices.append(i)

                # If we don't have enough matching features, append some
                # additional ones
                if len(matching_indices) < self.model_features // 2:
                    print(
                        f"Warning: Only found {
                            len(matching_indices)} matching features from training set")
                    # Add more features to reach at least half the model
                    # dimensions
                    remaining = list(
                        set(range(len(batch_feature_names))) - set(matching_indices))
                    matching_indices.extend(
                        remaining[:max(0, (self.model_features // 2) - len(matching_indices))])

                # Take only the first model_features indices (or fewer if not
                # enough available)
                matching_indices = matching_indices[:min(
                    len(matching_indices), self.model_features)]
                print(
                    f"Selected {
                        len(matching_indices)} features based on original training features")

                # Create a selected feature array
                batch_selected = batch_features[:, matching_indices]
                selected_names = [batch_feature_names[i]
                                  for i in matching_indices]

                # If we need additional features, we'll add zeros (padding)
                if batch_selected.shape[1] < self.model_features:
                    padding_size = self.model_features - \
                        batch_selected.shape[1]
                    print(
                        f"Adding {padding_size} padding features to match model dimensions")
                    padding = np.zeros((batch_selected.shape[0], padding_size))
                    batch_enhanced = np.hstack([batch_selected, padding])
                    enhanced_names = selected_names + \
                        [f"padding_{i}" for i in range(padding_size)]
                else:
                    batch_enhanced = batch_selected
                    enhanced_names = selected_names

            except Exception as e:
                print(f"Error during adaptive enhancement: {e}")
                print("Falling back to standard feature selection")
                use_adaptive = False

        if not use_adaptive:
            # Standard approach: just select the top features up to
            # model_features
            top_indices = list(
                range(min(self.model_features, batch_features.shape[1])))
            batch_enhanced = batch_features[:, top_indices]
            enhanced_names = [batch_feature_names[i] for i in top_indices]

            # If we need additional features, we'll add zeros (padding)
            if batch_enhanced.shape[1] < self.model_features:
                padding_size = self.model_features - batch_enhanced.shape[1]
                print(
                    f"Adding {padding_size} padding features to match model dimensions")
                padding = np.zeros((batch_enhanced.shape[0], padding_size))
                batch_enhanced = np.hstack([batch_enhanced, padding])
                enhanced_names = enhanced_names + \
                    [f"padding_{i}" for i in range(padding_size)]

        # Final dimension check
        if batch_enhanced.shape[1] != self.model_features:
            print(
                f"Warning: Dimension mismatch. Expected {
                    self.model_features}, got {
                    batch_enhanced.shape[1]}")
            # Force the correct dimensions as a last resort
            if batch_enhanced.shape[1] > self.model_features:
                batch_enhanced = batch_enhanced[:, :self.model_features]
            else:
                padding = np.zeros(
                    (batch_enhanced.shape[0],
                     self.model_features -
                     batch_enhanced.shape[1]))
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
        result_df.iloc[:, result_df.columns.get_loc('regime')] = all_regimes
        result_df.iloc[:, result_df.columns.get_loc(
            'regime_confidence')] = all_confidences

        print("Regime extension completed")
        print(f"Final regime distribution: {np.bincount(all_regimes)}")

        return result_df

    def train(self, input_df=None, n_regimes=4, alpha=0.5, beta=0.1, window_size=100,
              optimize_epsilon=True, model_type='xgboost', train_size=0.8,
              timestamp_col=None, price_col=None, volume_col=None,
              volatility_col=None, feature_params=None, enhancement_params=None, **kwargs):
        """
        Train the TDA volatility pipeline.

        Parameters:
        -----------
        input_df : pandas.DataFrame or None
            Input dataframe with market data
        n_regimes : int
            Number of regimes to identify
        alpha : float
            Temporal weighting factor
        beta : float
            Feature weighting factor
        window_size : int
            Size of sliding window for TDA
        optimize_epsilon : bool
            Whether to optimize epsilon threshold
        model_type : str
            Type of model to use for regime extension
        train_size : float
            Fraction of data to use for training
        timestamp_col : str or None
            Name of timestamp column
        price_col : str or None
            Name of price column
        volume_col : str or None
            Name of volume column
        volatility_col : str or None
            Name of volatility column
        feature_params : dict or None
            Parameters for feature extraction
        enhancement_params : dict or None
            Parameters for feature enhancement
        **kwargs : dict
            Additional parameters for TDA

        Returns:
        --------
        self
            The trained pipeline
        """
        # Update parameters if provided
        if timestamp_col is not None:
            self.timestamp_col = timestamp_col
        if price_col is not None:
            self.price_col = price_col
        if volume_col is not None:
            self.volume_col = volume_col
        if volatility_col is not None:
            self.volatility_col = volatility_col
            print(f"Setting volatility column to '{volatility_col}'")

        # Handle input data
        if input_df is not None:
            self.input_df = input_df

        # Ensure we have input data
        if self.input_df is None:
            raise ValueError(
                "Input data must be provided either through input_df parameter or self.input_df")

        # Set start_time for timing
        self.start_time = time.time()

        # Extract features
        print("Extracting features from input data...")
        feature_params = feature_params or {}
        # Capture tuple returned by _extract_features
        self.features, self.feature_names = self._extract_features(
            df=self.input_df,
            normalize=feature_params.get('normalize', True),
            include_original=feature_params.get('include_original', True),
            volatility_col=self.volatility_col,  # Explicitly pass volatility column
            **{k: v for k, v in feature_params.items() if k not in ['normalize', 'include_original']}
        )
        print(
            f"Working with {
                self.features.shape[1]} features from {
                self.features.shape[0]} observations")

        # Enhance features with information theory
        enhancement_params = enhancement_params or {}
        self._enhance_features(
            n_components=enhancement_params.get('n_components', 10),
            use_clustering=enhancement_params.get('use_clustering', True),
            use_log=enhancement_params.get('use_log', True),
            use_mi=enhancement_params.get('use_mi', True),
            use_te=enhancement_params.get('use_te', True),
            use_entropy=enhancement_params.get('use_entropy', True)
        )

        # Perform TDA analysis
        self._perform_tda_analysis(
            n_regimes=n_regimes,
            window_size=window_size,
            optimize_epsilon=optimize_epsilon,
            **kwargs
        )

        # Store results
        self.regimes = self.regime_labels

        # Train extension model if needed
        if model_type == 'xgboost':
            self._train_regime_extension_model()
        elif model_type is not None:
            print(
                f"Warning: Unsupported model type '{model_type}'. Using XGBoost instead.")
            self._train_regime_extension_model()

        return self
