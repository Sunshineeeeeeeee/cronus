import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from datetime import datetime
from volatility_regimes_identification import (
    MicrostructureFeatureEngine,
    InformationTheoryEnhancer,
    TopologicalDataAnalyzer,
    VolatilityRegimeAnalyzer
)

class VolatilityRegimesIdentifier:
    """
    Interface for identifying volatility regimes in tick data
    using the TDA volatility regimes package.
    """
    
    def __init__(self):
        """Initialize the volatility regimes identifier."""
        self.analyzer = None
        self.regimes = None
        self.regime_stats = None
        self.sample_size = None
        self.sampled_data = None
        self.pattern_classifier = None
        self.window_sizes = None  # Store window sizes
        self.output_dir = None  # Store output directory
        self.min_sequence_length = None
        self.smoothing_method = None
        self.smooth_sequences = None
        
    def identify_regimes(self, df, timestamp_col, price_col, volume_col, volatility_col, 
                        n_regimes=4, window_sizes=None, top_features=10, alpha=0.5, beta=0.1,
                        window_size=100, overlap=50, max_path_length=3, min_epsilon=0.1, max_epsilon=2.0,
                        sample_size=10000, sampling_method='sequential', output_dir=None,
                        smooth_sequences=True, min_sequence_length=20, smoothing_method='persistence_based'):
        """
        Identify volatility regimes using a two-stage approach with path complex and zigzag persistence.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe with tick data
        timestamp_col : str
            Column name for timestamps
        price_col : str
            Column name for price values
        volume_col : str
            Column name for volume values
        volatility_col : str
            Column name for volatility values
        n_regimes : int
            Number of regimes to detect
        window_sizes : list
            List of window sizes for feature calculation
        top_features : int
            Number of top features to include
        alpha : float
            Weight for temporal component
        beta : float
            Decay rate for temporal distance
        window_size : int
            Size of sliding window for zigzag persistence
        overlap : int
            Number of points to overlap between windows
        max_path_length : int
            Maximum length of paths to consider
        min_epsilon : float
            Minimum distance threshold
        max_epsilon : float
            Maximum distance threshold
        sample_size : int
            Number of points to sample for pattern learning
        sampling_method : str
            Method to use for sampling ('sequential' or 'chunks')
        output_dir : str
            Directory to save outputs
        smooth_sequences : bool
            Whether to smooth regime sequences
        min_sequence_length : int
            Minimum length for a sequence to be preserved in ticks
        smoothing_method : str
            Method to use for smoothing ('persistence_based', 'transition_prob', or 'confidence_weighted')
        """
        total_start_time = time.time()
        print("\nStarting volatility regime detection at", datetime.now().strftime("%H:%M:%S"))
        
        # Set up output directory
        if output_dir is None:
            self.output_dir = os.path.join('/Users/aleksandr/code/scripts/Diffusion/volatility_regimes_identification', 'results')
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)

        # Store parameters
        self.sample_size = min(sample_size, len(df))
        self.window_sizes = window_sizes if window_sizes is not None else [10, 30, 50]
        self.min_sequence_length = min_sequence_length
        self.smoothing_method = smoothing_method
        self.smooth_sequences = smooth_sequences
        
        # Stage 1: Sample and Learn Patterns
        print("\n=== Stage 1: Learning Patterns from Sample ===")
        sampling_start = time.time()
        self.sampled_data = self._sample_data(df, sampling_method)
        print(f"Sampling completed in {time.time() - sampling_start:.2f} seconds")
        
        # Initialize analyzer with sampled data
        self.analyzer = VolatilityRegimeAnalyzer(
            df=self.sampled_data,
            timestamp_col=timestamp_col,
            price_col=price_col,
            volume_col=volume_col,
            volatility_col=volatility_col
        )
        
        # Set window sizes explicitly to ensure consistency
        self.analyzer.window_sizes = self.window_sizes
        
        # Compute and enhance features on sample
        feature_start = time.time()
        print("\n--- STEP 1: Computing Features ---")
        _, _, _, self.analyzer = self.analyzer.compute_features(window_sizes=self.window_sizes)
        print(f"Feature computation completed in {time.time() - feature_start:.2f} seconds")
        
        enhancement_start = time.time()
        print("\n--- STEP 2: Enhancing Features ---")
        self.analyzer.enhance_features(n_features=top_features)
        print(f"Feature enhancement completed in {time.time() - enhancement_start:.2f} seconds")
        
        # Detect regimes on sample using path zigzag persistence
        regime_detection_start = time.time()
        print("\n--- STEP 3: Detecting Regimes ---")
        self.regimes = self.analyzer.detect_regimes(
            n_regimes=n_regimes,
            alpha=alpha,
            beta=beta,
            window_size=window_size,
            overlap=overlap,
            max_path_length=max_path_length,
            min_epsilon=min_epsilon,
            max_epsilon=max_epsilon,
            create_mapper=True,
            compute_homology=True,
            output_dir=self.output_dir
        )
        print(f"Regime detection completed in {time.time() - regime_detection_start:.2f} seconds")
        
        # Verify model is trained
        if self.regimes is None:
            raise ValueError("Failed to train regime detection model on sample data")
            
        # Analyze regimes to get statistics
        analysis_start = time.time()
        try:
            self.analyzer.regime_analysis = self.analyzer.tda.analyze_regimes()
            print(f"Regime analysis completed in {time.time() - analysis_start:.2f} seconds")
        except Exception as e:
            print(f"Warning: Error during regime analysis: {str(e)}")
            # Create a basic regime analysis with essential information to avoid downstream errors
            unique_regimes = np.unique(self.regimes)
            n_regimes = len(unique_regimes)
            regime_to_idx = {regime: idx for idx, regime in enumerate(unique_regimes)}
            idx_to_regime = {idx: regime for idx, regime in enumerate(unique_regimes)}
            
            self.analyzer.regime_analysis = {
                'regime_labels': self.regimes,
                'regime_stats': [{'regime_id': int(r), 'size': np.sum(self.regimes == r)} for r in unique_regimes],
                'transition_probs': np.ones((n_regimes, n_regimes)) / n_regimes,  # Uniform transition
                'n_regimes': n_regimes,
                'unique_regimes': unique_regimes.tolist(),
                'regime_to_idx': regime_to_idx,
                'idx_to_regime': idx_to_regime
            }
            print("Created fallback regime analysis")
        
        # Stage 2: Apply Patterns to Full Dataset
        print("\n=== Stage 2: Applying Patterns to Full Dataset ===")
        application_start = time.time()
        df_with_regimes = self._apply_patterns_to_full_dataset(
            df, timestamp_col, price_col, volume_col, volatility_col
        )
        print(f"Pattern application completed in {time.time() - application_start:.2f} seconds")
        
        # Calculate and store regime statistics
        stats_start = time.time()
        self._calculate_regime_statistics(df_with_regimes, volatility_col, timestamp_col)
        print(f"Statistics calculation completed in {time.time() - stats_start:.2f} seconds")
        
        # Save results
        saving_start = time.time()
        results_file = os.path.join(self.output_dir, 'tick_data_with_regimes.csv')
        df_with_regimes.to_csv(results_file, index=False)
        model_file = os.path.join(self.output_dir, 'regime_model.pkl')
        self.save_model(model_file)
        print(f"Results saving completed in {time.time() - saving_start:.2f} seconds")
        
        total_time = time.time() - total_start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        return df_with_regimes
    
    def _sample_data(self, df, sampling_method):
        """
        Sample data using the specified method, preserving sequential order.
        
        Parameters:
            df (pd.DataFrame): Input data
            sampling_method (str): Method to use for sampling
            
        Returns:
            pd.DataFrame: Sampled data
        """
        if sampling_method == 'sequential':
            # Take the first sample_size points
            sampled = df.iloc[:self.sample_size].copy()
        else:  # Default to sequential chunks
            # Take evenly spaced chunks to cover different time periods
            chunk_size = self.sample_size // 10
            total_size = len(df)
            step = total_size // 10
            
            chunks = []
            for i in range(0, total_size, step):
                if len(chunks) >= 10:  # Ensure we only get 10 chunks
                    break
                chunk = df.iloc[i:i+chunk_size].copy()
                chunks.append(chunk)
            
            sampled = pd.concat(chunks, ignore_index=True)
        
        return sampled
    
    def _apply_patterns_to_full_dataset(self, df, timestamp_col, price_col, volume_col, volatility_col):
        """
        Apply learned patterns to the full dataset.
        
        Parameters:
            df (pd.DataFrame): Full dataset
            timestamp_col (str): Name of timestamp column
            price_col (str): Name of price column
            volume_col (str): Name of volume column
            volatility_col (str): Name of volatility column
            
        Returns:
            pd.DataFrame: Full dataset with regime labels
        """
        print("Applying learned patterns to full dataset...")
        
        # Print debug info about the trained model
        print(f"Trained model has {len(np.unique(self.regimes))} unique regimes: {np.unique(self.regimes)}")
        print(f"Regime distribution in training data: {np.unique(self.regimes, return_counts=True)}")
        
        # Process the full dataset through the same pipeline
        full_analyzer = VolatilityRegimeAnalyzer(
            df=df,
            timestamp_col=timestamp_col,
            price_col=price_col,
            volume_col=volume_col,
            volatility_col=volatility_col
        )
        
        # Use the same feature computation but skip MI computation
        print("Computing microstructure features...")
        full_analyzer.features, full_analyzer.feature_names, _, _ = full_analyzer.compute_features(window_sizes=self.window_sizes)
        
        # Apply feature enhancement without computing MI
        if hasattr(self.analyzer, 'info_enhancer') and self.analyzer.info_enhancer is not None:
            print("Enhancing features using FFT approximation for mutual information...")
            # Copy necessary attributes from trained model
            n_features = len(self.analyzer.enhanced_feature_names) if hasattr(self.analyzer, 'enhanced_feature_names') else 10
            
            if hasattr(self.analyzer, 'enhanced_feature_names') and len(self.analyzer.enhanced_feature_names) > 0 and '_ent_weighted' in self.analyzer.enhanced_feature_names[0]:
                # If feature names contain _ent_weighted, we need to calculate the actual number of base features
                n_features = len([f for f in self.analyzer.enhanced_feature_names if not ('_ent_weighted' in f or '_kl' in f)])
            
            # Create a new info enhancer but copy over feature importance from trained model
            full_analyzer.info_enhancer = InformationTheoryEnhancer(
                full_analyzer.features, 
                full_analyzer.feature_names
            )
            
            # Copy feature importance from trained model to avoid recomputing
            if hasattr(self.analyzer.info_enhancer, 'feature_importance'):
                full_analyzer.info_enhancer.feature_importance = self.analyzer.info_enhancer.feature_importance
            
            # Copy entropy from trained model if available
            if hasattr(self.analyzer.info_enhancer, 'entropy'):
                full_analyzer.info_enhancer.entropy = self.analyzer.info_enhancer.entropy
            else:
                # If entropy is needed but not available from trained model, compute it
                print("Computing entropy using FFT approximation...")
                full_analyzer.info_enhancer.estimate_shannon_entropy()
                
            # Check if mutual information is needed but not available from trained model
            mi_needed = False
            if hasattr(self.analyzer, 'enhanced_feature_names'):
                for feat in self.analyzer.enhanced_feature_names:
                    if '_ent_weighted' in feat:
                        mi_needed = True
                        break
                    
            if mi_needed and not hasattr(self.analyzer.info_enhancer, 'mi_matrix'):
                print("Computing mutual information using FFT approximation for the full dataset...")
                # Always use FFT-based MI computation for the full dataset, regardless of size
                # Determine optimal bin size based on dataset characteristics
                n_samples = full_analyzer.info_enhancer.n_samples
                n_bins = min(256, max(64, int(np.sqrt(n_samples / 50))))  # Adjusted for better approximation
                
                # Force using FFT approximation
                full_analyzer.info_enhancer.compute_mutual_information_matrix(
                    use_fft=True,  # Always use FFT
                    fft_bins=n_bins
                )
                
                # Rank features after computing MI
                full_analyzer.info_enhancer.rank_features_by_importance()
                
            # Enhance features using FFT calculation if needed
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
        
        # Copy trained model components to the new analyzer
        full_analyzer.regime_labels = self.regimes  # Pass the regimes from the identifier to the analyzer
        
        # Ensure we copy the path zigzag persistence data as well
        if hasattr(self.analyzer, 'tda') and hasattr(self.analyzer.tda, 'path_zigzag_diagrams'):
            full_analyzer.tda = self.analyzer.tda
            print("Using path zigzag persistence information from trained model")
        else:
            # Fall back to simpler approach if no zigzag data available
            full_analyzer.tda = self.analyzer.tda
            print("Using standard persistence information from trained model")
            
        full_analyzer.regime_analysis = self.analyzer.regime_analysis
        
        # Use the learned patterns to label the full dataset
        df_with_regimes = full_analyzer.label_new_data(df)
        
        # Apply sequence smoothing to eliminate short sequences if requested
        if hasattr(self, 'smooth_sequences') and self.smooth_sequences:
            print(f"\n--- STEP 4: Smoothing Short Regime Sequences ---")
            smoothing_start = time.time()
            
            min_length = self.min_sequence_length if hasattr(self, 'min_sequence_length') else 20
            method = self.smoothing_method if hasattr(self, 'smoothing_method') else 'persistence_based'
            
            df_with_regimes = full_analyzer.smooth_regime_sequences(
                df_with_regimes,
                min_sequence_length=min_length,
                method=method
            )
            
            print(f"Sequence smoothing completed in {time.time() - smoothing_start:.2f} seconds")
        
        # Print distribution of regimes in the labeled data
        print(f"Regime distribution in labeled data: {np.unique(df_with_regimes['regime'], return_counts=True)}")
        
        return df_with_regimes
    
    def smooth_regime_sequences(self, df, min_sequence_length=20, method='persistence_based'):
        """
        Smooth regime sequences in an already labeled dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with regime labels to smooth
        min_sequence_length : int
            Minimum length for a sequence to be preserved
        method : str
            Method to use for smoothing:
            - 'persistence_based': Use TDA persistence concepts for merging
            - 'transition_prob': Use regime transition probabilities
            - 'confidence_weighted': Use regime confidence scores
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with smoothed regime labels
        """
        if self.analyzer is None:
            raise ValueError("You must identify regimes first")
            
        return self.analyzer.smooth_regime_sequences(df, min_sequence_length, method)
    
    def _calculate_regime_statistics(self, df_with_regimes, volatility_col, timestamp_col):
        """
        Calculate statistics for the identified regimes.
        
        Parameters:
            df_with_regimes (pd.DataFrame): Data with regime labels
            volatility_col (str): Name of volatility column
            timestamp_col (str): Name of timestamp column
        """
        # Get the unique regime labels which may not be sequential
        unique_regimes = np.unique(df_with_regimes['regime'])
        n_regimes = len(unique_regimes)
        
        # Create mappings between regime labels and indices
        regime_to_idx = {regime: idx for idx, regime in enumerate(unique_regimes)}
        idx_to_regime = {idx: regime for idx, regime in enumerate(unique_regimes)}
        
        # Calculate statistics for each regime
        regime_stats = {
            'regime_stats': [],
            'transition_probs': np.zeros((n_regimes, n_regimes)),
            'unique_regimes': unique_regimes.tolist(),
            'regime_to_idx': regime_to_idx,
            'idx_to_regime': idx_to_regime,
            'n_regimes': n_regimes
        }
        
        for regime in unique_regimes:
            regime_data = df_with_regimes[df_with_regimes['regime'] == regime]
            
            # Basic statistics
            stats = {
                'regime_id': int(regime),  # Convert to int to avoid numpy type issues
                'size': len(regime_data),
                'mean_vol': regime_data[volatility_col].mean() if len(regime_data) > 0 else 0
            }
            
            # Calculate duration if timestamps are available
            if len(regime_data) > 0 and pd.api.types.is_datetime64_any_dtype(regime_data[timestamp_col]):
                duration = (regime_data[timestamp_col].max() - 
                          regime_data[timestamp_col].min()).total_seconds()
                stats['duration'] = duration
                
            regime_stats['regime_stats'].append(stats)
        
        # Calculate transition probabilities using the mapping
        if len(df_with_regimes) > 1:
            for i in range(len(df_with_regimes) - 1):
                current_regime = df_with_regimes['regime'].iloc[i]
                next_regime = df_with_regimes['regime'].iloc[i + 1]
                
                # Map to indices for the transition matrix
                current_idx = regime_to_idx[current_regime]
                next_idx = regime_to_idx[next_regime]
                
                regime_stats['transition_probs'][current_idx, next_idx] += 1
            
            # Normalize transition probabilities
            for i in range(n_regimes):
                row_sum = regime_stats['transition_probs'][i].sum()
                if row_sum > 0:
                    regime_stats['transition_probs'][i] /= row_sum
        else:
            # If only one observation, use uniform transitions
            regime_stats['transition_probs'] = np.ones((n_regimes, n_regimes)) / n_regimes
        
        self.regime_stats = regime_stats
        
    def visualize_regimes(self, output_dir='./volatility_regimes'):
        """
        Visualize the identified regimes.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations
        """
        if self.regime_labels is None:
            raise ValueError("Regimes must be identified first")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get price data from features
        price_data = self.features[:, self.price_col_idx]
        
        # Plot price with regimes using the TDA's specialized method
        self.analyzer.plot_price_with_regimes(
            price_data=price_data,
            output_dir=output_dir,
            filename_prefix='volatility_'
        )
        
        print(f"Regime visualizations saved to {output_dir}/")
        
    def get_regime_statistics(self):
        """
        Get detailed statistics about the identified regimes.
        
        Returns:
        --------
        dict
            Dictionary containing regime statistics
        """
        if self.regime_stats is None:
            raise ValueError("You must identify regimes first")
            
        return self.regime_stats
    
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
        if self.analyzer is None or self.regimes is None:
            raise ValueError("You must identify regimes first")
            
        return self.analyzer.label_new_data(new_df)
    
    def predict_transitions(self, steps_ahead=10):
        """
        Predict future regime transitions.
        
        Parameters:
        -----------
        steps_ahead : int
            Number of steps to predict ahead
            
        Returns:
        --------
        list
            List of predicted regime transitions
        """
        if self.analyzer is None or self.regimes is None:
            raise ValueError("You must identify regimes first")
            
        return self.analyzer.predict_regime_transitions(steps_ahead)
    
    def save_model(self, filepath):
        """
        Save the regime identification model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
            
        Returns:
        --------
        None
        """
        if self.analyzer is None:
            raise ValueError("You must identify regimes first")
            
        self.analyzer.save_model(filepath)
        
    @classmethod
    def load_model(cls, filepath):
        """
        Load a saved regime identification model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        VolatilityRegimesIdentifier
            Loaded identifier
        """
        identifier = cls()
        identifier.analyzer = VolatilityRegimeAnalyzer.load_model(filepath)
        identifier.regimes = identifier.analyzer.get_regime_labels()
        identifier.regime_stats = identifier.analyzer.get_regime_analysis()
        
        return identifier

    def plot_price_by_regimes(self, df, price_col, output_dir='./volatility_regimes_results'):
        """
        Plot asset price colored by volatility regime, similar to the example plot.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data and regime labels
        price_col : str
            Name of the price column
        output_dir : str
            Directory to save the plot
        """
        if 'regime' not in df.columns:
            raise ValueError("DataFrame must contain 'regime' column. Run identify_regimes first.")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Plot price for each regime with different colors
        unique_regimes = sorted(df['regime'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regimes)))
        
        for regime, color in zip(unique_regimes, colors):
            mask = df['regime'] == regime
            regime_data = df[mask]
            
            plt.plot(regime_data.index, regime_data[price_col], 
                    marker='o', markersize=4, linestyle='-', linewidth=1,
                    color=color, label=f'Regime {regime + 1}',
                    alpha=0.8)
        
        # Customize the plot
        plt.title('Asset Price Colored by Volatility Regime', pad=20)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'price_by_regimes.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Price by regimes plot saved to {output_dir}/price_by_regimes.png")

    def visualize_path_zigzag_persistence(self, output_dir=None):
        """
        Visualize the path zigzag persistence analysis results.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations (defaults to self.output_dir)
        
        Returns:
        --------
        bool
            True if visualization was successful, False otherwise
        """
        if self.analyzer is None or not hasattr(self.analyzer, 'tda'):
            print("No analyzer with TDA pipeline found. Run identify_regimes first.")
            return False
            
        # Check if path zigzag persistence was computed
        if not hasattr(self.analyzer.tda, 'path_zigzag_diagrams') or self.analyzer.tda.path_zigzag_diagrams is None:
            print("No path zigzag persistence data found. Run identify_regimes with compute_homology=True.")
            return False
            
        # Use the specified output directory or default to self.output_dir
        if output_dir is None:
            output_dir = self.output_dir if self.output_dir is not None else './path_zigzag_results'
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Extract path zigzag data
            zigzag_data = self.analyzer.tda.path_zigzag_diagrams
            
            # Plot Betti numbers across windows
            plt.figure(figsize=(12, 6))
            betti_numbers = zigzag_data['betti_numbers']
            
            max_path_length = 0
            for key in betti_numbers:
                if key.startswith('betti_'):
                    dim = int(key.split('_')[1])
                    max_path_length = max(max_path_length, dim)
                    
            for dim in range(max_path_length + 1):
                betti_key = f'betti_{dim}'
                if betti_key in betti_numbers:
                    betti_values = betti_numbers[betti_key]
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
            if 'transitions' in betti_numbers and betti_numbers['transitions']:
                plt.figure(figsize=(12, 6))
                
                # Group by dimension
                transition_features = betti_numbers['transitions']
                max_dim = max([t['dimension'] for t in transition_features])
                
                for dim in range(1, max_dim + 1):
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
                
            # Plot correlation between zigzag persistence and regime changes
            if self.regime_labels is not None:
                plt.figure(figsize=(12, 6))
                
                # Extract regime transition points
                regime_changes = []
                for i in range(1, len(self.regime_labels)):
                    if self.regime_labels[i] != self.regime_labels[i-1]:
                        regime_changes.append(i)
                
                # Plot Betti numbers with regime change markers
                if 'betti_1' in betti_numbers:  # Use H1 as it's often most informative
                    betti_values = betti_numbers['betti_1']
                    plt.plot(range(len(betti_values)), betti_values, 
                           label='H1 Betti Numbers', linewidth=2)
                    
                    # Map regime change points to window indices
                    window_size = zigzag_data['window_size']
                    overlap = zigzag_data['overlap']
                    step = window_size - overlap
                    
                    for change_point in regime_changes:
                        # Map sample index to window index
                        window_idx = change_point // step
                        if window_idx < len(betti_values):
                            plt.axvline(x=window_idx, color='r', linestyle='--', alpha=0.5)
                
                plt.xlabel('Window Index')
                plt.ylabel('Betti Number')
                plt.title('Path Zigzag Persistence vs Regime Changes')
                plt.legend(['H1 Betti Numbers', 'Regime Change'])
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, 'zigzag_vs_regimes.png'))
                plt.close()
            
            print(f"Path zigzag persistence visualizations saved to {output_dir}")
            return True
            
        except Exception as e:
            print(f"Error in path zigzag persistence visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


# Example usage
if __name__ == "__main__":
    # Load your tick data
    # df = pd.read_csv("your_tick_data.csv")
    
    # Create identifier
    # identifier = VolatilityRegimesIdentifier()
    
    # Identify regimes with path zigzag persistence
    # df_with_regimes = identifier.identify_regimes(
    #     df,
    #     timestamp_col='Timestamp',
    #     price_col='Value',
    #     volume_col='Volume',
    #     volatility_col='Volatility',
    #     n_regimes=3,
    #     window_size=100,  # Size of sliding window for zigzag persistence
    #     overlap=50,       # Number of points to overlap between windows
    #     max_path_length=3,  # Maximum length of paths to consider
    #     min_epsilon=0.1,  # Minimum distance threshold
    #     max_epsilon=2.0,  # Maximum distance threshold
    # )
    
    # Visualize path zigzag persistence results
    # identifier.visualize_path_zigzag_persistence()
    
    # Visualize regimes
    # identifier.visualize_regimes()
    
    # Get regime statistics
    # stats = identifier.get_regime_statistics()
    
    # Predict future transitions
    # predictions = identifier.predict_transitions(steps_ahead=5)
    
    # Save model
    # identifier.save_model("volatility_regimes_model.pkl")
    
    # Load model
    # identifier = VolatilityRegimesIdentifier.load_model("volatility_regimes_model.pkl")
    
    print("To use this module, import it and create a VolatilityRegimesIdentifier instance.") 