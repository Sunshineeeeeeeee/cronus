import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mutual_info_score
from collections import Counter

class RegimeSmoother:
    """
    Implements Regime Smoothing with Information-Theoretic Penalty (RSIP) to address
    the issue of very short regime sequences in market regime identification.
    """
    
    def __init__(self, min_regime_length=None, alpha=0.7, beta=0.2, gamma=0.1, 
                 confidence_threshold=0.6, feature_cols=None):
        """
        Initialize the RegimeSmoother.
        
        Parameters:
        -----------
        min_regime_length : int or None
            Minimum acceptable regime length. If None, will be determined automatically
            using statistical significance testing.
        alpha : float
            Weight for confidence score differences.
        beta : float
            Weight for regime change penalty.
        gamma : float
            Weight for distribution divergence penalty.
        confidence_threshold : float
            Threshold for high confidence regime assignments.
        feature_cols : list of str or None
            Feature columns to use for distribution comparison. If None, will use all
            numeric columns except regime and confidence.
        """
        self.min_regime_length = min_regime_length
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.confidence_threshold = confidence_threshold
        self.feature_cols = feature_cols
        
    def determine_min_regime_length(self, df, regime_col='regime', 
                                  confidence_col='regime_confidence',
                                  significance_level=0.05):
        """
        Statistically determine the minimum significant regime length.
        
        Uses bootstrapping to find the minimum length at which regime sequences
        are statistically distinguishable from random noise.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with regime labels and features.
        regime_col : str
            Column name for regime labels.
        confidence_col : str
            Column name for confidence scores.
        significance_level : float
            Significance level for statistical testing.
            
        Returns:
        --------
        int
            Recommended minimum regime length.
        """
        print("Determining minimum regime length statistically...")
        
        # Extract actual regime sequences
        regimes = df[regime_col].values
        sequences = self._extract_sequences(regimes)
        
        # Get sequence lengths
        seq_lengths = [len(seq) for seq in sequences.values()]
        
        # Generate bootstrap samples of random regime assignments
        n_samples = 1000
        random_lengths = []
        
        for _ in range(n_samples):
            # Randomly permute the regimes
            random_regimes = np.random.permutation(regimes)
            random_sequences = self._extract_sequences(random_regimes)
            random_lengths.extend([len(seq) for seq in random_sequences.values()])
        
        # Find critical value - what sequence length is significantly different from random
        # Sort both actual and random sequence lengths
        seq_lengths.sort()
        random_lengths.sort()
        
        # Find the percentile in random distribution corresponding to significance level
        critical_idx = int((1 - significance_level) * len(random_lengths))
        critical_length = random_lengths[critical_idx]
        
        # Find the minimum length that's significantly different from random
        min_length = max(critical_length + 1, 3)  # At least 3 ticks
        
        print(f"Statistical analysis suggests minimum regime length of {min_length}")
        print(f"Random regime sequences are typically <= {critical_length} ticks at {significance_level} significance")
        
        return min_length
        
    def _extract_sequences(self, regimes):
        """
        Extract contiguous sequences of the same regime.
        
        Parameters:
        -----------
        regimes : numpy.ndarray
            Array of regime labels.
            
        Returns:
        --------
        dict
            Dictionary with sequence indices.
        """
        sequences = {}
        seq_id = 0
        current_regime = regimes[0]
        start_idx = 0
        
        for i, regime in enumerate(regimes):
            if regime != current_regime:
                sequences[seq_id] = {
                    'regime': current_regime,
                    'start': start_idx,
                    'end': i - 1,
                    'length': i - start_idx
                }
                seq_id += 1
                current_regime = regime
                start_idx = i
        
        # Add the last sequence
        sequences[seq_id] = {
            'regime': current_regime,
            'start': start_idx,
            'end': len(regimes) - 1,
            'length': len(regimes) - start_idx
        }
        
        return sequences
    
    def _compute_kl_divergence(self, dist1, dist2, bins=10):
        """
        Compute approximate KL divergence between two empirical distributions.
        
        Uses Jensen-Shannon divergence which is symmetric and bounded.
        
        Parameters:
        -----------
        dist1, dist2 : numpy.ndarray
            Arrays representing empirical distributions.
        bins : int
            Number of bins for histogram approximation.
            
        Returns:
        --------
        float
            Jensen-Shannon divergence between distributions.
        """
        # Handle edge cases
        if len(dist1) == 0 or len(dist2) == 0:
            return 1.0
            
        # Ensure arrays are numpy arrays
        dist1 = np.array(dist1).flatten()
        dist2 = np.array(dist2).flatten()
        
        # Get common range for binning
        min_val = min(dist1.min(), dist2.min())
        max_val = max(dist1.max(), dist2.max())
        
        # Create histograms
        hist1, _ = np.histogram(dist1, bins=bins, range=(min_val, max_val), density=True)
        hist2, _ = np.histogram(dist2, bins=bins, range=(min_val, max_val), density=True)
        
        # Add small constant to avoid zero probabilities
        hist1 = hist1 + 1e-10
        hist2 = hist2 + 1e-10
        
        # Normalize
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()
        
        # Compute Jensen-Shannon divergence
        js_div = jensenshannon(hist1, hist2)
        
        return js_div
    
    def smooth_regimes(self, df, regime_col='regime', confidence_col='regime_confidence'):
        """
        Apply regime smoothing to the input DataFrame, ensuring all sequences shorter
        than the minimum regime length are merged with neighboring sequences.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with regime labels and features.
        regime_col : str
            Column name for regime labels.
        confidence_col : str
            Column name for confidence scores.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with smoothed regime labels.
        """
        # Make a copy of input DataFrame
        result_df = df.copy()
        
        # Handle missing confidence scores (e.g., for TDA-processed data)
        if confidence_col in result_df.columns:
            if result_df[confidence_col].isna().any():
                print("Filling missing confidence scores with 1.0 for TDA-processed data")
                result_df[confidence_col] = result_df[confidence_col].fillna(1.0)
        else:
            # Create a dummy confidence column
            result_df[confidence_col] = 1.0
            
        # Determine feature columns if not provided
        if self.feature_cols is None:
            self.feature_cols = [col for col in result_df.columns 
                               if col not in [regime_col, confidence_col] 
                               and np.issubdtype(result_df[col].dtype, np.number)]
            print(f"Using features: {self.feature_cols}")
            
        # Determine minimum regime length if not provided
        if self.min_regime_length is None:
            self.min_regime_length = self.determine_min_regime_length(
                result_df, regime_col, confidence_col
            )
        
        # Extract original regime sequences
        orig_regimes = result_df[regime_col].values
        orig_sequences = self._extract_sequences(orig_regimes)
        
        # Print original sequence statistics
        print("\nOriginal regime sequences:")
        self._print_sequence_stats(orig_sequences)
        
        # Find short sequences to merge
        short_sequences = {
            seq_id: seq for seq_id, seq in orig_sequences.items() 
            if seq['length'] < self.min_regime_length
        }
        
        if not short_sequences:
            print("No short sequences found. No smoothing needed.")
            return result_df
            
        print(f"\nFound {len(short_sequences)} sequences shorter than {self.min_regime_length} ticks")
        
        # Process short sequences in order from shortest to longest
        sorted_short_seqs = sorted(
            short_sequences.items(), 
            key=lambda x: x[1]['length']
        )
        
        # Initialize smoothed regimes with original regimes
        smoothed_regimes = orig_regimes.copy()
        
        # Track merged sequences for reporting
        merges = []
        
        # For each short sequence, decide whether to merge with previous or next sequence
        for seq_id, seq in sorted_short_seqs:
            # Skip if this sequence has already been merged
            current_regime = smoothed_regimes[seq['start']]
            if current_regime != seq['regime']:
                continue
                
            # Get the sequence data
            seq_data = result_df.iloc[seq['start']:seq['end']+1]
            
            # Determine if we have previous and next sequences
            has_prev = seq_id > 0
            has_next = seq_id < max(orig_sequences.keys())
            
            # If we don't have one of the sequences, merge with the other
            if not has_prev:
                merge_with = 'next'
            elif not has_next:
                merge_with = 'prev'
            else:
                # Get previous and next sequence info
                prev_seq = orig_sequences[seq_id - 1]
                next_seq = orig_sequences[seq_id + 1]
                
                # Check if previous or next sequences have already been modified
                prev_regime = smoothed_regimes[prev_seq['start']]
                next_regime = smoothed_regimes[next_seq['start']]
                
                # Get the data for each sequence
                prev_data = result_df.iloc[prev_seq['start']:prev_seq['end']+1]
                next_data = result_df.iloc[next_seq['start']:next_seq['end']+1]
                
                # Compute information-theoretic measures
                
                # 1. Confidence-based cost (prefer to merge with the higher confidence regime)
                prev_conf = prev_data[confidence_col].mean()
                seq_conf = seq_data[confidence_col].mean()
                next_conf = next_data[confidence_col].mean()
                
                prev_conf_cost = self.alpha * abs(prev_conf - seq_conf)
                next_conf_cost = self.alpha * abs(next_conf - seq_conf)
                
                # 2. Regime change penalty (fixed cost)
                regime_change_cost = self.beta
                
                # 3. Feature distribution divergence
                prev_divs = []
                next_divs = []
                
                for col in self.feature_cols:
                    # Skip if the column has all identical values
                    if seq_data[col].nunique() <= 1:
                        continue
                        
                    prev_div = self._compute_kl_divergence(
                        seq_data[col].values,
                        prev_data[col].values
                    )
                    next_div = self._compute_kl_divergence(
                        seq_data[col].values,
                        next_data[col].values
                    )
                    
                    prev_divs.append(prev_div)
                    next_divs.append(next_div)
                
                # Average divergence across features
                prev_div_cost = self.gamma * np.mean(prev_divs) if prev_divs else 0
                next_div_cost = self.gamma * np.mean(next_divs) if next_divs else 0
                
                # Total cost
                prev_cost = prev_conf_cost + regime_change_cost + prev_div_cost
                next_cost = next_conf_cost + regime_change_cost + next_div_cost
                
                # Determine which sequence to merge with
                merge_with = 'prev' if prev_cost < next_cost else 'next'
                
                # Debug output
                print(f"Sequence {seq_id} (length {seq['length']}, regime {seq['regime']}):")
                print(f"  Prev cost: {prev_cost:.4f} (conf: {prev_conf_cost:.4f}, div: {prev_div_cost:.4f})")
                print(f"  Next cost: {next_cost:.4f} (conf: {next_conf_cost:.4f}, div: {next_div_cost:.4f})")
                print(f"  Decision: Merge with {merge_with}\n")
            
            # Perform the merge
            if merge_with == 'prev':
                prev_seq = orig_sequences[seq_id - 1]
                prev_regime = smoothed_regimes[prev_seq['start']]
                # Update regimes for the short sequence
                smoothed_regimes[seq['start']:seq['end']+1] = prev_regime
                merges.append(('prev', seq_id, prev_regime, seq['regime']))
            else:
                next_seq = orig_sequences[seq_id + 1]
                next_regime = smoothed_regimes[next_seq['start']]
                # Update regimes for the short sequence
                smoothed_regimes[seq['start']:seq['end']+1] = next_regime
                merges.append(('next', seq_id, next_regime, seq['regime']))
        
        # Update the result DataFrame with smoothed regimes
        result_df['smoothed_regime'] = smoothed_regimes
        
        # Check if there are still short sequences after the first pass
        smoothed_sequences = self._extract_sequences(smoothed_regimes)
        remaining_short = {seq_id: seq for seq_id, seq in smoothed_sequences.items() 
                           if seq['length'] < self.min_regime_length}
        
        # If there are still short sequences, do additional passes until all are merged
        passes = 1
        max_passes = 10  # Limit to prevent infinite loops
        
        while remaining_short and passes < max_passes:
            passes += 1
            print(f"\nPass {passes}: Found {len(remaining_short)} remaining short sequences")
            
            # Process remaining short sequences
            for seq_id, seq in sorted(remaining_short.items(), key=lambda x: x[1]['length']):
                # Determine if we have previous and next sequences
                has_prev = seq_id > 0
                has_next = seq_id < max(smoothed_sequences.keys())
                
                # Choose which sequence to merge with
                if not has_prev:
                    merge_with = 'next'
                elif not has_next:
                    merge_with = 'prev'
                else:
                    # Simple strategy: merge with the longer neighbor
                    prev_seq = smoothed_sequences[seq_id - 1]
                    next_seq = smoothed_sequences[seq_id + 1]
                    merge_with = 'prev' if prev_seq['length'] > next_seq['length'] else 'next'
                
                # Perform the merge
                if merge_with == 'prev':
                    prev_seq = smoothed_sequences[seq_id - 1]
                    prev_regime = smoothed_regimes[prev_seq['start']]
                    smoothed_regimes[seq['start']:seq['end']+1] = prev_regime
                else:
                    next_seq = smoothed_sequences[seq_id + 1]
                    next_regime = smoothed_regimes[next_seq['start']]
                    smoothed_regimes[seq['start']:seq['end']+1] = next_regime
            
            # Update after this pass
            result_df['smoothed_regime'] = smoothed_regimes
            smoothed_sequences = self._extract_sequences(smoothed_regimes)
            remaining_short = {seq_id: seq for seq_id, seq in smoothed_sequences.items() 
                               if seq['length'] < self.min_regime_length}
        
        # Print smoothed sequence statistics
        print("\nSmoothed regime sequences:")
        self._print_sequence_stats(smoothed_sequences)
        
        # Print merge statistics
        if merges:
            prev_merges = len([m for m in merges if m[0] == 'prev'])
            next_merges = len([m for m in merges if m[0] == 'next'])
            print(f"\nMerged {len(merges)} sequences: {prev_merges} with previous, {next_merges} with next")
        
        return result_df
    
    def _print_sequence_stats(self, sequences):
        """Print statistics for regime sequences."""
        # Group sequences by regime
        regime_sequences = {}
        for seq in sequences.values():
            regime = seq['regime']
            if regime not in regime_sequences:
                regime_sequences[regime] = []
            regime_sequences[regime].append(seq['length'])
        
        # Print statistics for each regime
        for regime, lengths in sorted(regime_sequences.items()):
            print(f"Regime {regime}:")
            print(f"  Number of sequences: {len(lengths)}")
            print(f"  Sequence lengths: {lengths}")
            print(f"  Average sequence length: {np.mean(lengths):.2f}")
            print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Median: {np.median(lengths)}") 