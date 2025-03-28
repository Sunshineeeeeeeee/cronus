import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from scipy.stats import kurtosis
from sklearn.metrics import mutual_info_score
import warnings
import itertools
import time

# Suppress warnings
warnings.filterwarnings('ignore')

class InformationTheoryEnhancer:
    """
    Applies information-theoretic measures to enhance feature relevance and extraction.
    
    Implements Section 2 of the TDA Pipeline:
    - Shannon Entropy Estimation
    - Kullback-Leibler Divergence
    - Transfer Entropy
    - Mutual Information Matrix
    - Feature Significance Ranking
    """
    
    def __init__(self, feature_array, feature_names, target_col_idx=None):
        """
        Initialize the information theory enhancer.
        
        Parameters:
        -----------
        feature_array : numpy.ndarray
            Array of shape (n_samples, n_features) containing extracted features
        feature_names : list
            List of feature names corresponding to columns in feature_array
        target_col_idx : int
            Index of the target variable (typically volatility) in feature_array
        """
        self.feature_array = np.copy(feature_array)
        self.feature_names = feature_names
        
        # If target index not provided, assume volatility is the target (look for it in names)
        if target_col_idx is None:
            try:
                target_col_idx = self.feature_names.index('volatility')
            except ValueError:
                target_col_idx = -1  # Default to last column
                
        self.target_col_idx = target_col_idx
        self.n_samples, self.n_features = self.feature_array.shape
        
        # Preprocess features
        self._preprocess_features()
        
        # Information-theoretic measures
        self.entropy = None
        self.mi_matrix = None
        self.feature_importance = None
        self.transfer_entropy = None
        
    def _preprocess_features(self):
        """Preprocess features to handle nans and scale data."""
        # Replace nans with mean of column
        for j in range(self.n_features):
            col = self.feature_array[:, j]
            mask = np.isnan(col)
            if mask.any():
                col[mask] = np.mean(col[~mask]) if np.any(~mask) else 0
                self.feature_array[:, j] = col
                
        # Store clean data
        self.clean_data = self.feature_array.copy()
                
        # Create a scaler for later use
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(self.clean_data)
        
    def _adjust_for_heavy_tails(self, x):
        """
        Adjust bandwidth and weights for heavy-tailed distributions.
        """
        # Compute excess kurtosis
        k = kurtosis(x.ravel())
        
        # Adjust bandwidth based on kurtosis
        tail_factor = np.log1p(abs(k)) if k > 0 else 1.0
        
        # Create weight adjustments for tails
        weights = np.ones_like(x.ravel())
        std = np.std(x)
        tail_mask = np.abs(x.ravel()) > 2 * std
        weights[tail_mask] = tail_factor
        
        return tail_factor, weights

    def estimate_shannon_entropy(self, bandwidth_method='silverman'):
        """
        Estimate Shannon entropy using kernel density estimation with heavy-tail adjustment.
        
        Parameters:
        -----------
        bandwidth_method : str
            Method for bandwidth selection ('silverman' or 'scott')
            
        Returns:
        --------
        numpy.ndarray
            Array of entropy values for each feature
        """
        print("Estimating Shannon entropy with heavy-tail adjustment...")
        start_time = time.time()
        
        n_features = self.feature_array.shape[1]
        entropy = np.zeros(n_features)
        
        for i in range(n_features):
            x = self.feature_array[:, i].reshape(-1, 1)
            
            # Compute robust statistics for bandwidth
            iqr = np.percentile(x, 75) - np.percentile(x, 25)
            std = np.std(x)
            n = len(x)
            
            # Use robust bandwidth estimation
            if bandwidth_method == 'silverman':
                bw = 0.9 * min(std, iqr/1.34) * n**(-0.2)
            else:  # scott
                bw = 1.06 * min(std, iqr/1.34) * n**(-0.2)
                
            # Ensure bandwidth is positive
            bw = max(bw, 1e-3)
            
            # Weight points based on their distance from the mean
            weights = np.ones(n)
            mean = np.mean(x)
            std = np.std(x)
            if std > 0:
                z_scores = np.abs((x - mean) / std).flatten()
                # Apply sigmoid weighting to reduce impact of outliers
                weights = 1 / (1 + np.exp(z_scores - 3))
            
            # Create weighted sample
            x_weighted = np.repeat(x, (weights * 100).astype(int), axis=0)
            
            # Fit kernel density with cosine kernel on weighted points
            kde = KernelDensity(
                bandwidth=bw,
                kernel='cosine'  # Using cosine kernel for better tail behavior
            )
            kde.fit(x_weighted)
            
            # Evaluate density at sample points with more points in tails
            std = np.std(x)
            if std > 0:
                eval_points = np.linspace(np.min(x) - 2*std, np.max(x) + 2*std, 1000).reshape(-1, 1)
            else:
                eval_points = np.linspace(np.min(x) - 0.1, np.max(x) + 0.1, 1000).reshape(-1, 1)
            
            # Get log density
            log_dens = kde.score_samples(eval_points)
            
            # Compute entropy using trapezoidal rule
            p = np.exp(log_dens)
            p = p / np.trapz(p, eval_points.flatten())  # Normalize
            entropy[i] = -np.trapz(p * np.log(p + 1e-10), eval_points.flatten())
        
        self.entropy = entropy
        print(f"Shannon entropy estimation completed in {time.time() - start_time:.2f} seconds")
        
        return entropy
        


    def compute_mutual_information_histogram(self, n_bins=64, max_sample_size=100000):
        """
        Compute mutual information using histogram-based approximation for efficient computation
        on large datasets. This method uses binning and histogram approximations to achieve
        O(n log n) complexity.
        
        Parameters:
        -----------
        n_bins : int
            Number of bins for histogram approximation
        max_sample_size : int
            Maximum sample size to use (for memory efficiency on huge datasets)
            
        Returns:
        --------
        numpy.ndarray
            Matrix of mutual information values
        """
        start_time = time.time()
        print(f"Computing mutual information using histogram approximation (n_bins={n_bins})...")
        
        # Check if we need to subsample (for memory efficiency with huge datasets)
        if self.n_samples > max_sample_size:
            print(f"Subsampling data from {self.n_samples} to {max_sample_size} samples")
            indices = np.random.choice(self.n_samples, max_sample_size, replace=False)
            data = self.scaled_data[indices]
        else:
            data = self.scaled_data
        
        n_samples, n_features = data.shape
        
        # Initialize mutual information matrix
        mi_matrix = np.zeros((n_features, n_features))
        
        # Precompute histograms for all features to avoid redundant computation
        all_histograms = []
        all_bin_edges = []
        for i in range(n_features):
            x = data[:, i]
            hist, bin_edges = np.histogram(x, bins=n_bins, density=True)
            all_histograms.append(hist)
            all_bin_edges.append(bin_edges)
            
            # Compute entropy directly from histogram (reuse later)
            px = hist / np.sum(hist)
            px = px[px > 0]  # Avoid log(0)
            entropy_i = -np.sum(px * np.log(px))
            
            # Store entropy on diagonal
            mi_matrix[i, i] = entropy_i
            
        # Compute MI for all pairs using histograms
        for i in range(n_features):
            for j in range(i+1, n_features):
                x = data[:, i]
                y = data[:, j]
                
                # Compute 2D histogram efficiently
                hist_joint, _, _ = np.histogram2d(x, y, bins=n_bins, density=True)
                
                # Approximate mutual information using discrete formulation
                px = all_histograms[i]
                py = all_histograms[j]
                
                # Create outer product for independent distribution (px * py)
                pxy_indep = np.outer(px, py)
                
                # Compute MI using KL divergence: sum(p_xy * log(p_xy / (px*py)))
                valid_mask = (hist_joint > 0) & (pxy_indep > 0)
                
                if np.any(valid_mask):
                    mi_value = np.sum(
                        hist_joint[valid_mask] * 
                        np.log(hist_joint[valid_mask] / pxy_indep[valid_mask])
                    )
                else:
                    mi_value = 0
                    
                mi_matrix[i, j] = mi_value
                mi_matrix[j, i] = mi_value  # Symmetric
                
        print(f"Histogram-based MI computation completed in {time.time() - start_time:.2f} seconds")
        
        return mi_matrix

    def compute_mutual_information_matrix(self, fast_approximation=False, use_fft=False, n_rff_features=50, fft_bins=64):
        """
        Compute mutual information between all pairs of features.
        
        Parameters:
        -----------
        fast_approximation : bool
            If True, use histogram-based approximation for faster computation
        use_fft : bool
            Deprecated parameter, kept for backward compatibility
        n_rff_features : int
            Unused parameter, kept for backward compatibility
        fft_bins : int
            Number of bins for histogram approximation (if fast_approximation=True)
            
        Returns:
        --------
        numpy.ndarray
            Array of shape (n_features, n_features) containing mutual information values
        """
        if fast_approximation:
            self.mi_matrix = self.compute_mutual_information_histogram(n_bins=fft_bins)
            return self.mi_matrix
            
        # Original KDE-based implementation for training phase
        start_time = time.time()
        print("Computing mutual information matrix using KDE...")
        
        # Initialize mutual information matrix
        mi_matrix = np.zeros((self.n_features, self.n_features))
        
        # Pre-compute all bandwidths and tail factors
        bandwidths = np.zeros(self.n_features)
        tail_factors = np.zeros(self.n_features)
        for i in range(self.n_features):
            x = self.scaled_data[:, i].reshape(-1, 1)
            tail_factor, _ = self._adjust_for_heavy_tails(x)
            bandwidths[i] = 1.06 * np.std(x) * self.n_samples**(-1/5) * tail_factor
            tail_factors[i] = tail_factor
        
        # Create all pairs of features at once
        feature_pairs = np.array(list(itertools.combinations(range(self.n_features), 2)))
        
        # Process all pairs in parallel
        for i, j in feature_pairs:
            x = self.scaled_data[:, i].reshape(-1, 1)
            y = self.scaled_data[:, j].reshape(-1, 1)
            
            # Rest of the original KDE implementation...
            bw_x = bandwidths[i]
            bw_y = bandwidths[j]
            bw_joint = np.mean([bw_x, bw_y])
            
            kde_x = KernelDensity(bandwidth=bw_x, kernel='cosine').fit(x)
            kde_y = KernelDensity(bandwidth=bw_y, kernel='cosine').fit(y)
            kde_joint = KernelDensity(bandwidth=bw_joint, kernel='cosine').fit(np.column_stack([x, y]))
            
            log_px = kde_x.score_samples(x)
            log_py = kde_y.score_samples(y)
            log_pxy = kde_joint.score_samples(np.column_stack([x, y]))
            
            mi = np.mean(log_pxy - log_px - log_py)
            tail_emphasis = np.maximum(tail_factors[i], tail_factors[j])
            mi *= tail_emphasis
            
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
        
        # Fill diagonal with entropy values
        np.fill_diagonal(mi_matrix, self.entropy)
        
        self.mi_matrix = mi_matrix
        print(f"Mutual information computation completed in {time.time() - start_time:.2f} seconds")
        return mi_matrix
        
    def compute_kl_divergence(self, window_size=100, gap_size=20):
        """
        Compute Kullback-Leibler divergence between current and recent distributions.
        Vectorized implementation for better performance.
        
        D_KL(P^t_current || P^t_recent) = ∑_i P^t_current(i) log[P^t_current(i)/P^t_recent(i)]
        
        Parameters:
        -----------
        window_size : int
            Size of window for computing distributions
        gap_size : int
            Size of gap between current and recent windows
            
        Returns:
        --------
        numpy.ndarray
            Array of shape (n_samples, n_features) containing KL divergence values
        """
        start_time = time.time()
        print("Computing KL divergence between current and recent windows...")
        
        # Initialize output array
        kl_divergence = np.zeros((self.n_samples, self.n_features))
        
        # Check if we have enough samples for the window size and gap
        if window_size + gap_size >= self.n_samples:
            print(f"Warning: Window size ({window_size}) + gap size ({gap_size}) is larger than number of samples ({self.n_samples}).")
            print("Adjusting window size to fit the data...")
            window_size = min(window_size, self.n_samples // 2)
            gap_size = min(gap_size, self.n_samples // 4)
        
        # Create window indices for all features at once
        start_idx = window_size + gap_size
        n_windows = max(0, self.n_samples - start_idx)
        
        if n_windows == 0:
            print("Warning: No valid windows found for KL divergence computation.")
            return kl_divergence
        
        # Process each feature
        for j in range(self.n_features):
            # Get feature data
            feature_data = self.scaled_data[:, j]
            
            # Create windows for this feature
            current_windows = np.zeros((n_windows, window_size))
            recent_windows = np.zeros((n_windows, window_size))
            
            for i in range(n_windows):
                current_windows[i] = feature_data[start_idx + i - window_size:start_idx + i]
                recent_windows[i] = feature_data[start_idx + i - window_size - gap_size:start_idx + i - gap_size]
            
            # Compute adaptive bins for all windows at once
            n_bins = max(5, min(20, int(np.sqrt(window_size / 5))))
            all_data = np.concatenate([current_windows.ravel(), recent_windows.ravel()])
            bin_edges = np.percentile(all_data, np.linspace(0, 100, n_bins + 1))
            
            # Compute histograms for all windows at once using vectorized operations
            current_hist = np.zeros((n_windows, n_bins))
            recent_hist = np.zeros((n_windows, n_bins))
            
            for i in range(n_windows):
                current_hist[i] = np.histogram(current_windows[i], bins=bin_edges, density=True)[0]
                recent_hist[i] = np.histogram(recent_windows[i], bins=bin_edges, density=True)[0]
            
            # Add small constant and normalize
            current_hist = current_hist + 1e-10
            recent_hist = recent_hist + 1e-10
            current_hist = current_hist / current_hist.sum(axis=1, keepdims=True)
            recent_hist = recent_hist / recent_hist.sum(axis=1, keepdims=True)
            
            # Compute kurtosis for all windows at once
            kurtosis_values = np.array([kurtosis(window) for window in current_windows])
            tail_factors = np.log1p(np.abs(kurtosis_values)) * (kurtosis_values > 0) + 1.0
            
            # Compute KL divergence for all windows at once
            kl_div = np.sum(current_hist * np.log(current_hist / recent_hist), axis=1) * tail_factors
            
            # Store results
            kl_divergence[start_idx:, j] = kl_div
        
        # Store as class attribute and return
        self.kl_divergence = kl_divergence
        print(f"KL divergence computation completed in {time.time() - start_time:.2f} seconds")
        return kl_divergence
        
    def compute_transfer_entropy(self, lag=1, k=1):
        """
        Compute transfer entropy to capture causal information flow between variables.
        
        TE_X→Y = ∑ p(y_{t+1}, y_t, x_t) log[p(y_{t+1}|y_t, x_t)/p(y_{t+1}|y_t)]
        
        Parameters:
        -----------
        lag : int
            Time lag for computing conditional probabilities
        k : int
            Number of nearest neighbors for density estimation
            
        Returns:
        --------
        numpy.ndarray
            Array of shape (n_features, n_features) containing transfer entropy values
        """
        start_time = time.time()
        print("\n======================================================================")
        print("COMPUTING TRANSFER ENTROPY: Analyzing causal information flow between variables")
        print("======================================================================\n")
        
        try:
            # Initialize transfer entropy matrix
            te_matrix = np.zeros((self.n_features, self.n_features))
            
            # Check if we have enough samples for the given lag
            if lag >= self.n_samples - 5:  # Need at least a few samples after lag
                print(f"Warning: Lag ({lag}) is too large for number of samples ({self.n_samples}).")
                lag = max(1, int(self.n_samples / 10))  # Reasonable default
                print(f"Setting lag to {lag}")
            
            # Process all pairs of features
            for i in range(self.n_features):
                target_future = self.scaled_data[lag:, i].copy()
                target_present = self.scaled_data[:-lag, i].copy()
                
                # Ensure we have valid data
                if len(target_future) < 10 or len(target_present) < 10:
                    print(f"Warning: Not enough samples for feature {i} after applying lag. Skipping.")
                    continue
                    
                for j in range(self.n_features):
                    if i == j:
                        continue
                        
                    source_present = self.scaled_data[:-lag, j].copy()
                    
                    # Skip if any NaN values
                    if np.isnan(target_future).any() or np.isnan(target_present).any() or np.isnan(source_present).any():
                        print(f"Warning: NaN values detected for pair ({i},{j}). Skipping.")
                        continue
                    
                    # Skip if any array has constant values (would cause binning issues)
                    if np.std(target_future) < 1e-10 or np.std(target_present) < 1e-10 or np.std(source_present) < 1e-10:
                        print(f"Warning: Constant values detected for pair ({i},{j}). Skipping.")
                        continue
                    
                    try:
                        # Calculate entropy terms using binning approach
                        # Use adaptive number of bins based on sample size
                        n_samples = len(target_present)
                        nbins = max(3, min(20, int(np.sqrt(n_samples / 10))))
                        
                        # Calculate entropy of target_future given target_present
                        h_y_given_yp = self._binned_conditional_entropy(target_future, target_present, nbins)
                        
                        # Calculate entropy of target_future given target_present and source_present
                        h_y_given_yp_xp = self._binned_conditional_entropy(
                            target_future, 
                            np.column_stack([target_present, source_present]),
                            nbins
                        )
                        
                        # Transfer entropy is the difference of these conditional entropies
                        te = h_y_given_yp - h_y_given_yp_xp
                        te_matrix[j, i] = max(0, te)  # Ensure non-negative
                    except Exception as e:
                        print(f"Error calculating transfer entropy for pair ({i},{j}): {str(e)}")
                        te_matrix[j, i] = 0  # Set to zero on error
            
            # Emphasize transfer entropy to/from target variable
            if self.target_col_idx is not None:
                target_factor = 1.5  # Give more weight to target-related transfer entropy
                te_matrix[:, self.target_col_idx] *= target_factor  # Transfer TO target
                te_matrix[self.target_col_idx, :] *= target_factor  # Transfer FROM target
                
            # Store as class attribute and return
            self.transfer_entropy = te_matrix
            print(f"\nTransfer entropy computation completed in {time.time() - start_time:.2f} seconds")
            print(f"Maximum transfer entropy value: {te_matrix.max():.4f}")
            
            # Check if we found meaningful transfer entropy
            if te_matrix.max() < 1e-6:
                print("Warning: Very low transfer entropy values detected. This might indicate:")
                print("  - Insufficient or too noisy data")
                print("  - Lag parameter needs adjustment (current lag={lag})")
                print("  - Variables may not have causal relationships")
            
            print("======================================================================\n")
            return te_matrix
            
        except Exception as e:
            print(f"Error in transfer entropy computation: {str(e)}")
            print(f"Creating fallback transfer entropy matrix with zeros")
            fallback_matrix = np.zeros((self.n_features, self.n_features))
            self.transfer_entropy = fallback_matrix
            print("======================================================================\n")
            return fallback_matrix
        
    def _binned_conditional_entropy(self, y, x, bins):
        """Helper method to calculate conditional entropy using binning.
        
        Parameters:
        -----------
        y : numpy.ndarray
            Target variable (1D array)
        x : numpy.ndarray
            Condition variable (1D or 2D array)
        bins : int
            Number of bins to use
            
        Returns:
        --------
        float
            Conditional entropy H(Y|X)
        """
        if y.ndim > 1:
            y = y.flatten()
        
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            
        # Bin the data
        y_bins = np.linspace(y.min(), y.max(), bins+1)
        y_digitized = np.digitize(y, y_bins) - 1
        
        # Handle multi-dimensional x
        n_dim_x = x.shape[1]
        x_digitized = np.zeros((x.shape[0], n_dim_x), dtype=int)
        
        for j in range(n_dim_x):
            x_bins = np.linspace(x[:, j].min(), x[:, j].max(), bins+1)
            x_digitized[:, j] = np.digitize(x[:, j], x_bins) - 1
        
        # Calculate conditional entropy based on dimensionality of x
        conditional_entropy = 0
        
        if n_dim_x == 1:
            # For 1D condition, use simple binning approach
            for i in range(bins):
                # Create boolean mask for condition x=i
                x_mask = (x_digitized[:, 0] == i)
                p_x = np.mean(x_mask)
                
                if p_x > 0:
                    # Get y values where x=i
                    y_given_x = y_digitized[x_mask]
                    
                    if len(y_given_x) > 0:
                        # Count occurrences of each y bin
                        y_counts = np.bincount(y_given_x, minlength=bins)
                        # Calculate probabilities
                        y_probs = y_counts / np.sum(y_counts)
                        # Only use non-zero probabilities for entropy
                        valid_probs = y_probs[y_probs > 0]
                        
                        # Calculate entropy for this condition
                        entropy_y_given_x = -np.sum(valid_probs * np.log(valid_probs))
                        conditional_entropy += p_x * entropy_y_given_x
        else:
            # For multidimensional x, use combined bin index approach
            # Create a unique integer code for each combination of x bins
            max_bin = bins
            x_combined = np.zeros(x.shape[0], dtype=int)
            
            for j in range(n_dim_x):
                x_combined += x_digitized[:, j] * (max_bin ** j)
            
            # Get unique combinations and their probabilities
            unique_x_combs, counts = np.unique(x_combined, return_counts=True)
            probs_x = counts / len(x_combined)
            
            # Calculate entropy for each unique combination
            for idx, x_comb in enumerate(unique_x_combs):
                x_mask = (x_combined == x_comb)
                p_x = probs_x[idx]
                
                # Get y values for this combination
                y_given_x = y_digitized[x_mask]
                
                if len(y_given_x) > 0:
                    # Calculate entropy
                    y_counts = np.bincount(y_given_x, minlength=bins)
                    y_probs = y_counts / np.sum(y_counts)
                    # Only use non-zero probabilities for entropy
                    valid_probs = y_probs[y_probs > 0]
                    
                    if len(valid_probs) > 0:
                        entropy_y_given_x = -np.sum(valid_probs * np.log(valid_probs))
                        conditional_entropy += p_x * entropy_y_given_x
                
        return conditional_entropy
        
    def rank_features_by_importance(self):
        """
        Rank features by importance using normalized mutual information with target.
        
        NMI(X_j, σ) = I(X_j; σ)/sqrt(H(X_j) · H(σ))
        
        Returns:
        --------
        tuple
            (ranked_indices, importance_scores)
        """
        start_time = time.time()
        print("Ranking features by importance...")
        
        if self.mi_matrix is None:
            self.compute_mutual_information_matrix()
            
        # Extract mutual information with target column
        target_mi = self.mi_matrix[:, self.target_col_idx]
        
        # Normalize by entropy
        target_entropy = self.entropy[self.target_col_idx]
        feature_entropy = self.entropy
        
        # Compute normalized mutual information
        nmi = target_mi / np.sqrt(feature_entropy * target_entropy)
        
        # Sort by importance (descending)
        ranked_indices = np.argsort(-nmi)
        importance_scores = nmi[ranked_indices]
        
        # Create feature ranking dictionary
        feature_ranking = {
            self.feature_names[idx]: (score, rank) 
            for rank, (idx, score) in enumerate(zip(ranked_indices, importance_scores))
        }
        
        # Store as class attribute
        self.feature_importance = {
            'scores': nmi,
            'ranked_indices': ranked_indices,
            'ranked_names': [self.feature_names[idx] for idx in ranked_indices],
            'ranking': feature_ranking
        }
        
        print(f"Feature ranking completed in {time.time() - start_time:.2f} seconds")
        return ranked_indices, importance_scores
    
    def select_top_features(self, n_features=10, min_score=0.1):
        """
        Select top features based on importance scores.
        
        Parameters:
        -----------
        n_features : int
            Maximum number of features to select
        min_score : float
            Minimum importance score threshold
            
        Returns:
        --------
        list
            Indices of selected features
        """
        if self.feature_importance is None:
            # If ranking hasn't been done and we have MI matrix, do the ranking
            if self.mi_matrix is not None:
                self.rank_features_by_importance()
            else:
                # If no MI matrix, we can't rank, so just use first n_features
                print("WARNING: No feature importance data available. Using first N features.")
                return list(range(min(n_features, self.n_features)))
        
        # Get ranked indices and scores
        ranked_indices = self.feature_importance['ranked_indices']
        scores = self.feature_importance['scores']
        
        # Select features that meet both criteria
        selected_indices = [idx for idx, score in zip(ranked_indices, scores[ranked_indices]) 
                           if score >= min_score][:n_features]
        
        # If we don't have enough features meeting the criteria, just use top n
        if len(selected_indices) < n_features:
            selected_indices = ranked_indices[:n_features]
        
        return selected_indices
    
    def enhance_features(self, n_features=10, include_entropy=True, include_kl=True, 
                        include_transfer_entropy=False, include_high_signal=True, 
                        is_training=True, min_snr=2.0):
        """
        Create an enhanced feature set using information theory measures.
        
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
            Whether to include high signal-to-noise ratio features
        is_training : bool
            Whether this is the training phase (if False, skip MI computation and use fast approximation)
        min_snr : float
            Minimum signal-to-noise ratio for high signal features
            
        Returns:
        --------
        tuple
            (enhanced_features, enhanced_feature_names)
        """
        start_time = time.time()
        print("Creating enhanced feature set...")
        
        # Ensure all necessary calculations are done, but only if in training mode
        if is_training:
            if self.entropy is None:
                self.estimate_shannon_entropy()
            
            if self.mi_matrix is None:
                self.compute_mutual_information_matrix()
            
            if self.feature_importance is None:
                self.rank_features_by_importance()
            
            if not hasattr(self, 'kl_divergence') and include_kl:
                self.compute_kl_divergence()
            
            if self.transfer_entropy is None and include_transfer_entropy:
                self.compute_transfer_entropy()
        else:
            # In non-training mode, ensure we have entropy if needed
            if include_entropy and self.entropy is None:
                self.estimate_shannon_entropy()
            
            # Only compute MI if absolutely necessary 
            if self.mi_matrix is None and self.feature_importance is None:
                # For large datasets (millions of observations), use histogram approximation
                if self.n_samples > 100000:
                    print(f"Large dataset detected ({self.n_samples} samples). Computing mutual information using histogram approximation...")
                    self.compute_mutual_information_matrix(fast_approximation=True)
                else:
                    # For smaller datasets, use histogram approximation
                    print("Computing mutual information using histogram approximation for non-training phase...")
                    self.compute_mutual_information_matrix(fast_approximation=True)
                
                # Rank features after computing MI
                self.rank_features_by_importance()
        
        # Select top features
        top_indices = self.select_top_features(n_features)
        
        # Create base feature set
        enhanced_features = self.clean_data[:, top_indices]
        enhanced_names = [self.feature_names[idx] for idx in top_indices]
        
        # In non-training mode, we only return the base features to match training dimensions
        if not is_training:
            return enhanced_features, enhanced_names
        
        # Add additional features only during training phase
        # Add entropy-weighted features if requested
        if include_entropy and self.entropy is not None:
            entropy_weights = self.entropy[top_indices] / np.sum(self.entropy[top_indices])
            entropy_weighted = enhanced_features * entropy_weights.reshape(1, -1)
            
            enhanced_features = np.column_stack([enhanced_features, entropy_weighted])
            enhanced_names.extend([f"{name}_ent_weighted" for name in enhanced_names[:len(top_indices)]])
        
        # Add KL divergence features if requested
        if include_kl and hasattr(self, 'kl_divergence'):
            kl_features = self.kl_divergence[:, top_indices]
            
            enhanced_features = np.column_stack([enhanced_features, kl_features])
            enhanced_names.extend([f"{name}_kl" for name in enhanced_names[:len(top_indices)]])
        
        # Add transfer entropy features if requested
        if include_transfer_entropy and self.transfer_entropy is not None:
            # Get transfer entropy values for each feature with the target (volatility)
            if self.target_col_idx is not None:
                te_values = self.transfer_entropy[:, self.target_col_idx]
                # Use only top features
                te_values = te_values[top_indices]
                # Weight features by their transfer entropy
                te_weighted = enhanced_features[:, :len(top_indices)] * te_values.reshape(1, -1)
                
                enhanced_features = np.column_stack([enhanced_features, te_weighted])
                enhanced_names.extend([f"{name}_te_weighted" for name in enhanced_names[:len(top_indices)]])
        
        # Add high signal-to-noise features if requested
        if include_high_signal and self.mi_matrix is not None:
            high_signal_features = []
            high_signal_names = []
            
            # First compute signal-to-noise ratio for each feature
            for j in range(self.n_features):
                # Only process top features
                if j not in top_indices:
                    continue
                    
                # Signal strength: mutual information with target
                signal = self.mi_matrix[j, self.target_col_idx]
                
                # Noise estimate: average MI with non-target features
                noise_mi = np.delete(self.mi_matrix[j, :], [j, self.target_col_idx])
                noise = np.mean(noise_mi)
                
                # Compute SNR
                snr = signal / (noise + 1e-10)
                
                if snr >= min_snr:
                    feature_name = self.feature_names[j]
                    # Get the raw feature
                    feature = self.scaled_data[:, j]
                    
                    try:
                        # Create high signal feature variations
                        if hasattr(self, 'kl_divergence') and j < self.kl_divergence.shape[1]:
                            # 1. KL-weighted feature - check shape compatibility first
                            kl_feature = self.kl_divergence[:, j]
                            if len(kl_feature) == len(feature):
                                kl_weighted = feature * (1 + np.log1p(kl_feature))
                                high_signal_features.append(kl_weighted)
                                high_signal_names.append(f"{feature_name}_high_signal_kl")
                        
                        # 2. Tail-emphasized feature
                        tail_factor, _ = self._adjust_for_heavy_tails(feature.reshape(-1, 1))
                        tail_weighted = feature * tail_factor
                        high_signal_features.append(tail_weighted)
                        high_signal_names.append(f"{feature_name}_high_signal_tail")
                        
                        # 3. Information-ratio feature
                        info_ratio = signal / (np.sum(noise_mi) + 1e-10)
                        info_weighted = feature * info_ratio
                        high_signal_features.append(info_weighted)
                        high_signal_names.append(f"{feature_name}_high_signal_info")
                    except Exception as e:
                        print(f"Warning: Error creating high signal features for {feature_name}: {str(e)}")
                        continue
            
            if high_signal_features:
                try:
                    # Combine and normalize high signal features
                    high_signal_data = np.column_stack(high_signal_features)
                    high_signal_data = StandardScaler().fit_transform(high_signal_data)
                    enhanced_features = np.column_stack([enhanced_features, high_signal_data])
                    enhanced_names.extend(high_signal_names)
                except Exception as e:
                    print(f"Warning: Error combining high signal features: {str(e)}")
                    print("Continuing without high signal features")
        
        print(f"Feature enhancement completed in {time.time() - start_time:.2f} seconds")
        print(f"Enhanced feature set contains {enhanced_features.shape[1]} features")
        return enhanced_features, enhanced_names 