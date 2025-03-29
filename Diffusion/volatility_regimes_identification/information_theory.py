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
    
    def __init__(self, lambda_ent=0.5, bins=10):
        """
        Initialize the information theory enhancer.
        
        Parameters:
        -----------
        lambda_ent : float
            Weight for entropy in feature weighting (default 0.5)
        bins : int
            Number of bins for histogram-based estimations (default 10)
        """
        # Set parameters
        self.lambda_ent = lambda_ent
        self.bins = bins
        
        # Information-theoretic measures will be computed when features are provided
        self.feature_array = None
        self.feature_names = None
        self.target_col_idx = None
        self.n_samples = None
        self.n_features = None
        self.entropy = None
        self.mi_matrix = None
        self.feature_importance = None
        self.transfer_entropy = None
        self.clean_data = None
        self.scaled_data = None
        self.scaler = None

    def setup(self, feature_array, feature_names, target_col_idx=None):
        """
        Set up the feature array and preprocess it.
        
        Parameters:
        -----------
        feature_array : numpy.ndarray
            Array of shape (n_samples, n_features) containing extracted features
        feature_names : list
            List of feature names corresponding to columns in feature_array
        target_col_idx : int or None
            Index of the target variable (typically volatility) in feature_array
        """
        self.feature_array = np.copy(feature_array)
        self.feature_names = feature_names
        
        # If target index not provided, assume volatility is the target (look for it in names)
        if target_col_idx is None:
            try:
                volatility_indices = [i for i, name in enumerate(self.feature_names) 
                                   if 'volatil' in name.lower()]
                if volatility_indices:
                    target_col_idx = volatility_indices[0]
                else:
                    target_col_idx = -1  # Default to last column
            except ValueError:
                target_col_idx = -1  # Default to last column
                
        self.target_col_idx = target_col_idx
        self.n_samples, self.n_features = self.feature_array.shape
        
        # Preprocess features
        self._preprocess_features()
        
        return self

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
            
            # Mutual information can be approximated as E[log(p(x,y)/(p(x)p(y)))]
            # When using log densities: E[log_pxy - log_px - log_py]
            # With KDE, this can sometimes be negative due to estimation errors
            # Take absolute value or max with 0 to ensure non-negativity
            mi_raw = np.mean(log_pxy - log_px - log_py)
            mi = max(0, mi_raw)  # Ensure non-negativity
            
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
    
    def enhance_features(self, feature_array, feature_names, n_clusters=10, use_entropy=True, 
                         use_mi=True, use_te=True, use_clustering=True, use_log=True):
        """
        Enhance features using information theory metrics.
        
        Parameters:
        -----------
        feature_array : numpy.ndarray
            Array of shape (n_samples, n_features) containing features
        feature_names : list
            List of feature names corresponding to columns in feature_array
        n_clusters : int
            Number of clusters to form for dimensionality reduction
        use_entropy : bool
            Whether to use entropy weighting
        use_mi : bool
            Whether to use mutual information
        use_te : bool
            Whether to use transfer entropy
        use_clustering : bool
            Whether to use clustering-based enhancement
        use_log : bool
            Whether to use log transform for skewed features
            
        Returns:
        --------
        tuple
            (enhanced_features, enhanced_feature_names)
            Enhanced feature array and corresponding feature names
        """
        start_time = time.time()
        
        # Set up the feature array and preprocess it
        self.setup(feature_array, feature_names)
        
        # First, calculate entropy for feature weighting
        if self.entropy is None and use_entropy:
            self.estimate_shannon_entropy()
            
        # Calculate mutual information matrix
        if self.mi_matrix is None and use_mi:
            self.compute_mutual_information_matrix()
            
        # Calculate transfer entropy if needed
        if self.transfer_entropy is None and use_te:
            self.compute_transfer_entropy()
        
        # Calculate feature importance using multiple information criteria
        self._calculate_feature_importance(lambda_ent=self.lambda_ent)
        
        # Select top features based on their importance
        n_features = min(n_clusters, self.n_features)
        top_feature_indices = np.argsort(self.feature_importance)[-n_features:]
        top_features = self.feature_array[:, top_feature_indices]
        
        # Create enhanced feature matrix
        enhanced_features = top_features
        enhanced_feature_names = [self.feature_names[i] for i in top_feature_indices]
        
        # Optionally add entropy-weighted features
        if use_entropy and self.entropy is not None:
            # Apply log transformation for skewed features if requested
            if use_log:
                log_weights = np.log1p(self.entropy[top_feature_indices] + 1e-10)
                log_weights = log_weights / (np.max(log_weights) + 1e-10)  # Normalize
                weighted_features = top_features * log_weights.reshape(1, -1)
            else:
                # Min-max normalization of entropy
                weights = (self.entropy[top_feature_indices] - np.min(self.entropy) + 1e-10) / \
                         (np.max(self.entropy) - np.min(self.entropy) + 1e-10)
                weighted_features = top_features * weights.reshape(1, -1)
            
            # Add entropy-weighted features
            enhanced_features = np.hstack((enhanced_features, weighted_features))
            weighted_names = [f"{name}_ent_weighted" for name in enhanced_feature_names]
            enhanced_feature_names.extend(weighted_names)
            
        print(f"Feature enhancement took {time.time() - start_time:.2f} seconds")
        print(f"Enhanced features shape: {enhanced_features.shape}")
        
        return enhanced_features, enhanced_feature_names

    def enhance_features_adaptive(self, n_features=10, is_training=True, 
                                 alpha=0.6, beta=0.3, gamma=0.1, lambda_redundancy=0.5,
                                 use_progressive_selection=True, initial_threshold=0.3):
        """
        Enhanced feature selection and enhancement using a progressive, adaptive approach.
        This implements the proposed optimization strategy with progressive filtering,
        redundancy minimization, and adaptive enhancement.
        
        Parameters:
        -----------
        n_features : int
            Number of features to select
        is_training : bool
            Whether this is the training phase
        alpha : float
            Weight for normalized mutual information in relevance score
        beta : float
            Weight for normalized transfer entropy in relevance score
        gamma : float
            Weight for signal-to-noise ratio in relevance score
        lambda_redundancy : float
            Regularization parameter for redundancy penalty
        use_progressive_selection : bool
            Whether to use the progressive selection approach (if False, falls back to original method)
        initial_threshold : float
            Threshold for initial feature selection (fraction of max score)
            
        Returns:
        --------
        tuple
            (enhanced_features, enhanced_feature_names)
        """
        start_time = time.time()
        print("Creating enhanced feature set using adaptive approach...")
        
        # If not in training mode, or not using progressive selection, fall back to original method
        if not is_training or not use_progressive_selection:
            return self.enhance_features(
                n_features=n_features, 
                is_training=is_training,
                include_entropy=True,
                include_kl=True,
                include_transfer_entropy=True,
                include_high_signal=True
            )
        
        # Step 1: Compute fast information measures
        # Always compute entropy as it's relatively inexpensive
        if self.entropy is None:
            self.estimate_shannon_entropy()
        
        # Step 2: Create and use the progressive feature selector
        selector = ProgressiveFeatureSelector(
            self.feature_array,
            self.feature_names,
            alpha=alpha, 
            beta=beta, 
            gamma=gamma, 
            lambda_redundancy=lambda_redundancy
        )
        
        # Step 3: Perform progressive selection
        selected_indices, characteristics = selector.final_feature_selection(
            n_features=n_features,
            refine_with_te=True  # Compute transfer entropy for candidates
        )
        
        # Step 4: Get adaptive enhancement strategy for each feature
        enhancement_strategy = selector.get_adaptive_enhancement_strategy()
        
        # Step 5: Apply adaptive enhancement based on feature characteristics
        # Create base feature set first
        enhanced_features = self.clean_data[:, selected_indices]
        enhanced_names = [self.feature_names[idx] for idx in selected_indices]
        
        # Track added feature count for logging
        added_features = {
            'entropy_weighted': 0,
            'kl_weighted': 0,
            'te_weighted': 0,
            'tail_emphasized': 0
        }
        
        # Apply enhancements based on individual feature characteristics
        for i, idx in enumerate(selected_indices):
            strategy = enhancement_strategy[idx]
            feature_name = self.feature_names[idx]
            
            # Apply entropy weighting if recommended
            if strategy['include_entropy_weighted'] and self.entropy is not None:
                entropy_weight = self.entropy[idx]
                if entropy_weight > 0:
                    entropy_weighted = enhanced_features[:, i] * entropy_weight
                    enhanced_features = np.column_stack([enhanced_features, entropy_weighted])
                    enhanced_names.append(f"{feature_name}_ent_weighted")
                    added_features['entropy_weighted'] += 1
            
            # Apply transfer entropy weighting if recommended
            if strategy['include_te_weighted'] and hasattr(self, 'transfer_entropy') and self.transfer_entropy is not None:
                te_to_target = self.transfer_entropy[idx, self.target_col_idx]
                if te_to_target > 0:
                    te_weighted = enhanced_features[:, i] * te_to_target
                    enhanced_features = np.column_stack([enhanced_features, te_weighted])
                    enhanced_names.append(f"{feature_name}_te_weighted")
                    added_features['te_weighted'] += 1
            
            # Apply KL weighting if recommended
            if strategy['include_kl_weighted'] and hasattr(self, 'kl_divergence') and self.kl_divergence is not None:
                if idx < self.kl_divergence.shape[1]:
                    kl_feature = self.kl_divergence[:, idx]
                    if len(kl_feature) == enhanced_features.shape[0]:
                        kl_weighted = enhanced_features[:, i] * (1 + np.log1p(kl_feature))
                        enhanced_features = np.column_stack([enhanced_features, kl_weighted])
                        enhanced_names.append(f"{feature_name}_kl_weighted")
                        added_features['kl_weighted'] += 1
            
            # Apply tail emphasis if recommended
            if strategy['include_tail_emphasis']:
                feature = self.scaled_data[:, idx].reshape(-1, 1)
                tail_factor, _ = self._adjust_for_heavy_tails(feature)
                if tail_factor > 1.0:  # Only apply if there's significant tail effect
                    tail_weighted = enhanced_features[:, i] * tail_factor
                    enhanced_features = np.column_stack([enhanced_features, tail_weighted])
                    enhanced_names.append(f"{feature_name}_tail_emphasized")
                    added_features['tail_emphasized'] += 1
        
        print(f"Adaptive feature enhancement completed in {time.time() - start_time:.2f} seconds")
        print(f"Base features: {len(selected_indices)}")
        print(f"Added features: entropy-weighted: {added_features['entropy_weighted']}, "
              f"TE-weighted: {added_features['te_weighted']}, "
              f"KL-weighted: {added_features['kl_weighted']}, "
              f"tail-emphasized: {added_features['tail_emphasized']}")
        print(f"Total enhanced feature set contains {enhanced_features.shape[1]} features")
        
        return enhanced_features, enhanced_names

    def enhance_features_batch(self, n_features=10, include_entropy=True):
        """
        Lightweight feature enhancement method optimized for batch processing.
        Uses fast histogram approximations and simplified enhancement to maximize throughput.
        
        Parameters:
        -----------
        n_features : int
            Number of features to select
        include_entropy : bool
            Whether to include entropy weighting
            
        Returns:
        --------
        tuple
            (enhanced_features, enhanced_feature_names)
        """
        start_time = time.time()
        print("Creating batch-optimized feature set...")
        
        # Ensure entropy is computed (fast computation)
        if self.entropy is None:
            self.estimate_shannon_entropy()
        
        # Create a ProgressiveFeatureSelector for fast selection
        selector = ProgressiveFeatureSelector(self.feature_array, self.feature_names)
        
        # Use fast feature selection optimized for batch processing
        selected_indices = selector.fast_feature_selection(
            n_features=n_features,
            include_redundancy=True
        )
        
        # Create base feature set
        enhanced_features = self.clean_data[:, selected_indices]
        enhanced_names = [self.feature_names[idx] for idx in selected_indices]
        
        # Only apply entropy weighting if requested (other enhancements are skipped for speed)
        if include_entropy and self.entropy is not None:
            # Get entropy values for selected features
            feature_entropy = self.entropy[selected_indices]
            
            # Normalize entropy weights
            total_entropy = np.sum(feature_entropy)
            if total_entropy > 0:
                entropy_weights = feature_entropy / total_entropy
                
                # Create entropy-weighted features
                entropy_weighted = enhanced_features * entropy_weights.reshape(1, -1)
                enhanced_features = np.column_stack([enhanced_features, entropy_weighted])
                enhanced_names.extend([f"{name}_ent_weighted" for name in enhanced_names[:len(selected_indices)]])
        
        print(f"Batch feature enhancement completed in {time.time() - start_time:.2f} seconds")
        print(f"Enhanced feature set contains {enhanced_features.shape[1]} features")
        
        return enhanced_features, enhanced_names

    def _calculate_feature_importance(self, lambda_ent=0.5, lambda_mi=0.5):
        """
        Calculate feature importance using a combination of entropy and mutual information.
        
        Parameters:
        -----------
        lambda_ent : float
            Weight for entropy in importance calculation
        lambda_mi : float
            Weight for mutual information in importance calculation
            
        Returns:
        --------
        numpy.ndarray
            Array of feature importance scores
        """
        if self.entropy is None:
            self.estimate_shannon_entropy()
            
        if self.mi_matrix is None:
            self.compute_mutual_information_matrix()
            
        # Normalize entropy
        norm_entropy = (self.entropy - np.min(self.entropy) + 1e-10) / (np.max(self.entropy) - np.min(self.entropy) + 1e-10)
        
        # For MI, use the average MI with other features if no target
        if self.target_col_idx is not None and self.target_col_idx >= 0:
            # Use MI with target
            mi_with_target = self.mi_matrix[:, self.target_col_idx]
            # Normalize MI
            norm_mi = (mi_with_target - np.min(mi_with_target) + 1e-10) / (np.max(mi_with_target) - np.min(mi_with_target) + 1e-10)
        else:
            # Use average MI with other features
            avg_mi = np.zeros(self.n_features)
            for i in range(self.n_features):
                # Exclude self-MI
                mi_values = np.delete(self.mi_matrix[i, :], i)
                avg_mi[i] = np.mean(mi_values)
            # Normalize average MI
            norm_mi = (avg_mi - np.min(avg_mi) + 1e-10) / (np.max(avg_mi) - np.min(avg_mi) + 1e-10)
        
        # Combine entropy and MI for feature importance
        self.feature_importance = lambda_ent * norm_entropy + lambda_mi * norm_mi
        
        # Ensure all importance values are positive
        self.feature_importance = np.maximum(self.feature_importance, 0)
        
        return self.feature_importance

class ProgressiveFeatureSelector:
    """
    Progressive feature selection using information-theoretic measures.
    Implements a two-stage selection process with lazy evaluation of expensive measures.
    
    Key components:
    1. Fast initial filtering using histogram-based MI and entropy
    2. Redundancy minimization for feature selection
    3. Advanced relevance scoring incorporating multiple information measures
    4. Adaptive enhancement based on feature characteristics
    """
    
    def __init__(self, features, feature_names, alpha=0.6, beta=0.3, gamma=0.1, lambda_redundancy=0.5):
        """
        Initialize the progressive feature selector.
        
        Parameters:
        -----------
        features : numpy.ndarray
            Feature array of shape (n_samples, n_features)
        feature_names : list
            List of feature names
        alpha : float
            Weight for normalized mutual information in relevance score
        beta : float
            Weight for transfer entropy in relevance score
        gamma : float
            Weight for signal-to-noise ratio in relevance score
        lambda_redundancy : float
            Regularization parameter for redundancy penalty
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_redundancy = lambda_redundancy
        
        # Store feature data directly
        self.feature_array = features
        self.feature_names = feature_names
        self.n_samples, self.n_features = features.shape
        
        # Try to identify target column (volatility-related feature)
        self.target_col_idx = -1  # Default to last column
        for i, name in enumerate(feature_names):
            if 'volatil' in name.lower():
                self.target_col_idx = i
                break
        
        # Selection results
        self.relevance_scores = None
        self.redundancy_matrix = None
        self.selected_indices = None
        self.feature_characteristics = None
        
        # Compute information-theoretic metrics for selection
        self.entropy = None
        self.mi_matrix = None
        self.transfer_entropy = None

    def compute_initial_scores(self, fast_approximation=True, bins=64):
        """
        Compute initial feature relevance scores using fast approximations.
        This provides a quick first-pass filtering of features.
        
        Parameters:
        -----------
        fast_approximation : bool
            Whether to use fast histogram-based approximations
        bins : int
            Number of bins for histogram approximation
            
        Returns:
        --------
        numpy.ndarray
            Array of initial relevance scores for each feature
        """
        print("Computing initial feature relevance scores...")
        start_time = time.time()
        
        # Ensure entropy is computed (fast computation)
        if self.entropy is None:
            self.entropy = self.compute_shannon_entropy()
        
        # Compute fast mutual information if needed
        if self.mi_matrix is None:
            if fast_approximation:
                self.mi_matrix = self.compute_mutual_information_histogram(n_bins=bins)
            else:
                self.mi_matrix = self.compute_mutual_information_matrix()
        
        # Calculate normalized mutual information with target
        target_entropy = self.entropy[self.target_col_idx]
        feature_entropy = self.entropy
        target_mi = self.mi_matrix[:, self.target_col_idx]
        
        # Compute normalized mutual information (NMI)
        nmi = target_mi / np.sqrt(feature_entropy * target_entropy)
        nmi = np.nan_to_num(nmi, nan=0.0)
        
        # Compute signal-to-noise ratio for each feature
        snr = np.zeros(self.n_features)
        for j in range(self.n_features):
            # Signal: MI with target
            signal = self.mi_matrix[j, self.target_col_idx]
            
            # Noise: average MI with other features (excluding self and target)
            noise_mi = np.delete(self.mi_matrix[j, :], [j, self.target_col_idx])
            noise = np.mean(noise_mi) if len(noise_mi) > 0 else 0
            
            # Compute SNR
            snr[j] = signal / (noise + 1e-10)
        
        # Combine into initial relevance score (without transfer entropy)
        # We'll add transfer entropy later for promising features
        initial_scores = self.alpha * nmi + self.gamma * snr / np.max(snr)
        
        print(f"Initial scoring completed in {time.time() - start_time:.2f} seconds")
        print(f"Top 5 features by initial score:")
        top_indices = np.argsort(-initial_scores)[:5]
        for idx in top_indices:
            print(f"  {self.feature_names[idx]}: {initial_scores[idx]:.4f}")
            
        return initial_scores
    
    def compute_feature_redundancy(self, candidate_indices):
        """
        Compute redundancy between features to minimize information overlap.
        
        Parameters:
        -----------
        candidate_indices : list
            Indices of candidate features to analyze for redundancy
            
        Returns:
        --------
        numpy.ndarray
            Redundancy matrix for candidate features
        """
        n_candidates = len(candidate_indices)
        redundancy = np.zeros((n_candidates, n_candidates))
        
        # Use MI to compute normalized redundancy
        for i, idx_i in enumerate(candidate_indices):
            for j, idx_j in enumerate(candidate_indices):
                if i == j:
                    continue
                    
                # Mutual information between features
                mi_ij = self.mi_matrix[idx_i, idx_j]
                
                # Normalize by minimum entropy
                h_i = self.entropy[idx_i]
                h_j = self.entropy[idx_j]
                min_entropy = min(h_i, h_j)
                
                if min_entropy > 0:
                    # Redundancy is normalized mutual information between features
                    redundancy[i, j] = mi_ij / min_entropy
                    
        return redundancy
        
    def first_stage_selection(self, initial_threshold=0.3, max_candidates=30):
        """
        Perform first-stage feature selection using initial scores.
        
        Parameters:
        -----------
        initial_threshold : float
            Threshold for initial feature selection (fraction of max score)
        max_candidates : int
            Maximum number of candidate features to retain
            
        Returns:
        --------
        list
            Indices of candidate features passing first-stage selection
        """
        # Compute initial scores if not already done
        if self.relevance_scores is None:
            self.relevance_scores = self.compute_initial_scores()
        
        # Calculate threshold value
        threshold = initial_threshold * np.max(self.relevance_scores)
        
        # Select features above threshold
        candidates = np.where(self.relevance_scores >= threshold)[0]
        
        # Limit number of candidates if needed
        if len(candidates) > max_candidates:
            candidates = np.argsort(-self.relevance_scores)[:max_candidates]
            
        print(f"First-stage selection: {len(candidates)} candidates from {self.n_features} features")
        return sorted(candidates)
    
    def compute_transfer_entropy_for_candidates(self, candidate_indices):
        """
        Compute transfer entropy selectively for candidate features.
        
        Parameters:
        -----------
        candidate_indices : list
            Indices of features to compute transfer entropy for
            
        Returns:
        --------
        numpy.ndarray
            Transfer entropy matrix for candidate features
        """
        if self.transfer_entropy is None:
            # Create a reduced version of the feature array with only candidates
            # and target to speed up computation
            reduced_features = np.zeros((self.n_samples, len(candidate_indices) + 1))
            
            # Add candidate features
            for i, idx in enumerate(candidate_indices):
                reduced_features[:, i] = self.feature_array[:, idx]
                
            # Add target as the last column
            reduced_features[:, -1] = self.feature_array[:, self.target_col_idx]
            
            # Create temporary enhancer for transfer entropy computation
            temp_names = [self.feature_names[idx] for idx in candidate_indices] + [self.feature_names[self.target_col_idx]]
            temp_enhancer = InformationTheoryEnhancer(reduced_features, temp_names, target_col_idx=len(candidate_indices))
            
            # Compute transfer entropy only for this reduced set
            temp_enhancer.estimate_shannon_entropy()
            te_matrix = temp_enhancer.compute_transfer_entropy(lag=1)
            
            return te_matrix
        else:
            # If transfer entropy is already computed for all features, extract the subset
            full_te = self.transfer_entropy
            # Add target to candidate indices if not already included
            if self.target_col_idx not in candidate_indices:
                indices = candidate_indices + [self.target_col_idx]
            else:
                indices = candidate_indices
                
            # Extract submatrix
            return full_te[np.ix_(indices, indices)]
    
    def compute_enhancement_potential(self, candidate_indices, te_matrix):
        """
        Compute enhancement potential for features based on transfer entropy.
        
        Parameters:
        -----------
        candidate_indices : list
            Indices of candidate features
        te_matrix : numpy.ndarray
            Transfer entropy matrix for candidate features
            
        Returns:
        --------
        numpy.ndarray
            Enhancement potential scores for candidate features
        """
        enhancement_potential = np.zeros(len(candidate_indices))
        
        # Identify target index in the transfer entropy matrix
        target_idx = len(candidate_indices)  # If target is the last column in the reduced matrix
        
        for i, idx in enumerate(candidate_indices):
            # Transfer entropy from feature to target
            te_to_target = te_matrix[i, target_idx]
            
            # Mutual information with target
            mi_with_target = self.mi_matrix[idx, self.target_col_idx]
            
            # Enhancement potential: ratio of TE to MI, higher means more causal information
            if mi_with_target > 0:
                enhancement_potential[i] = te_to_target / mi_with_target
            
        return enhancement_potential
    
    def final_feature_selection(self, n_features=10, refine_with_te=True):
        """
        Perform final feature selection using a greedy approach to maximize
        relevance and minimize redundancy.
        
        Parameters:
        -----------
        n_features : int
            Number of features to select
        refine_with_te : bool
            Whether to refine scores using transfer entropy
            
        Returns:
        --------
        tuple
            (selected_indices, feature_characteristics) where feature_characteristics
            contains information about each selected feature for adaptive enhancement
        """
        print(f"Performing final feature selection to select {n_features} features...")
        start_time = time.time()
        
        # First stage selection to get candidates
        candidates = self.first_stage_selection()
        
        # Refine scores with transfer entropy if requested
        if refine_with_te:
            print("Computing transfer entropy for candidate features...")
            te_matrix = self.compute_transfer_entropy_for_candidates(candidates)
            
            # Compute enhancement potential for each feature
            enhancement_potential = self.compute_enhancement_potential(candidates, te_matrix)
            
            # Update relevance scores with transfer entropy component
            for i, idx in enumerate(candidates):
                # TE from feature to target, normalized by entropy
                if idx < len(self.entropy) and self.entropy[idx] > 0:
                    te_component = te_matrix[i, -1] / self.entropy[idx]  # Assuming target is last
                    self.relevance_scores[idx] += self.beta * te_component
        
        # Compute redundancy among candidates
        redundancy = self.compute_feature_redundancy(candidates)
        
        # Store redundancy for later use
        self.redundancy_matrix = redundancy
        
        # Greedy selection of features
        selected = []
        remaining = set(candidates)
        
        # Feature characteristics for adaptive enhancement
        characteristics = {}
        
        # Keep track of indices mapping between candidates and original features
        candidate_to_idx = {i: idx for i, idx in enumerate(candidates)}
        
        # First, select the feature with highest relevance score
        best_first = np.argmax([self.relevance_scores[idx] for idx in candidates])
        first_idx = candidates[best_first]
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Initialize characteristics for first feature
        characteristics[first_idx] = {
            'name': self.feature_names[first_idx],
            'relevance': self.relevance_scores[first_idx],
            'entropy': self.entropy[first_idx] if first_idx < len(self.entropy) else 0,
            'snr': 0,  # Will be computed later
            'heavy_tailed': False,  # Will be determined later
            'te_importance': 0  # Will be updated if TE is available
        }
        
        # Iteratively select remaining features
        while len(selected) < n_features and remaining:
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining:
                # Get original index and its position in candidates list
                candidate_idx = candidates.index(idx)
                
                # Calculate redundancy penalty: max redundancy with already selected features
                redundancy_penalty = 0
                for sel_idx in selected:
                    sel_candidate_idx = candidates.index(sel_idx)
                    if candidate_idx < len(redundancy) and sel_candidate_idx < len(redundancy):
                        current_redundancy = redundancy[candidate_idx, sel_candidate_idx]
                        redundancy_penalty = max(redundancy_penalty, current_redundancy)
                
                # Apply redundancy penalty to score
                adjusted_score = self.relevance_scores[idx] - self.lambda_redundancy * redundancy_penalty
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
                
                # Store feature characteristics for adaptive enhancement
                characteristics[best_idx] = {
                    'name': self.feature_names[best_idx],
                    'relevance': self.relevance_scores[best_idx],
                    'entropy': self.entropy[best_idx] if best_idx < len(self.entropy) else 0,
                    'snr': 0,  # Will be computed later
                    'heavy_tailed': False,  # Will be determined later
                    'te_importance': 0  # Will be updated if TE is available
                }
        
        # Compute additional characteristics for selected features
        for idx in selected:
            # Compute signal-to-noise ratio
            signal = self.mi_matrix[idx, self.target_col_idx]
            noise_mi = np.delete(self.mi_matrix[idx, :], [idx, self.target_col_idx])
            noise = np.mean(noise_mi) if len(noise_mi) > 0 else 0
            snr = signal / (noise + 1e-10)
            characteristics[idx]['snr'] = snr
            
            # Check if heavy-tailed
            feature = self.feature_array[:, idx]
            try:
                k = kurtosis(feature)
                characteristics[idx]['heavy_tailed'] = k > 2.0
            except:
                characteristics[idx]['heavy_tailed'] = False
            
            # Update transfer entropy importance if available
            if refine_with_te and hasattr(self, 'transfer_entropy') and self.transfer_entropy is not None:
                te_to_target = self.transfer_entropy[idx, self.target_col_idx]
                characteristics[idx]['te_importance'] = te_to_target
        
        # Store results
        self.selected_indices = selected
        self.feature_characteristics = characteristics
        
        print(f"Final feature selection completed in {time.time() - start_time:.2f} seconds")
        print(f"Selected {len(selected)} features with adaptive characteristics")
        
        # Print selected features and their properties
        print("\nSelected features:")
        for i, idx in enumerate(selected):
            char = characteristics[idx]
            print(f"{i+1}. {char['name']} (relevance: {char['relevance']:.4f}, "
                  f"SNR: {char['snr']:.2f}, heavy-tailed: {char['heavy_tailed']})")
        
        return selected, characteristics
    
    def get_adaptive_enhancement_strategy(self):
        """
        Determine which enhancement techniques to apply to each selected feature.
        
        Returns:
        --------
        dict
            Enhancement strategy for each selected feature
        """
        if self.selected_indices is None or self.feature_characteristics is None:
            raise ValueError("Feature selection must be performed first")
            
        enhancement_strategy = {}
        
        for idx in self.selected_indices:
            char = self.feature_characteristics[idx]
            
            # Initialize strategy with base feature inclusion
            strategy = {
                'include_base': True,
                'include_entropy_weighted': False,
                'include_kl_weighted': False,
                'include_te_weighted': False,
                'include_tail_emphasis': False
            }
            
            # Apply entropy weighting to high-entropy features
            if char['entropy'] > np.median([self.feature_characteristics[i]['entropy'] 
                                           for i in self.selected_indices]):
                strategy['include_entropy_weighted'] = True
            
            # Apply TE weighting to features with high causal influence
            if char['te_importance'] > 0.01:
                strategy['include_te_weighted'] = True
            
            # Apply tail emphasis to heavy-tailed features
            if char['heavy_tailed']:
                strategy['include_tail_emphasis'] = True
            
            # Consider KL divergence for features with temporal dynamics
            # This would need additional analysis, so we'll use a placeholder condition
            if hasattr(self, 'mi_matrix') and self.mi_matrix is not None:
                # Check if KL values are significant for this feature
                if idx < self.mi_matrix.shape[1]:
                    kl_values = self.mi_matrix[:, idx]
                    if np.mean(kl_values) > 0.1:  # Arbitrary threshold
                        strategy['include_kl_weighted'] = True
            
            enhancement_strategy[idx] = strategy
        
        return enhancement_strategy
    
    def fast_feature_selection(self, n_features=10, include_redundancy=False):
        """
        Fast feature selection for extension batch processing.
        Uses only histogram-based MI estimation for efficiency.
        
        Parameters:
        -----------
        n_features : int
            Number of features to select
        include_redundancy : bool
            Whether to include redundancy minimization (slower but more effective)
            
        Returns:
        --------
        list
            Indices of selected features
        """
        print("Performing fast feature selection for batch processing...")
        start_time = time.time()
        
        # Ensure entropy is computed (fast computation)
        if self.entropy is None:
            self.entropy = self.compute_shannon_entropy()
        
        # Compute fast mutual information if needed
        if self.mi_matrix is None:
            self.mi_matrix = self.compute_mutual_information_histogram()
            
        # Calculate normalized mutual information with target
        target_entropy = self.entropy[self.target_col_idx]
        feature_entropy = self.entropy
        target_mi = self.mi_matrix[:, self.target_col_idx]
        
        # Compute normalized mutual information (NMI)
        nmi = target_mi / np.sqrt(feature_entropy * target_entropy)
        nmi = np.nan_to_num(nmi, nan=0.0)
        
        if not include_redundancy:
            # Simple approach: select top features by NMI only
            selected = np.argsort(-nmi)[:n_features]
        else:
            # More advanced approach with redundancy minimization
            # First select top candidate features (30% more than needed)
            n_candidates = min(int(n_features * 1.3) + 2, self.n_features)
            candidates = np.argsort(-nmi)[:n_candidates]
            
            # Compute redundancy among candidates
            redundancy = self.compute_feature_redundancy(candidates)
            
            # Greedy selection with redundancy minimization
            selected = []
            remaining = set(candidates)
            
            # First, select the feature with highest NMI
            best_first = np.argmax([nmi[idx] for idx in candidates])
            first_idx = candidates[best_first]
            selected.append(first_idx)
            remaining.remove(first_idx)
            
            # Iteratively select remaining features
            lambda_redundancy = 0.3  # Lower weight for redundancy in fast selection
            
            while len(selected) < n_features and remaining:
                best_score = -float('inf')
                best_idx = None
                
                for idx in remaining:
                    # Calculate redundancy penalty: max redundancy with already selected features
                    redundancy_penalty = 0
                    
                    for sel_idx in selected:
                        # Find these indices in the candidates list
                        sel_candidate_idx = np.where(candidates == sel_idx)[0][0]
                        idx_candidate_idx = np.where(candidates == idx)[0][0]
                        
                        if idx_candidate_idx < len(redundancy) and sel_candidate_idx < len(redundancy):
                            current_redundancy = redundancy[idx_candidate_idx, sel_candidate_idx]
                            redundancy_penalty = max(redundancy_penalty, current_redundancy)
                    
                    # Apply redundancy penalty to NMI score
                    adjusted_score = nmi[idx] - lambda_redundancy * redundancy_penalty
                    
                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_idx = idx
                
                if best_idx is not None:
                    selected.append(best_idx)
                    remaining.remove(best_idx)
        
        print(f"Fast feature selection completed in {time.time() - start_time:.2f} seconds")
        print(f"Selected {len(selected)} features")
        
        return list(selected)

    def compute_shannon_entropy(self, bins=10):
        """
        Compute Shannon entropy for each feature using histogram approximation.
        
        Parameters:
        -----------
        bins : int
            Number of bins for histogram
            
        Returns:
        --------
        numpy.ndarray
            Entropy values for each feature
        """
        n_features = self.feature_array.shape[1]
        entropy = np.zeros(n_features)
        
        for i in range(n_features):
            # Get feature values
            x = self.feature_array[:, i]
            
            # Create histogram
            hist, _ = np.histogram(x, bins=bins, density=True)
            
            # Remove zeros and calculate entropy
            hist = hist[hist > 0]
            entropy[i] = -np.sum(hist * np.log(hist)) / bins
            
        return entropy
    
    def compute_mutual_information_histogram(self, n_bins=64, max_sample_size=100000):
        """
        Compute mutual information using histogram-based approximation.
        
        Parameters:
        -----------
        n_bins : int
            Number of bins for histogram
        max_sample_size : int
            Maximum sample size to use for computation
            
        Returns:
        --------
        numpy.ndarray
            Mutual information matrix
        """
        # Check if we need to subsample (for memory efficiency with huge datasets)
        if self.n_samples > max_sample_size:
            indices = np.random.choice(self.n_samples, max_sample_size, replace=False)
            data = self.feature_array[indices]
        else:
            data = self.feature_array
            
        n_features = data.shape[1]
        mi_matrix = np.zeros((n_features, n_features))
        
        # Compute mutual information for each pair of features
        for i in range(n_features):
            for j in range(i, n_features):
                if i == j:
                    # Self-mutual information is entropy
                    if self.entropy is not None:
                        mi_matrix[i, i] = self.entropy[i]
                    else:
                        x = data[:, i]
                        hist, _ = np.histogram(x, bins=n_bins, density=True)
                        hist = hist[hist > 0]
                        mi_matrix[i, i] = -np.sum(hist * np.log(hist)) / n_bins
                else:
                    # Compute mutual information between features i and j
                    x = data[:, i]
                    y = data[:, j]
                    
                    # Create 2D histogram
                    hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
                    
                    # Normalize to get joint probability
                    hist_2d = hist_2d / float(np.sum(hist_2d))
                    
                    # Compute marginal probabilities
                    hist_x = np.sum(hist_2d, axis=1)
                    hist_y = np.sum(hist_2d, axis=0)
                    
                    # Compute mutual information
                    mi = 0.0
                    for k in range(n_bins):
                        for l in range(n_bins):
                            if hist_2d[k, l] > 0 and hist_x[k] > 0 and hist_y[l] > 0:
                                mi += hist_2d[k, l] * np.log(hist_2d[k, l] / (hist_x[k] * hist_y[l]))
                                
                    # Store in matrix (symmetric)
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi
                    
        return mi_matrix
        
    def compute_mutual_information_matrix(self):
        """
        Wrapper for compatibility with enhancer method.
        
        Returns:
        --------
        numpy.ndarray
            Mutual information matrix
        """
        return self.compute_mutual_information_histogram()

    def select_features(self, n_features=10, include_redundancy=False):
        """
        Select features using two-stage process.
        
        Parameters:
        -----------
        n_features : int
            Number of features to select
        include_redundancy : bool
            Whether to include redundancy minimization
            
        Returns:
        --------
        list
            Indices of selected features
        """
        # First stage: initial scoring and filtering
        candidate_indices = self.first_stage_selection(initial_threshold=0.2, max_candidates=min(30, self.n_features))
        
        # If we only need a fast selection without redundancy minimization
        if not include_redundancy or len(candidate_indices) <= n_features:
            # Simply return top n_features from candidates
            if len(candidate_indices) <= n_features:
                return candidate_indices
            else:
                # Compute initial scores if not done yet
                if self.relevance_scores is None:
                    self.relevance_scores = self.compute_initial_scores()
                # Get top n_features by relevance score
                return sorted(candidate_indices[:n_features])
        
        # Compute redundancy matrix for candidates
        redundancy = self.compute_feature_redundancy(candidate_indices)
        
        # Final feature selection
        selected = []
        remaining = candidate_indices.copy()
        
        # First, select the most relevant feature
        if self.relevance_scores is None:
            self.relevance_scores = self.compute_initial_scores()
        
        # Get index of most relevant feature
        best_idx = remaining[np.argmax([self.relevance_scores[i] for i in remaining])]
        selected.append(best_idx)
        remaining.remove(best_idx)
        
        # Select remaining features
        while len(selected) < n_features and remaining:
            best_score = -float('inf')
            best_feature = None
            
            for idx in remaining:
                # Relevance score
                relevance = self.relevance_scores[idx]
                
                # Redundancy penalty
                redundancy_penalty = 0
                for sel_idx in selected:
                    # Find positions in candidate_indices
                    sel_pos = candidate_indices.index(sel_idx)
                    idx_pos = candidate_indices.index(idx)
                    redundancy_penalty += redundancy[sel_pos, idx_pos]
                
                # Compute final score
                if len(selected) > 0:
                    redundancy_penalty /= len(selected)
                score = relevance - self.lambda_redundancy * redundancy_penalty
                
                if score > best_score:
                    best_score = score
                    best_feature = idx
            
            if best_feature is not None:
                selected.append(best_feature)
                remaining.remove(best_feature)
            else:
                break
        
        return selected 