import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import kurtosis
import warnings
import itertools

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
        Estimate Shannon entropy using cosine kernel with heavy-tail emphasis.
        """
        print("Estimating Shannon entropy with heavy-tail adjustment...")
        
        self.entropy = np.zeros(self.n_features)
        
        for j in range(self.n_features):
            x = self.scaled_data[:, j].reshape(-1, 1)
            
            # Adjust for heavy tails
            tail_factor, weights = self._adjust_for_heavy_tails(x)
            
            # Determine base bandwidth
            if bandwidth_method == 'silverman':
                bw = 1.06 * np.std(x) * self.n_samples**(-1/5)
            else:
                bw = 3.49 * np.std(x) * self.n_samples**(-1/3)
            
            # Adjust bandwidth for heavy tails
            bw *= tail_factor
            
            # Apply weights by replicating points in tails
            x_weighted = []
            for i, point in enumerate(x):
                # Add more copies of points in the tails
                n_copies = int(np.ceil(weights[i]))
                x_weighted.extend([point] * n_copies)
            x_weighted = np.array(x_weighted).reshape(-1, 1)
            
            # Fit kernel density with cosine kernel on weighted points
            kde = KernelDensity(
                bandwidth=bw,
                kernel='cosine'  # Using cosine kernel for better tail behavior
            )
            kde.fit(x_weighted)
            
            # Evaluate density at sample points with more points in tails
            std = np.std(x)
            eval_points = np.linspace(x.min() - std * tail_factor, 
                                    x.max() + std * tail_factor,
                                    num=min(1000, self.n_samples))
            eval_points = eval_points.reshape(-1, 1)
            
            log_dens = kde.score_samples(eval_points)
            dens = np.exp(log_dens)
            
            # Normalize densities
            dens = dens / np.sum(dens)
            
            # Calculate entropy with tail emphasis
            entropy_j = -np.sum(dens * np.log(dens + 1e-10))
            self.entropy[j] = entropy_j
            
        return self
        
    def compute_mutual_information_matrix(self):
        """
        Compute mutual information between all pairs of features using KDE.
        Vectorized implementation for better performance.
        
        I(X;Y) = ∫∫ p(x,y) log(p(x,y)/(p(x)p(y))) dx dy
        
        Returns:
        --------
        numpy.ndarray
            Array of shape (n_features, n_features) containing mutual information values
        """
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
            
            # Compute bandwidths for joint distribution
            bw_x = bandwidths[i]
            bw_y = bandwidths[j]
            bw_joint = np.mean([bw_x, bw_y])
            
            # Fit KDEs
            kde_x = KernelDensity(bandwidth=bw_x, kernel='cosine').fit(x)
            kde_y = KernelDensity(bandwidth=bw_y, kernel='cosine').fit(y)
            kde_joint = KernelDensity(bandwidth=bw_joint, kernel='cosine').fit(np.column_stack([x, y]))
            
            # Evaluate densities at sample points
            log_px = kde_x.score_samples(x)
            log_py = kde_y.score_samples(y)
            log_pxy = kde_joint.score_samples(np.column_stack([x, y]))
            
            # Compute mutual information with tail emphasis
            mi = np.mean(log_pxy - log_px - log_py)
            tail_emphasis = np.maximum(tail_factors[i], tail_factors[j])
            mi *= tail_emphasis
            
            # Fill symmetric matrix
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
        
        # Fill diagonal with entropy values
        np.fill_diagonal(mi_matrix, self.entropy)
        
        # Store as class attribute and return
        self.mi_matrix = mi_matrix
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
        print("Computing transfer entropy between variables...")
        
        # Initialize transfer entropy matrix
        te_matrix = np.zeros((self.n_features, self.n_features))
        
        # Simplified transfer entropy calculation based on conditional mutual information
        for i in range(self.n_features):
            target_future = self.scaled_data[lag:, i]
            target_present = self.scaled_data[:-lag, i]
            
            for j in range(self.n_features):
                if i == j:
                    continue
                    
                source_present = self.scaled_data[:-lag, j]
                
                # Calculate entropy terms using binning approach
                nbins = max(5, min(20, int(np.sqrt(len(target_present) / 5))))
                
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
                
        # Store as class attribute and return
        self.transfer_entropy = te_matrix
        return te_matrix
        
    def _binned_conditional_entropy(self, y, x, bins):
        """Helper method to calculate conditional entropy using binning."""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            
        # Bin the data
        y_bins = np.linspace(y.min(), y.max(), bins+1)
        y_digitized = np.digitize(y, y_bins) - 1
        
        if x.shape[1] == 1:
            x_bins = np.linspace(x.min(), x.max(), bins+1)
            x_digitized = np.digitize(x, x_bins) - 1
        else:
            # If x is multidimensional, bin each dimension separately
            x_digitized = np.zeros((x.shape[0], x.shape[1]), dtype=int)
            for j in range(x.shape[1]):
                x_bins = np.linspace(x[:, j].min(), x[:, j].max(), bins+1)
                x_digitized[:, j] = np.digitize(x[:, j], x_bins) - 1
                
        # Calculate joint and conditional counts
        conditional_entropy = 0
        
        # Handle different dimensions
        if x.shape[1] == 1:
            for i in range(bins):
                x_mask = x_digitized == i
                p_x = np.mean(x_mask)
                
                if p_x > 0:
                    # Calculate entropy of y given x=i
                    y_given_x = y_digitized[x_mask]
                    
                    if len(y_given_x) > 0:
                        y_counts = np.bincount(y_given_x, minlength=bins)
                        y_probs = y_counts / np.sum(y_counts)
                        
                        # Calculate entropy
                        entropy_y_given_x = -np.sum(y_probs * np.log(y_probs + 1e-10))
                        conditional_entropy += p_x * entropy_y_given_x
        else:
            # For multidimensional x, we'll use a simplified approach with unique combinations
            x_combinations = np.array([x_digitized[:, j] * (bins**(j)) for j in range(x.shape[1])]).T
            x_flat = np.sum(x_combinations, axis=1)
            
            unique_x = np.unique(x_flat)
            for x_val in unique_x:
                x_mask = x_flat == x_val
                p_x = np.mean(x_mask)
                
                if p_x > 0:
                    y_given_x = y_digitized[x_mask]
                    
                    if len(y_given_x) > 0:
                        y_counts = np.bincount(y_given_x, minlength=bins)
                        y_probs = y_counts / np.sum(y_counts)
                        
                        entropy_y_given_x = -np.sum(y_probs * np.log(y_probs + 1e-10))
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
            self.rank_features_by_importance()
            
        # Get ranked indices and scores
        ranked_indices = self.feature_importance['ranked_indices']
        scores = self.feature_importance['scores']
        
        # Select features that meet both criteria
        selected_indices = [idx for idx, score in zip(ranked_indices, scores[ranked_indices]) 
                           if score >= min_score][:n_features]
        
        return selected_indices
    
    def enhance_features(self, n_features=10, include_entropy=True, include_kl=True):
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
            
        Returns:
        --------
        tuple
            (enhanced_features, enhanced_feature_names)
        """
        print("Creating enhanced feature set...")
        
        # Ensure all necessary calculations are done
        if self.entropy is None:
            self.estimate_shannon_entropy()
            
        if self.mi_matrix is None:
            self.compute_mutual_information_matrix()
            
        if self.feature_importance is None:
            self.rank_features_by_importance()
            
        if not hasattr(self, 'kl_divergence') and include_kl:
            self.compute_kl_divergence()
            
        # Select top features
        top_indices = self.select_top_features(n_features)
        
        # Create base feature set
        enhanced_features = self.clean_data[:, top_indices]
        enhanced_names = [self.feature_names[idx] for idx in top_indices]
        
        # Add entropy-weighted features if requested
        if include_entropy:
            entropy_weights = self.entropy[top_indices] / np.sum(self.entropy[top_indices])
            entropy_weighted = enhanced_features * entropy_weights.reshape(1, -1)
            
            enhanced_features = np.column_stack([enhanced_features, entropy_weighted])
            enhanced_names.extend([f"{name}_ent_weighted" for name in enhanced_names[:len(top_indices)]])
            
        # Add KL divergence features if requested
        if include_kl and hasattr(self, 'kl_divergence'):
            kl_features = self.kl_divergence[:, top_indices]
            
            enhanced_features = np.column_stack([enhanced_features, kl_features])
            enhanced_names.extend([f"{name}_kl" for name in enhanced_names[:len(top_indices)]])
            
        return enhanced_features, enhanced_names
    
    def derive_high_signal_features(self, window_size=100, min_snr=2.0):
        """
        Derive features with high signal-to-noise ratio using information theory measures.
        
        Parameters:
        -----------
        window_size : int
            Window size for computing local statistics
        min_snr : float
            Minimum signal-to-noise ratio threshold
            
        Returns:
        --------
        tuple
            (derived_features, feature_names)
        """
        print("Deriving high signal-to-noise features...")
        
        if self.mi_matrix is None:
            self.compute_mutual_information_matrix()
            
        if not hasattr(self, 'kl_divergence'):
            self.compute_kl_divergence(window_size=window_size)
            
        derived_features = []
        derived_names = []
        
        # Compute signal-to-noise ratio for each feature
        for j in range(self.n_features):
            # Signal strength: mutual information with target
            signal = self.mi_matrix[j, self.target_col_idx]
            
            # Noise estimate: average MI with non-target features
            noise_mi = np.delete(self.mi_matrix[j, :], [j, self.target_col_idx])
            noise = np.mean(noise_mi)
            
            # Compute SNR
            snr = signal / (noise + 1e-10)
            
            if snr >= min_snr:
                # Get the raw feature
                feature = self.scaled_data[:, j]
                
                # Compute local information measures
                local_kl = self.kl_divergence[:, j]
                
                # Create enhanced feature variations
                # 1. KL-weighted feature
                kl_weighted = feature * (1 + np.log1p(local_kl))
                
                # 2. Tail-emphasized feature
                tail_factor, _ = self._adjust_for_heavy_tails(feature.reshape(-1, 1))
                tail_weighted = feature * tail_factor
                
                # 3. Information-ratio feature
                info_ratio = signal / (np.sum(noise_mi) + 1e-10)
                info_weighted = feature * info_ratio
                
                # Add derived features
                derived_features.extend([
                    kl_weighted,
                    tail_weighted,
                    info_weighted
                ])
                
                base_name = self.feature_names[j]
                derived_names.extend([
                    f"{base_name}_kl_weighted",
                    f"{base_name}_tail_weighted",
                    f"{base_name}_info_weighted"
                ])
                
        if not derived_features:
            print("No features met the minimum SNR threshold")
            return np.array([]), []
            
        # Combine and normalize derived features
        derived_features = np.column_stack(derived_features)
        derived_features = StandardScaler().fit_transform(derived_features)
        
        return derived_features, derived_names 