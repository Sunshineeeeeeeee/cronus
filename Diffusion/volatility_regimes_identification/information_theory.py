import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import warnings

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
        
    def estimate_shannon_entropy(self, bandwidth_method='silverman'):
        """
        Estimate Shannon entropy for each feature dimension using kernel density estimation.
        
        H(X_j) = -∑_i p(x_{j,i}) log p(x_{j,i})
        
        Parameters:
        -----------
        bandwidth_method : str
            Method to determine kernel bandwidth: 'silverman' or 'scott'
            
        Returns:
        --------
        self
        """
        print("Estimating Shannon entropy for each feature...")
        
        # Initialize entropy array
        self.entropy = np.zeros(self.n_features)
        
        # Estimate entropy for each feature
        for j in range(self.n_features):
            x = self.scaled_data[:, j].reshape(-1, 1)
            
            # Determine bandwidth using rule of thumb
            if bandwidth_method == 'silverman':
                # Silverman's rule: h = 1.06 * σ * n^(-1/5)
                bw = 1.06 * np.std(x) * self.n_samples**(-1/5)
            else:
                # Scott's rule: h = 3.49 * σ * n^(-1/3)
                bw = 3.49 * np.std(x) * self.n_samples**(-1/3)
                
            # Fit kernel density estimator
            kde = KernelDensity(bandwidth=bw, kernel='gaussian')
            kde.fit(x)
            
            # Evaluate density at sample points
            log_dens = kde.score_samples(x)
            dens = np.exp(log_dens)
            
            # Normalize densities to ensure they sum to 1
            dens = dens / np.sum(dens)
            
            # Calculate entropy (avoiding log(0))
            entropy_j = -np.sum(dens * np.log(dens + 1e-10))
            self.entropy[j] = entropy_j
            
        return self
        
    def compute_kl_divergence(self, window_size=100):
        """
        Compute Kullback-Leibler divergence between local and global distributions.
        
        D_KL(P^t_local || P_global) = ∑_i P^t_local(i) log[P^t_local(i)/P_global(i)]
        
        Parameters:
        -----------
        window_size : int
            Size of local window for computing local distributions
            
        Returns:
        --------
        numpy.ndarray
            Array of shape (n_samples, n_features) containing KL divergence values
        """
        print("Computing KL divergence between local and global distributions...")
        
        kl_divergence = np.zeros((self.n_samples, self.n_features))
        
        for j in range(self.n_features):
            x = self.scaled_data[:, j]
            
            # Compute global distribution (using histogram)
            global_hist, bin_edges = np.histogram(x, bins=20, density=True)
            global_hist = global_hist + 1e-10  # Avoid division by zero
            
            # Compute local distributions for each window
            for i in range(self.n_samples):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(self.n_samples, i + window_size // 2)
                
                local_x = x[start_idx:end_idx]
                
                if len(local_x) < 5:
                    kl_divergence[i, j] = 0
                    continue
                
                # Compute local histogram with same bins as global
                local_hist, _ = np.histogram(local_x, bins=bin_edges, density=True)
                local_hist = local_hist + 1e-10  # Avoid division by zero
                
                # Calculate KL divergence
                kl_div = np.sum(local_hist * np.log(local_hist / global_hist))
                kl_divergence[i, j] = kl_div
                
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
        
    def compute_mutual_information_matrix(self):
        """
        Compute mutual information between all pairs of features using a simplified approach.
        
        Returns:
        --------
        numpy.ndarray
            Array of shape (n_features, n_features) containing mutual information values
        """
        print("Computing mutual information matrix...")
        
        # Initialize mutual information matrix
        mi_matrix = np.zeros((self.n_features, self.n_features))
        
        # For the diagonal, use the entropy
        for i in range(self.n_features):
            mi_matrix[i, i] = self.entropy[i]
        
        # Use correlation coefficient as an approximation for MI for all non-diagonal elements
        # This avoids numerical issues with binning and is faster
        if self.n_samples > 1:
            corr_matrix = np.corrcoef(self.scaled_data.T)
            
            # Convert correlation to mutual information approximation
            # Using the formula for Gaussian variables: MI ≈ -0.5 * log(1 - ρ²)
            for i in range(self.n_features):
                for j in range(i+1, self.n_features):
                    # Use correlation-based approximation
                    rho = corr_matrix[i, j]
                    # Avoid numerical issues with perfect correlation
                    rho_squared = min(rho**2, 0.999)
                    mi = -0.5 * np.log(1 - rho_squared)
                    
                    # Make symmetric
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi
        
        # Store as class attribute and return
        self.mi_matrix = mi_matrix
        return mi_matrix
        
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