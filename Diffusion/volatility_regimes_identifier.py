import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
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
        
    def identify_regimes(self, df, timestamp_col, price_col, volume_col, volatility_col, 
                        n_regimes=4, window_sizes=None, top_features=10, alpha=0.5, beta=0.1,
                        window_size=100, overlap=50, max_path_length=3, min_epsilon=0.1, max_epsilon=2.0,
                        sample_size=10000, sampling_method='sequential', output_dir=None):
        """
        Identify volatility regimes using a two-stage approach with path complex and zigzag persistence:
        1. Learn patterns from a representative sample
        2. Apply patterns to the full dataset
        
        Parameters:
            df (pd.DataFrame): Input data
            timestamp_col (str): Name of timestamp column
            price_col (str): Name of price column
            volume_col (str): Name of volume column
            volatility_col (str): Name of volatility column
            n_regimes (int): Number of regimes to identify
            window_sizes (list): List of window sizes for feature extraction
            top_features (int): Number of top features to use
            alpha (float): Weight for temporal component
            beta (float): Decay rate for temporal distance
            window_size (int): Size of sliding window for zigzag persistence
            overlap (int): Number of points to overlap between windows
            max_path_length (int): Maximum length of paths to consider
            min_epsilon (float): Minimum distance threshold
            max_epsilon (float): Maximum distance threshold
            sample_size (int): Size of the sample to use for pattern learning
            sampling_method (str): Method to use for sampling ('sequential' or 'chunks')
            output_dir (str): Directory for outputs (default: Diffusion/volatility_regimes_identification/results)
        """
        print("Beginning two-stage volatility regime detection...")
        
        # Set up output directory
        if output_dir is None:
            # Create results directory in volatility_regimes_identification
            self.output_dir = os.path.join('/Users/aleksandr/code/scripts/Diffusion/volatility_regimes_identification', 'results')
        else:
            self.output_dir = output_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Store parameters
        self.sample_size = min(sample_size, len(df))
        self.window_sizes = window_sizes if window_sizes is not None else [10, 30, 50]
        
        # Stage 1: Sample and Learn Patterns
        print("\n=== Stage 1: Learning Patterns from Sample ===")
        self.sampled_data = self._sample_data(df, sampling_method)
        
        # Initialize analyzer with sampled data
        self.analyzer = VolatilityRegimeAnalyzer(
            df=self.sampled_data,
            timestamp_col=timestamp_col,
            price_col=price_col,
            volume_col=volume_col,
            volatility_col=volatility_col
        )
        
        # Compute and enhance features on sample
        print("\n--- STEP 1: Computing Features ---")
        self.analyzer.compute_features(window_sizes=self.window_sizes)
        print("\n--- STEP 2: Enhancing Features ---")
        self.analyzer.enhance_features(n_features=top_features)
        
        # Detect regimes on sample using path zigzag persistence
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
        
        # Verify model is trained
        if self.regimes is None:
            raise ValueError("Failed to train regime detection model on sample data")
            
        # Analyze regimes to get statistics
        self.analyzer.regime_analysis = self.analyzer.tda.analyze_regimes()
        
        # Stage 2: Apply Patterns to Full Dataset
        print("\n=== Stage 2: Applying Patterns to Full Dataset ===")
        df_with_regimes = self._apply_patterns_to_full_dataset(
            df, timestamp_col, price_col, volume_col, volatility_col
        )
        
        # Calculate and store regime statistics
        self._calculate_regime_statistics(df_with_regimes, volatility_col, timestamp_col)
        
        # Save results to CSV
        results_file = os.path.join(self.output_dir, 'tick_data_with_regimes.csv')
        df_with_regimes.to_csv(results_file, index=False)
        print(f"Saved data with regime labels to {results_file}")
        
        # Save model
        model_file = os.path.join(self.output_dir, 'regime_model.pkl')
        self.save_model(model_file)
        print(f"\nSaved model to {model_file}")
        
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
        
        # Process the full dataset through the same pipeline
        full_analyzer = VolatilityRegimeAnalyzer(
            df=df,
            timestamp_col=timestamp_col,
            price_col=price_col,
            volume_col=volume_col,
            volatility_col=volatility_col
        )
        
        # Use the same feature computation and enhancement as the sample
        print("Computing microstructure features...")
        full_analyzer.compute_features(window_sizes=self.window_sizes)
        full_analyzer.enhance_features(n_features=len(self.analyzer.enhanced_feature_names))
        
        # Copy trained model components to the new analyzer
        full_analyzer.regime_labels = self.regimes
        
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
        
        return df_with_regimes
    
    def _calculate_regime_statistics(self, df_with_regimes, volatility_col, timestamp_col):
        """
        Calculate statistics for the identified regimes.
        
        Parameters:
            df_with_regimes (pd.DataFrame): Data with regime labels
            volatility_col (str): Name of volatility column
            timestamp_col (str): Name of timestamp column
        """
        n_regimes = len(np.unique(df_with_regimes['regime']))
        
        # Calculate statistics for each regime
        regime_stats = {
            'regime_stats': [],
            'transition_probs': np.zeros((n_regimes, n_regimes))
        }
        
        for regime in range(n_regimes):
            regime_data = df_with_regimes[df_with_regimes['regime'] == regime]
            
            # Basic statistics
            stats = {
                'regime_id': regime,
                'size': len(regime_data),
                'mean_vol': regime_data[volatility_col].mean()
            }
            
            # Calculate duration if timestamps are available
            if pd.api.types.is_datetime64_any_dtype(regime_data[timestamp_col]):
                duration = (regime_data[timestamp_col].max() - 
                          regime_data[timestamp_col].min()).total_seconds()
                stats['duration'] = duration
                
            regime_stats['regime_stats'].append(stats)
        
        # Calculate transition probabilities
        for i in range(len(df_with_regimes['regime']) - 1):
            current_regime = df_with_regimes['regime'].iloc[i]
            next_regime = df_with_regimes['regime'].iloc[i + 1]
            regime_stats['transition_probs'][current_regime, next_regime] += 1
        
        # Normalize transition probabilities
        for i in range(n_regimes):
            row_sum = regime_stats['transition_probs'][i].sum()
            if row_sum > 0:
                regime_stats['transition_probs'][i] /= row_sum
        
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