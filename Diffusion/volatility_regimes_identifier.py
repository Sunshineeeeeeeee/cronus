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
    A simplified interface for identifying volatility regimes in tick data
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
                        sample_size=10000, sampling_method='sequential', output_dir=None):
        """
        Identify volatility regimes using a two-stage approach:
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
        
        # Detect regimes on sample
        print("\n--- STEP 3: Detecting Regimes ---")
        self.regimes = self.analyzer.detect_regimes(
            n_regimes=n_regimes,
            alpha=alpha,
            beta=beta,
            create_mapper=True,  # Ensure mapper graph is created
            compute_homology=True,  # Ensure persistent homology is computed
            output_dir=self.output_dir  # Pass output directory
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
        full_analyzer.tda = self.analyzer.tda
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


# Example usage
if __name__ == "__main__":
    # Load your tick data
    # df = pd.read_csv("your_tick_data.csv")
    
    # Create identifier
    # identifier = VolatilityRegimesIdentifier()
    
    # Identify regimes
    # df_with_regimes = identifier.identify_regimes(df)
    
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