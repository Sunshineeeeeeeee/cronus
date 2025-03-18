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
        
    def identify_regimes(self, df, timestamp_col, price_col, volume_col, volatility_col, 
                        n_regimes=4, window_sizes=None, top_features=10, alpha=0.5, beta=0.1):
        """
        Identify volatility regimes in the given data.
        
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
        """
        print("Beginning volatility regime detection...")
        
        # Initialize analyzer if not already done
        if self.analyzer is None:
            self.analyzer = VolatilityRegimeAnalyzer(
                df=df,
                timestamp_col=timestamp_col,
                price_col=price_col,
                volume_col=volume_col,
                volatility_col=volatility_col
            )
        
        # Compute and enhance features
        self.analyzer.compute_features(window_sizes=window_sizes)
        self.analyzer.enhance_features(n_features=top_features)
        
        # Detect regimes
        self.analyzer.detect_regimes(
            n_regimes=n_regimes,
            alpha=alpha,
            beta=beta
        )
        
        # Get regime results and store them
        self.regimes = self.analyzer.get_regime_labels()
        
        # Calculate and store regime statistics
        df_with_regimes = df.copy()
        df_with_regimes['regime'] = self.regimes
        
        # Calculate regime statistics
        regime_stats = {
            'regime_stats': [],
            'transition_probs': np.zeros((n_regimes, n_regimes))
        }
        
        # Calculate statistics for each regime
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
        for i in range(len(self.regimes) - 1):
            current_regime = self.regimes[i]
            next_regime = self.regimes[i + 1]
            regime_stats['transition_probs'][current_regime, next_regime] += 1
        
        # Normalize transition probabilities
        for i in range(n_regimes):
            row_sum = regime_stats['transition_probs'][i].sum()
            if row_sum > 0:
                regime_stats['transition_probs'][i] /= row_sum
        
        self.regime_stats = regime_stats
        
        return df_with_regimes
    
    def visualize_regimes(self, output_dir='./volatility_regimes'):
        """
        Visualize the identified regimes.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualizations
            
        Returns:
        --------
        None
        """
        if self.analyzer is None or self.regimes is None:
            raise ValueError("You must identify regimes first")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        self.analyzer.visualize_results(output_dir=output_dir)
        
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