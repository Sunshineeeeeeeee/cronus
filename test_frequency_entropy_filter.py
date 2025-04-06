#!/usr/bin/env python
"""
Test script for the new frequency-entropy 2D filter in the TDA Mapper pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from Diffusion.Volatility_regimes.tda_mapper_pipeline_unified import TDAMapperPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

def create_test_data(n_samples=1000):
    """
    Create synthetic test data with different volatility and frequency regimes.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create time series with different regimes
    t = np.linspace(0, 10, n_samples)
    
    # Create different frequency and volatility regimes
    values = np.zeros_like(t)
    volatility = np.ones_like(t)
    
    # Low frequency, low volatility regime (0-200)
    values[:200] = 0.5 * np.sin(1 * t[:200])
    volatility[:200] = 0.2
    
    # Low frequency, high volatility regime (200-400)
    values[200:400] = 0.5 * np.sin(1 * t[200:400])
    volatility[200:400] = 1.0
    
    # High frequency, low volatility regime (400-600)
    values[400:600] = 0.5 * np.sin(5 * t[400:600])
    volatility[400:600] = 0.2
    
    # High frequency, high volatility regime (600-800)
    values[600:800] = 0.5 * np.sin(5 * t[600:800])
    volatility[600:800] = 1.0
    
    # Mixed frequencies (chaotic), medium volatility (800-1000)
    values[800:] = 0.3 * np.sin(2 * t[800:]) + 0.2 * np.sin(5 * t[800:])
    volatility[800:] = 0.6
    
    # Add noise scaled by volatility
    values = values + np.random.normal(0, 1, n_samples) * volatility * 0.1
    
    # Create DataFrame
    df = pd.DataFrame({
        'Timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='5min'),
        'Value': values + 100,  # Add offset to ensure positive values
        'Volume': np.random.exponential(scale=100, size=n_samples),
        'Volatility': volatility
    })
    
    return df

def test_frequency_entropy_filter():
    """
    Test the frequency-entropy filter functionality.
    """
    print("=== Testing Frequency-Entropy Filter ===")
    
    # Generate test data
    print("\nGenerating synthetic test data...")
    df = create_test_data(n_samples=1000)
    print(f"Created test data with shape: {df.shape}")
    
    # Initialize TDA Mapper pipeline with both filter types for comparison
    print("\nInitializing TDA Mapper pipeline...")
    pipeline_volatility = TDAMapperPipeline(
        window_size=50,
        overlap=25,
        max_homology_dimension=2,
        verbose=True
    )
    
    pipeline_freq_entropy = TDAMapperPipeline(
        window_size=50,
        overlap=25,
        max_homology_dimension=2,
        verbose=True
    )
    
    # Run pipeline with volatility filter
    print("\nRunning pipeline with volatility filter...")
    start_time = time.time()
    results_volatility = pipeline_volatility.run_pipeline(
        data=df,
        feature_columns=['Value'],
        volatility_column='Volatility',
        filter_type='volatility',
        n_intervals=10,
        overlap_perc=0.3,
        return_visualizations=True
    )
    elapsed_volatility = time.time() - start_time
    print(f"Volatility filter pipeline completed in {elapsed_volatility:.2f} seconds")
    
    # Run pipeline with frequency-entropy filter
    print("\nRunning pipeline with frequency-entropy filter...")
    start_time = time.time()
    results_freq_entropy = pipeline_freq_entropy.run_pipeline(
        data=df,
        feature_columns=['Value'],
        volatility_column='Volatility',
        filter_type='frequency_entropy',
        n_intervals=10,
        overlap_perc=0.3,
        return_visualizations=True
    )
    elapsed_freq_entropy = time.time() - start_time
    print(f"Frequency-entropy filter pipeline completed in {elapsed_freq_entropy:.2f} seconds")
    
    # Compare mapper complexes
    print("\n=== Comparing Results ===")
    print(f"Volatility Filter - Nodes: {len(results_volatility['mapper_graph']['nodes'])}, " +
          f"Edges: {len(results_volatility['mapper_graph']['links'])//2}")
    print(f"Frequency-Entropy Filter - Nodes: {len(results_freq_entropy['mapper_graph']['nodes'])}, " +
          f"Edges: {len(results_freq_entropy['mapper_graph']['links'])//2}")
    
    # Compare Betti numbers
    print("\nBetti Numbers (Volatility):", results_volatility['betti_numbers'])
    print("Betti Numbers (Freq-Entropy):", results_freq_entropy['betti_numbers'])
    
    # Save visualization HTML files
    if 'mapper_visualization' in results_volatility:
        with open('mapper_visualization_volatility.html', 'w') as f:
            f.write(results_volatility['mapper_visualization'])
        print("\nSaved volatility mapper visualization to 'mapper_visualization_volatility.html'")
    
    if 'mapper_visualization' in results_freq_entropy:
        with open('mapper_visualization_freq_entropy.html', 'w') as f:
            f.write(results_freq_entropy['mapper_visualization'])
        print("Saved frequency-entropy mapper visualization to 'mapper_visualization_freq_entropy.html'")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_frequency_entropy_filter() 