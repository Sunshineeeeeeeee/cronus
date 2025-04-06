#!/usr/bin/env python
"""
Debug script for the frequency-entropy 2D filter implementation.
"""

import pandas as pd
import numpy as np
import logging
import os
import sys

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

# Make sure our package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import after path setup
from Diffusion.Volatility_regimes.tda_mapper_pipeline_unified import TDAMapperPipeline

def create_test_data():
    """
    Create synthetic test data with different volatility and frequency regimes.
    """
    np.random.seed(42)
    
    # Create a time series with different volatility regimes
    n_samples = 500
    t = np.linspace(0, 10, n_samples)
    
    # Different frequency components
    base_freq = np.sin(t)                   # Low frequency
    high_freq = np.sin(5 * t)               # High frequency
    mixed_freq = np.sin(t) + 0.5 * np.sin(5 * t)  # Mixed frequency
    
    # Different volatility regimes
    low_vol = np.random.normal(0, 0.1, n_samples)
    high_vol = np.random.normal(0, 0.5, n_samples)
    
    # Create combined signal
    values = np.zeros(n_samples)
    volatility = np.ones(n_samples)
    
    # Segment 1: Low frequency, low volatility (0-100)
    values[:100] = base_freq[:100] + low_vol[:100]
    volatility[:100] = 0.1
    
    # Segment 2: Low frequency, high volatility (100-200)
    values[100:200] = base_freq[100:200] + high_vol[100:200]
    volatility[100:200] = 0.5
    
    # Segment 3: High frequency, low volatility (200-300)
    values[200:300] = high_freq[200:300] + low_vol[200:300]
    volatility[200:300] = 0.1
    
    # Segment 4: High frequency, high volatility (300-400)
    values[300:400] = high_freq[300:400] + high_vol[300:400]
    volatility[300:400] = 0.5
    
    # Segment 5: Mixed frequency, medium volatility (400-500)
    values[400:] = mixed_freq[400:] + np.random.normal(0, 0.3, n_samples)[400:]
    volatility[400:] = 0.3
    
    # Create DataFrame
    df = pd.DataFrame({
        'Timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='5min'),
        'Value': values + 100,  # Add offset to ensure positive values
        'Volume': np.random.exponential(scale=100, size=n_samples),
        'Volatility': volatility
    })
    
    return df

def main():
    """
    Run debug test for the frequency-entropy filter.
    """
    print("\n=== Running Frequency-Entropy Filter Debug Test ===")
    
    # Generate synthetic data
    data = create_test_data()
    print(f"Created test data with shape: {data.shape}")
    print(f"First few rows:\n{data.head()}")
    
    # Initialize TDA Mapper pipeline
    pipeline = TDAMapperPipeline(
        window_size=20,
        overlap=10,
        max_homology_dimension=2,
        verbose=True
    )
    
    # Run pipeline with frequency-entropy filter
    print("\nRunning pipeline with frequency-entropy filter...")
    results = pipeline.run_pipeline(
        data=data,
        feature_columns=['Value'],
        volatility_column='Volatility',
        filter_type='frequency_entropy',  # Specify frequency-entropy filter
        n_intervals=10,
        overlap_perc=0.3,
        return_visualizations=True
    )
    
    # Print mapper complex summary
    print("\n=== Mapper Complex Summary ===")
    print(f"Nodes: {len(results['mapper_graph']['nodes'])}")
    print(f"Edges: {len(results['mapper_graph']['links'])//2}")
    
    # Print betti numbers
    print("\n=== Betti Numbers ===")
    print(results['betti_numbers'])
    
    # Save visualization
    if 'mapper_visualization' in results:
        with open('debug_freq_entropy_visualization.html', 'w') as f:
            f.write(results['mapper_visualization'])
        print(f"\nSaved visualization to 'debug_freq_entropy_visualization.html'")
    
    print("\nDebug test completed.")

if __name__ == "__main__":
    main() 