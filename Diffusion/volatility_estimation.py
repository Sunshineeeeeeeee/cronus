import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fft, stats, optimize
from scipy.signal import spectrogram

def estimate_tick_volatility(df, symbol_col='SYMBOL', timestamp_col='TIMESTAMP', 
                                    price_col='VALUE', volume_col='VOLUME',
                                    method='all', window_size=1000, step_size=100,
                                    num_freqs=50, smooth_factor=0.5):
    """
    Estimate tick-level volatility using advanced methods including:
    1. Localized Fourier Analysis (Malliavin-Mancino on sliding windows)
    2. Maximum Likelihood Estimation (MLE) with stochastic volatility model
    3. Wavelet-based multi-scale volatility
    
    Parameters:
    -----------
    method : str
        Method to use: 'fourier', 'mle', 'wavelet', or 'all'
    window_size : int
        Size of sliding window (in number of ticks)
    step_size : int
        Step size for sliding window
    num_freqs : int
        Number of Fourier frequencies to use
    smooth_factor : float
        Smoothing factor for volatility estimates (0-1)
        
    Returns:
    --------
    pd.DataFrame
        Original dataframe with added volatility columns
    """
    print(f"Estimating advanced tick-level volatility for {len(df)} ticks...")
    
    result_df = df.copy()
    
    for symbol in df[symbol_col].unique():
        symbol_mask = result_df[symbol_col] == symbol
        symbol_data = result_df[symbol_mask].copy().sort_values(timestamp_col)
        
        symbol_data[timestamp_col] = pd.to_datetime(symbol_data[timestamp_col])
        
        symbol_data['log_price'] = np.log(symbol_data[price_col])
        symbol_data['return'] = symbol_data['log_price'].diff().fillna(0)
        symbol_data['abs_return'] = np.abs(symbol_data['return'])
        
        symbol_data['time_delta'] = symbol_data[timestamp_col].diff().dt.total_seconds().fillna(0.001)
        
        if method in ['fourier', 'all']:
            symbol_data['fourier_vol'] = np.nan
        if method in ['mle', 'all']:
            symbol_data['mle_vol'] = np.nan
        if method in ['wavelet', 'all']:
            symbol_data['wavelet_vol'] = np.nan
        
        n_ticks = len(symbol_data)
        
        # =============== LOCALIZED FOURIER VOLATILITY ESTIMATION  ===============
        if method in ['fourier', 'all']:
            print(f"Computing localized Fourier volatility for {symbol}...")
            
            fourier_volatility = np.zeros(n_ticks)
            
            windows_calculated = 0
            
            for start_idx in range(0, n_ticks, step_size):
                end_idx = min(start_idx + window_size, n_ticks)
                
                if end_idx - start_idx < 50:
                    continue
                
                # Picking window
                window_data = symbol_data.iloc[start_idx:end_idx]
                
                times = window_data['time_delta'].cumsum().values
                log_prices = window_data['log_price'].values
                T = times[-1] - times[0]
                
                if T <= 0:
                    continue  
                
                try:
                    # Compute Fourier coefficients for this window
                    fourier_coeffs = compute_malliavin_mancino_fourier(log_prices, times, T, num_freqs)
                    
                    # Compute integrated variance for this window
                    power_spectrum = fourier_coeffs['power']
                    integrated_variance = np.sum(power_spectrum) / (2 * T)
                    
                    # Convert to per-tick volatility
                    window_volatility = np.sqrt(integrated_variance)
                    
                    # Assign to all ticks in this window
                    window_range = slice(start_idx, end_idx)
                    fourier_volatility[window_range] = np.maximum(
                        fourier_volatility[window_range],
                        window_volatility
                    )
                    
                    windows_calculated += 1
                    
                except Exception as e:
                    print(f"Warning: Error in Fourier calculation for window {start_idx}-{end_idx}: {e}")
            
            print(f"Calculated Fourier volatility using {windows_calculated} windows")
            
            if windows_calculated > 0:
                for i in range(n_ticks):
                    if fourier_volatility[i] == 0:
                        left_idx = max(0, i-1)
                        while left_idx >= 0 and fourier_volatility[left_idx] == 0:
                            left_idx -= 1
                        
                        right_idx = min(n_ticks-1, i+1)
                        while right_idx < n_ticks and fourier_volatility[right_idx] == 0:
                            right_idx += 1
                        
                        if left_idx >= 0 and right_idx < n_ticks:
                            if i - left_idx <= right_idx - i:
                                fourier_volatility[i] = fourier_volatility[left_idx]
                            else:
                                fourier_volatility[i] = fourier_volatility[right_idx]
                        elif left_idx >= 0:
                            fourier_volatility[i] = fourier_volatility[left_idx]
                        elif right_idx < n_ticks:
                            fourier_volatility[i] = fourier_volatility[right_idx]
            
            # Smooth the volatility estimates
            if smooth_factor > 0:
                window_smooth = int(window_size * smooth_factor)
                fourier_volatility = pd.Series(fourier_volatility).rolling(
                    window=window_smooth, min_periods=1, center=True
                ).mean().values
            
            # Save results
            symbol_data['fourier_vol'] = fourier_volatility
        
        # =============== MLE-BASED VOLATILITY ESTIMATION ===============
        if method in ['mle', 'all']:
            print(f"Computing MLE-based volatility for {symbol}...")
            
            mle_volatility = np.zeros(n_ticks)
            
            windows_calculated = 0
            
            for start_idx in range(0, n_ticks, step_size):
                end_idx = min(start_idx + window_size, n_ticks)
                
                if end_idx - start_idx < 50:
                    continue
                
                window_data = symbol_data.iloc[start_idx:end_idx]
                returns = window_data['return'].values
                
                try:
                    # Use MLE to fit a stochastic volatility model
                    # Simplified approach with constant volatility in the window
                    def neg_log_likelihood(params):
                        sigma = params[0]  
                        if sigma <= 0:
                            return 1e10  
                        
                        ll = -np.sum(stats.norm.logpdf(returns, loc=0, scale=sigma))
                        return ll
                    
                    initial_guess = [np.std(returns)]
                    
                    # Optimize to find MLE parameters
                    result = optimize.minimize(
                        neg_log_likelihood, 
                        initial_guess,
                        bounds=[(1e-10, None)]
                    )
                    
                    mle_vol = result.x[0]
                    
                    window_range = slice(start_idx, end_idx)
                    mle_volatility[window_range] = np.maximum(
                        mle_volatility[window_range],
                        mle_vol
                    )
                    
                    windows_calculated += 1
                    
                except Exception as e:
                    print(f"Warning: Error in MLE calculation for window {start_idx}-{end_idx}: {e}")
            
            print(f"Calculated MLE volatility using {windows_calculated} windows")
            
            if windows_calculated > 0:
                for i in range(n_ticks):
                    if mle_volatility[i] == 0:
                        left_idx = max(0, i-1)
                        while left_idx >= 0 and mle_volatility[left_idx] == 0:
                            left_idx -= 1
                        
                        right_idx = min(n_ticks-1, i+1)
                        while right_idx < n_ticks and mle_volatility[right_idx] == 0:
                            right_idx += 1
                        
                        if left_idx >= 0 and right_idx < n_ticks:
                            if i - left_idx <= right_idx - i:
                                mle_volatility[i] = mle_volatility[left_idx]
                            else:
                                mle_volatility[i] = mle_volatility[right_idx]
                        elif left_idx >= 0:
                            mle_volatility[i] = mle_volatility[left_idx]
                        elif right_idx < n_ticks:
                            mle_volatility[i] = mle_volatility[right_idx]
            
            if smooth_factor > 0:
                window_smooth = int(window_size * smooth_factor)
                mle_volatility = pd.Series(mle_volatility).rolling(
                    window=window_smooth, min_periods=1, center=True
                ).mean().values
            
            symbol_data['mle_vol'] = mle_volatility
        
        # =============== WAVELET-BASED MULTI-SCALE VOLATILITY  ===============
        if method in ['wavelet', 'all']:
            print(f"Computing wavelet-based volatility for {symbol}...")
            
            # Use the spectrogram function to perform a windowed FFT (similar to wavelet analysis)
            # This gives us frequency components over time
            returns = symbol_data['return'].values
            
            returns = np.nan_to_num(returns)
            
            try:
                # Calculate spectrogram (windowed FFT)
                f, t, Sxx = spectrogram(
                    returns,
                    fs=1.0,  # Sampling frequency (normalized)
                    nperseg=min(256, n_ticks//10),  # Window size
                    noverlap=min(128, n_ticks//20),  # Overlap
                    scaling='spectrum'  # Return spectrum
                )
                
                # Extract volatility at different frequency bands
                # Low frequencies (long-term volatility)
                low_freq_idx = slice(0, len(f)//10)
                # Medium frequencies
                med_freq_idx = slice(len(f)//10, len(f)//3)
                # High frequencies (short-term volatility)
                high_freq_idx = slice(len(f)//3, None)
                
                # Calculate power in each frequency band over time
                low_power = np.sqrt(np.sum(Sxx[low_freq_idx, :], axis=0))
                med_power = np.sqrt(np.sum(Sxx[med_freq_idx, :], axis=0))
                high_power = np.sqrt(np.sum(Sxx[high_freq_idx, :], axis=0))
                
                # Interpolate to get values for each tick
                wavelet_vol = np.zeros(n_ticks)
                tick_positions = np.linspace(0, 1, n_ticks)
                t_positions = np.linspace(0, 1, len(t))
                
                # Combine frequency bands with more weight to medium frequencies
                combined_power = (low_power * 0.3 + med_power * 0.5 + high_power * 0.2)
                
                # Interpolate to get values for each tick
                wavelet_vol = np.interp(tick_positions, t_positions, combined_power)
                
                # Scale to match typical volatility levels
                scale_factor = np.std(returns) / np.mean(wavelet_vol)
                wavelet_vol *= scale_factor
                
                # Smooth the results
                if smooth_factor > 0:
                    window_smooth = int(window_size * smooth_factor)
                    wavelet_vol = pd.Series(wavelet_vol).rolling(
                        window=window_smooth, min_periods=1, center=True
                    ).mean().values
                
                symbol_data['wavelet_vol'] = wavelet_vol
                
            except Exception as e:
                print(f"Warning: Error in wavelet calculation: {e}")
                # Fallback to EWMA volatility
                symbol_data['wavelet_vol'] = np.sqrt(
                   (symbol_data['return']**2).ewm(span=window_size//5, min_periods=10).mean()
                )
        
        # 4. COMBINED ADAPTIVE VOLATILITY
        if method == 'all':
            # Combine the methods using weights that can adapt based on return characteristics
            valid_methods = []
            
            if 'fourier_vol' in symbol_data.columns and not symbol_data['fourier_vol'].isna().all():
                valid_methods.append('fourier_vol')
            
            if 'mle_vol' in symbol_data.columns and not symbol_data['mle_vol'].isna().all():
                valid_methods.append('mle_vol')
                
            if 'wavelet_vol' in symbol_data.columns and not symbol_data['wavelet_vol'].isna().all():
                valid_methods.append('wavelet_vol')
            
            if len(valid_methods) > 0:
                # Initialize with equal weights
                weights = {method: 1.0 / len(valid_methods) for method in valid_methods}
                
                # Calculate combined volatility
                symbol_data['combined_vol'] = 0
                for method in valid_methods:
                    symbol_data['combined_vol'] += weights[method] * symbol_data[method]
        
        vol_columns = [col for col in symbol_data.columns 
                     if col.endswith('_vol') and col in symbol_data.columns]
        
        for col in vol_columns:
            result_df.loc[symbol_mask, col] = symbol_data[col]
        
        result_df.loc[symbol_mask, 'return'] = symbol_data['return']
    
    print("Completed advanced tick-level volatility estimation")
    return result_df

def compute_malliavin_mancino_fourier(log_prices, times, T, num_freqs):
    """
    Compute Fourier coefficients using Malliavin-Mancino estimator.
    
    Parameters:
    -----------
    log_prices : numpy.ndarray
        Log prices
    times : numpy.ndarray
        Observation times in seconds
    T : float
        Total time period in seconds
    num_freqs : int
        Number of Fourier frequencies to use
        
    Returns:
    --------
    dict
        Fourier coefficients and related data
    """
    # Calculate returns
    returns = np.diff(log_prices)
    return_times = times[1:]
    
    # Generate Fourier frequencies
    freqs = np.arange(1, num_freqs + 1)
    
    # Initialize Fourier coefficient arrays
    fourier_cos = np.zeros(num_freqs)
    fourier_sin = np.zeros(num_freqs)
    
    # Compute Fourier coefficients
    for i, freq in enumerate(freqs):
        # Calculate angular frequency
        omega = 2 * np.pi * freq / T
        
        # Calculate Fourier coefficients for this frequency
        cos_term = np.cos(omega * return_times)
        sin_term = np.sin(omega * return_times)
        
        fourier_cos[i] = np.sum(returns * cos_term)
        fourier_sin[i] = np.sum(returns * sin_term)
    
    # Calculate power spectrum (for volatility)
    power_spectrum = fourier_cos**2 + fourier_sin**2
    
    # Calculate autocorrelation function via inverse Fourier transform
    acf = fft.irfft(power_spectrum, n=num_freqs*2)
    
    return {
        'cos': fourier_cos,
        'sin': fourier_sin,
        'power': power_spectrum,
        'acf': acf,
        'freqs': freqs
    }