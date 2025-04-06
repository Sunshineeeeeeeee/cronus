import numpy as np
import pandas as pd
import warnings
import time
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')

class MicrostructureFeatureEngine:
    """
    Extracts robust microstructure features from tick data to capture
    price formation processes, order flow dynamics, and volatility signatures.
    
    Optimized for high-frequency tick data analysis with specific focus on
    detecting volatility regimes using modern market microstructure metrics.
    """
    
    def __init__(self, timestamp_col='Timestamp', price_col='Value', 
                 volume_col='Volume', volatility_col='Volatility'):
        """
        Initialize the feature engine.
        
        Parameters:
        -----------
        timestamp_col : str
            Column name for timestamps
        price_col : str
            Column name for price values
        volume_col : str
            Column name for volume values
        volatility_col : str
            Column name for volatility values
        """
        self.timestamp_col = timestamp_col
        self.price_col = price_col
        self.volume_col = volume_col
        self.volatility_col = volatility_col
        self.features = {}
        
    def extract_features(self, tick_data, window_sizes=[10, 50, 100], normalize=True,
                         include_original=True):
        """
        Extract microstructure features from tick data with robust NaN handling.
        
        Parameters:
        -----------
        tick_data : pandas.DataFrame
            DataFrame with tick data
        window_sizes : list
            List of window sizes for feature calculation
        normalize : bool
            Whether to normalize features
        include_original : bool
            Whether to include original price and volume columns
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with extracted features and no NaN values
        """
        start_time = time.time()
        print(f"Extracting microstructure features with window sizes: {window_sizes}")
        
        # Create a copy of input data to avoid modifying the original
        df = tick_data.copy()
        self.df = df
        
        # Check for NaN values in input data
        print("\n=== Input Data Analysis ===")
        for col in [self.timestamp_col, self.price_col, self.volume_col, self.volatility_col]:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    print(f"WARNING: Found {nan_count} NaN values in input column '{col}'")
                    # Fill NaNs in input data
                    if col == self.timestamp_col:
                        # For timestamps, forward fill
                        df[col] = df[col].ffill()
                    else:
                        # For other columns, use forward fill then backward fill
                        df[col] = df[col].ffill().bfill()
        
        # Ensure timestamp is in datetime format
        if df[self.timestamp_col].dtype != 'datetime64[ns]':
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        
        # Create feature DataFrame
        feature_df = pd.DataFrame(index=df.index)
        
        # Add original columns if requested
        if include_original:
            feature_df[self.price_col] = df[self.price_col]
            feature_df[self.volume_col] = df[self.volume_col]
            if self.volatility_col in df.columns:
                feature_df[self.volatility_col] = df[self.volatility_col]
        
        # ==================== PREPROCESSING ====================
        # Calculate log prices and returns
        df['log_price'] = np.log(df[self.price_col])
        df['log_return'] = df['log_price'].diff().fillna(0)
        df['return_sign'] = np.sign(df['log_return'])
        
        # Calculate time deltas in seconds for temporal analysis
        df['time_delta'] = df[self.timestamp_col].diff().dt.total_seconds()
        # Fill first row with median time delta to avoid NaN
        df['time_delta'].fillna(df['time_delta'].median() if len(df) > 1 else 1.0, inplace=True)
        # Ensure no zero time deltas (replace with small value)
        df['time_delta'] = df['time_delta'].replace(0, 0.001)
        
        # ==================== CORE MICROSTRUCTURE FEATURES ====================
        print("Computing core microstructure features...")
        
        # 1. TICK METRICS - capture order flow patterns
        feature_df['tick_direction'] = df['return_sign']
        # Immediately executable tick direction (next tick moves in same direction)
        feature_df['tick_reversal'] = (df['return_sign'] * df['return_sign'].shift(1)).fillna(0)
        # Consecutive ticks in same direction
        feature_df['tick_run_length'] = (df['return_sign'] == df['return_sign'].shift(1)).astype(int)
        # Volatility signature at tick level
        feature_df['tick_volatility'] = np.abs(df['log_return'])
        
        # 2. BUY/SELL PRESSURE - approximated using tick rule
        df['buy_volume'] = np.where(df['log_return'] >= 0, df[self.volume_col], 0)
        df['sell_volume'] = np.where(df['log_return'] < 0, df[self.volume_col], 0)
        # Immediate order flow imbalance
        feature_df['volume_imbalance'] = (df['buy_volume'] - df['sell_volume']) / (df['buy_volume'] + df['sell_volume'] + 1e-8)
        
        # 3. PRICE IMPACT - how trades move prices
        # Kyle's lambda (simple version) - price impact per unit volume
        feature_df['kyle_lambda'] = np.abs(df['log_return']) / (np.sqrt(df[self.volume_col]) + 1e-8)
        
        # 4. AMIHUD ILLIQUIDITY - price impact relative to volume
        feature_df['amihud_illiquidity'] = np.abs(df['log_return']) / (df[self.volume_col] + 1e-8)
        
        # 5. REALIZED VOLATILITY METRICS
        # Tick-by-tick realized volatility (Rogers-Satchell)
        df['high_low_ratio'] = np.log(df[self.price_col] / df[self.price_col].shift(1).fillna(df[self.price_col].iloc[0]))
        feature_df['rs_volatility'] = df['high_low_ratio']**2
        
        # 6. TIME-WEIGHTED FEATURES
        # Trading intensity (volume per time)
        feature_df['trading_intensity'] = df[self.volume_col] / df['time_delta']
        # Return per time (volatility per time unit)
        feature_df['return_intensity'] = np.abs(df['log_return']) / df['time_delta']
        
        # ==================== WINDOW-BASED METRICS ====================
        print("Computing window-based features...")
        
        for window in window_sizes:
            # Set window-specific min_periods to reduce NaNs while maintaining statistical validity
            min_periods = min(max(2, window // 5), 5)  # Balanced approach
            window_str = f'{window}'
            
            # WINDOW-BASED METRICS - apply over specific lookback periods
            
            # 1. VOLATILITY ESTIMATORS
            # Realized volatility (sum of squared returns)
            rollsum = df['log_return'].pow(2).rolling(window=window, min_periods=min_periods).sum()
            feature_df[f'realized_vol_{window_str}'] = np.sqrt(rollsum)
            # Absolute return as volatility proxy
            feature_df[f'abs_return_{window_str}'] = df['log_return'].abs().rolling(window=window, min_periods=min_periods).mean()
            
            # Modified Garman-Klass volatility estimator for tick data
            # Get high, low, close prices in window
            price_window = df[self.price_col].rolling(window=window, min_periods=min_periods)
            price_max = price_window.max()
            price_min = price_window.min()
            price_open = df[self.price_col].shift(window).fillna(df[self.price_col].iloc[0])
            price_close = df[self.price_col]
            
            # Calculate spread ratio (safer than direct log differences)
            hl_ratio = price_max / price_min
            
            # Handle near-zero or negative spreads (avoid numerical issues)
            min_spread_pct = 0.0001  # Minimum 0.01% spread to avoid numerical issues
            hl_ratio = np.maximum(hl_ratio, 1 + min_spread_pct)
            
            # Calculate safe log values
            log_hl = np.log(hl_ratio)
            
            # Create a safer close-open ratio
            co_ratio = price_close / price_open
            
            # Safety adjustments for close-open
            co_ratio = np.where(co_ratio <= 0, 1.0001, co_ratio)  # Ensure positive
            co_ratio = np.where(np.abs(co_ratio - 1) < min_spread_pct, 
                               1 + np.sign(co_ratio - 1) * min_spread_pct, 
                               co_ratio)  # Ensure minimum spread
            
            log_co = np.log(co_ratio)
            
            # Modified Garman-Klass calculation that ensures non-negative values
            # The original term: 0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2
            # can be negative in some cases, causing NaNs when taking sqrt
            gk_term = 0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2
            
            # Ensure result is non-negative before sqrt (use max with small positive value)
            gk_term = np.maximum(gk_term, 1e-10)
            
            # Compute final Garman-Klass volatility
            feature_df[f'gk_vol_{window_str}'] = np.sqrt(gk_term)
            
            # Additional check for remaining NaNs
            if np.isnan(feature_df[f'gk_vol_{window_str}']).any():
                print(f"Still found NaNs in gk_vol_{window_str}, using fallback")
                # Fallback to simpler volatility estimate
                feature_df[f'gk_vol_{window_str}'] = feature_df[f'gk_vol_{window_str}'].fillna(
                    feature_df[f'realized_vol_{window_str}'])
            
            # Safeguard against any remaining NaNs or infinities
            feature_df[f'gk_vol_{window_str}'] = feature_df[f'gk_vol_{window_str}'].replace([np.inf, -np.inf, np.nan],
                                                 feature_df[f'realized_vol_{window_str}'].mean())
            
            # 2. ORDER FLOW METRICS
            # Rolling order flow imbalance
            buy_sum = df['buy_volume'].rolling(window=window, min_periods=min_periods).sum()
            sell_sum = df['sell_volume'].rolling(window=window, min_periods=min_periods).sum()
            denom = buy_sum + sell_sum
            feature_df[f'order_flow_imbalance_{window_str}'] = ((buy_sum - sell_sum) / denom.replace(0, np.nan)).fillna(0)
            
            # 3. PRICE EFFICIENCY MEASURES
            # Variance ratio (test of random walk hypothesis)
            # Compare variance of k-period returns with k times variance of 1-period returns
            if window >= 4:  # Need reasonable window size
                short_var = df['log_return'].rolling(window=window//2, min_periods=min_periods//2).var()
                long_var = df['log_return'].rolling(window=window, min_periods=min_periods).var()
                # Adjust for time scale
                feature_df[f'variance_ratio_{window_str}'] = (short_var / (long_var/2 + 1e-8)).fillna(1.0)
            
            # 4. MARKET IMPACT
            # Rolling Kyle's lambda
            feature_df[f'kyle_lambda_{window_str}'] = feature_df['kyle_lambda'].rolling(window=window, min_periods=min_periods).mean()
            
            # 5. AMIHUD MEASURE (price impact per volume)
            feature_df[f'amihud_{window_str}'] = feature_df['amihud_illiquidity'].rolling(window=window, min_periods=min_periods).mean()
            
            # 6. TICK DYNAMICS
            # Directional tick patterns
            feature_df[f'tick_pattern_{window_str}'] = df['return_sign'].rolling(window=window, min_periods=min_periods).mean()
            
            # 7. VOLUME PROFILE
            # Relative trading volume
            feature_df[f'rel_volume_{window_str}'] = df[self.volume_col].rolling(window=window, min_periods=min_periods).mean() / df[self.volume_col].mean()
        
        # ==================== TEMPORAL CYCLICAL FEATURES ====================
        # Get temporal indicators for intraday patterns (useful for capturing regime changes)
        df['hour'] = df[self.timestamp_col].dt.hour
        df['minute'] = df[self.timestamp_col].dt.minute
        
        # Time of day as cyclical features (sine/cosine encoding)
        minutes_in_day = 24 * 60
        df['time_of_day'] = (df['hour'] * 60 + df['minute']) / minutes_in_day
        feature_df['sin_time'] = np.sin(2 * np.pi * df['time_of_day'])
        feature_df['cos_time'] = np.cos(2 * np.pi * df['time_of_day'])
        
        # ==================== VOLATILITY REGIME SPECIFIC FEATURES ====================
        if self.volatility_col in df.columns:
            # Use volatility as direct input if available
            feature_df['volatility'] = df[self.volatility_col]
            
            # Calculate volatility-adjusted features
            feature_df['vol_adjusted_return'] = df['log_return'] / (df[self.volatility_col] + 1e-8)
            feature_df['vol_impact_ratio'] = feature_df['kyle_lambda'] / (df[self.volatility_col] + 1e-8)
            
            # Volatility trend indicators
            for window in window_sizes:
                min_periods = max(2, window // 5)
                vol_ma = df[self.volatility_col].rolling(window=window, min_periods=min_periods).mean()
                
                # Ensure vol_ma is positive (avoid division by zero or negative values)
                vol_ma_safe = np.maximum(vol_ma, 1e-10)
                
                # Calculate volatility trend ratio (current vs moving average)
                # Cap extreme values to prevent outliers
                vol_ratio = df[self.volatility_col] / vol_ma_safe
                
                # Cap extreme ratios to reasonable range
                vol_ratio = np.minimum(np.maximum(vol_ratio, 0.1), 10.0)
                
                # Calculate trend as percentage change from moving average
                feature_df[f'vol_trend_{window}'] = vol_ratio - 1
                
                # Additional safety check for any remaining NaNs
                if np.isnan(feature_df[f'vol_trend_{window}']).any():
                    print(f"Found NaNs in vol_trend_{window}, using fallback")
                    # Replace NaNs with zeros (no trend)
                    feature_df[f'vol_trend_{window}'] = feature_df[f'vol_trend_{window}'].fillna(0)
        
        # ==================== CLEAN AND PROCESS FEATURES ====================
        # Check for NaN values before filling
        nan_counts = feature_df.isna().sum()
        total_nan = nan_counts.sum()
        
        # Keep track of problematic columns
        problematic_columns = []
        
        if total_nan > 0:
            print(f"\nFound {total_nan} NaN values in features before filling")
            print("Top NaN columns:")
            
            # Identify columns with NaN values
            columns_with_nans = {}
            for col, count in nan_counts.items():
                if count > 0:
                    nan_percent = (count / len(feature_df)) * 100
                    columns_with_nans[col] = (count, nan_percent)
                    print(f"  {col}: {count} NaNs ({nan_percent:.2f}%)")
            
            # Decide which columns to drop (those with more than 5% NaNs)
            columns_to_drop = []
            for col, (count, percent) in columns_with_nans.items():
                if percent > 5.0:
                    columns_to_drop.append(col)
                    problematic_columns.append(col)
            
            if columns_to_drop:
                print("\nDropping columns with >5% NaN values:")
                for col in columns_to_drop:
                    print(f"  - {col} ({columns_with_nans[col][1]:.2f}% NaNs)")
                feature_df = feature_df.drop(columns=columns_to_drop)
            
            # Handle remaining NaNs
            print("\nApplying robust NaN handling for remaining columns...")
            
            # Replace infinity values with NaN
            feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Forward fill (most appropriate for time series data)
            feature_df = feature_df.ffill()
            
            # Backward fill for any remaining NaNs (especially at beginning)
            feature_df = feature_df.bfill()
            
            # Fill any still-remaining NaNs with column median or zero
            for col in feature_df.columns:
                if feature_df[col].isna().any():
                    median = feature_df[col].median()
                    if pd.isna(median):  # If median is NaN, use 0
                        feature_df[col].fillna(0, inplace=True)
                    else:
                        feature_df[col].fillna(median, inplace=True)
        
        # Final check for any remaining NaNs
        if feature_df.isna().any().any():
            print("WARNING: Still have NaN values after filling! Filling with zeros.")
            feature_df.fillna(0, inplace=True)
        else:
            print("All NaN values successfully handled.")
        
        # Print remaining columns after NaN handling
        remaining_columns = feature_df.columns.tolist()
        print(f"\nRemaining {len(remaining_columns)} features after NaN handling:")
        for col in remaining_columns:
            print(f"  - {col}")
            
        # If any columns were problematic but not dropped, add a note
        low_nan_cols = [col for col in problematic_columns if col in remaining_columns]
        if low_nan_cols:
            print("\nThe following columns had NaNs but were kept (NaNs filled):")
            for col in low_nan_cols:
                print(f"  - {col}")
        
        # Normalize features if requested
        if normalize:
            print("\nNormalizing features...")
            scaler = StandardScaler()
            
            # Identify columns to normalize (exclude original price, volume, volatility)
            cols_to_normalize = [col for col in feature_df.columns if col not in 
                               [self.price_col, self.volume_col, self.volatility_col]]
            
            if cols_to_normalize:
                # Apply normalization
                feature_df[cols_to_normalize] = scaler.fit_transform(feature_df[cols_to_normalize])
        
        # Store feature names (useful for inspection and debugging)
        self.feature_names = feature_df.columns.tolist()
        
        # Store features dictionary for compatibility with existing code
        for col in feature_df.columns:
            self.features[col] = feature_df[col].values
        
        # Report feature extraction success
        n_features = len(feature_df.columns)
        print(f"Successfully extracted {n_features} microstructure features in {time.time() - start_time:.2f} seconds")
        print(f"Feature set optimized for volatility regime detection with zero NaNs")
        
        return feature_df
        
    def get_feature_importance_for_volatility(self, feature_df):
        """
        Estimate feature importance for volatility prediction.
        Simple correlation-based importance calculation.
        
        Parameters:
        -----------
        feature_df : pandas.DataFrame
            DataFrame with features and volatility column
            
        Returns:
        --------
        pandas.Series
            Feature importance scores
        """
        if self.volatility_col not in feature_df.columns:
            print("Volatility column not found in feature DataFrame")
            return None
        
        # Calculate absolute correlation with volatility
        corr = feature_df.corr()[self.volatility_col].abs()
        
        # Sort by importance
        importance = corr.sort_values(ascending=False)
        
        return importance 