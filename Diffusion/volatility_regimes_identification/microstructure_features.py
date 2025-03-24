import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class MicrostructureFeatureEngine:
    """
    Extracts comprehensive microstructure features from tick data to capture
    order flow dynamics, momentum, and volatility characteristics.
    
    Implements Section 1 of the TDA Pipeline:
    - Microstructure Features (tick imbalance, volume profiles, etc.)
    - Order Flow Metrics (imbalance, impact)
    - Momentum Features (short-term momentum, RSI, etc.)
    """
    
    def __init__(self, tick_data, timestamp_col='Timestamp', price_col='Value', 
                 volume_col='Volume', volatility_col='Volatility'):
        """
        Initialize the feature engine with tick data.
        
        Parameters:
        -----------
        tick_data : pandas.DataFrame
            DataFrame with tick data
        timestamp_col : str
            Column name for timestamps
        price_col : str
            Column name for price values
        volume_col : str
            Column name for volume values
        volatility_col : str
            Column name for volatility values
        """
        self.df = tick_data.copy()
        self.timestamp_col = timestamp_col
        self.price_col = price_col
        self.volume_col = volume_col
        self.volatility_col = volatility_col
        self.window_sizes = None  
        
        # Ensure timestamp is in datetime format
        if self.df[timestamp_col].dtype != 'datetime64[ns]':
            self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
            
        # Calculate log returns
        self.df['log_price'] = np.log(self.df[price_col])
        self.df['return'] = self.df['log_price'].diff().fillna(0)
        self.df['return_sign'] = np.sign(self.df['return'])
        
        # Calculate time deltas in seconds
        self.df['time_delta'] = self.df[timestamp_col].diff().dt.total_seconds().fillna(0.001)
        
        # Initialize feature dictionary to store all computed features
        self.features = {}
        
    def _compute_time_features(self, window_size=100):
        """
        Compute time-based features such as intraday seasonality indicators.
        """
        # Extract hour of day
        self.df['hour'] = self.df[self.timestamp_col].dt.hour
        self.df['minute'] = self.df[self.timestamp_col].dt.minute
        self.df['second'] = self.df[self.timestamp_col].dt.second
        
        # Calculate seconds from market open
        self.df['day'] = self.df[self.timestamp_col].dt.date
        
        market_opens = self.df.groupby('day')[self.timestamp_col].min().reset_index()
        market_opens.rename(columns={self.timestamp_col: 'market_open'}, inplace=True)
        
        self.df = pd.merge(self.df, market_opens, on='day')
        self.df['seconds_from_open'] = (self.df[self.timestamp_col] - self.df['market_open']).dt.total_seconds()
        
        # Calculate time of day as fraction (0-1)
        seconds_in_day = 24 * 60 * 60
        self.df['time_of_day'] = (self.df['hour'] * 3600 + self.df['minute'] * 60 + self.df['second']) / seconds_in_day
        
        # Calculate sinusoidal time features to capture periodicity
        self.df['sin_time'] = np.sin(2 * np.pi * self.df['time_of_day'])
        self.df['cos_time'] = np.cos(2 * np.pi * self.df['time_of_day'])
        
        # Store features
        self.features.update({
            'seconds_from_open': self.df['seconds_from_open'].values,
            'sin_time': self.df['sin_time'].values,
            'cos_time': self.df['cos_time'].values
        })
        
        return self
        
    def compute_microstructure_features(self, window_sizes=[10, 50, 100]):
        """
        Compute microstructure features including:
        - Tick imbalance (TI_Δt)
        - Trade volume profile (TVP_Δt)
        - Trade frequency (TF_Δt)
        - VWAP deviation
        
        Parameters:
        -----------
        window_sizes : list
            List of window sizes for feature calculation
        """
        print("Computing microstructure features...")
        
        # Calculate daily mean volume for volume profile normalization
        self.df['date'] = self.df[self.timestamp_col].dt.date
        daily_mean_volume = self.df.groupby('date')[self.volume_col].transform('mean')
        
        for window in window_sizes:
            window_str = f'{window}'
            
            # 1. Tick imbalance: Sum of return signs in window divided by window size
            self.df[f'tick_imbalance_{window_str}'] = (
                self.df['return_sign'].rolling(window=window, min_periods=1).mean()
            )
            
            # 2. Trade volume profile: Window volume / daily mean volume
            self.df[f'trade_volume_profile_{window_str}'] = (
                self.df[self.volume_col].rolling(window=window, min_periods=1).sum() / 
                (daily_mean_volume * window)
            )
            
            # 3. Trade frequency: Number of trades divided by time span
            # For tick data, we can use the inverse of average time delta
            self.df[f'time_sum_{window_str}'] = self.df['time_delta'].rolling(window=window, min_periods=1).sum()
            self.df[f'trade_frequency_{window_str}'] = window / self.df[f'time_sum_{window_str}'].replace(0, np.nan)
            
            # 4. VWAP calculation
            self.df[f'vwap_{window_str}'] = (
                (self.df[self.price_col] * self.df[self.volume_col]).rolling(window=window, min_periods=1).sum() /
                self.df[self.volume_col].rolling(window=window, min_periods=1).sum()
            )
            
            # VWAP deviation
            self.df[f'vwap_deviation_{window_str}'] = (
                (self.df[self.price_col] - self.df[f'vwap_{window_str}']) / 
                self.df[f'vwap_{window_str}']
            )
            
            # Store computed features
            self.features.update({
                f'tick_imbalance_{window_str}': self.df[f'tick_imbalance_{window_str}'].values,
                f'trade_volume_profile_{window_str}': self.df[f'trade_volume_profile_{window_str}'].values,
                f'trade_frequency_{window_str}': self.df[f'trade_frequency_{window_str}'].values,
                f'vwap_deviation_{window_str}': self.df[f'vwap_deviation_{window_str}'].values
            })
            
        # Clean up temporary columns
        self.df = self.df.drop(columns=[col for col in self.df.columns if col.startswith('time_sum_')])
            
        return self
            
    def compute_order_flow_metrics(self, window_sizes=[10, 50, 100]):
        """
        Compute order flow metrics including:
        - Order flow imbalance (OFI_Δt)
        - Price impact coefficient (λ_Δt)
        
        Parameters:
        -----------
        window_sizes : list
            List of window sizes for feature calculation
        """
        print("Computing order flow metrics...")
        
        # Approximate buy/sell volume using tick rule
        # If price goes up, consider it a buy; if price goes down, consider it a sell
        self.df['buy_volume'] = np.where(self.df['return'] >= 0, self.df[self.volume_col], 0)
        self.df['sell_volume'] = np.where(self.df['return'] < 0, self.df[self.volume_col], 0)
        
        for window in window_sizes:
            window_str = f'{window}'
            
            # 1. Order Flow Imbalance (OFI)
            self.df[f'buy_volume_sum_{window_str}'] = self.df['buy_volume'].rolling(window=window, min_periods=1).sum()
            self.df[f'sell_volume_sum_{window_str}'] = self.df['sell_volume'].rolling(window=window, min_periods=1).sum()
            
            # Calculate order flow imbalance
            numerator = self.df[f'buy_volume_sum_{window_str}'] - self.df[f'sell_volume_sum_{window_str}']
            denominator = self.df[f'buy_volume_sum_{window_str}'] + self.df[f'sell_volume_sum_{window_str}']
            
            self.df[f'order_flow_imbalance_{window_str}'] = numerator / denominator.replace(0, np.nan)
            
            # 2. Price Impact Coefficient
            # Calculate absolute price change over rolling window
            self.df[f'abs_price_change_{window_str}'] = self.df[self.price_col].diff(window).abs()
            
            # Calculate square root of volume over window
            self.df[f'sqrt_volume_{window_str}'] = np.sqrt(
                self.df[self.volume_col].rolling(window=window, min_periods=1).sum()
            )
            
            # Price impact coefficient
            self.df[f'price_impact_{window_str}'] = (
                self.df[f'abs_price_change_{window_str}'] / 
                self.df[f'sqrt_volume_{window_str}'].replace(0, np.nan)
            )
            
            # Store computed features
            self.features.update({
                f'order_flow_imbalance_{window_str}': self.df[f'order_flow_imbalance_{window_str}'].values,
                f'price_impact_{window_str}': self.df[f'price_impact_{window_str}'].values
            })
            
        # Clean up temporary columns
        cols_to_drop = [col for col in self.df.columns if any(x in col for x in 
                       ['buy_volume_sum_', 'sell_volume_sum_', 'abs_price_change_', 'sqrt_volume_'])]
        self.df = self.df.drop(columns=cols_to_drop)
            
        return self
        
    def compute_dmi_features(self, window_sizes=[10, 50, 100]):
        """
        Compute Directional Movement Index (DMI) and related features including:
        - True Range (TR)
        - Plus/Minus Directional Movement (+/-DM)
        - Plus/Minus Directional Indicators (+/-DI)
        - Average Directional Index (ADX)
        - Directional Movement Ratio (DMR)
        - Volatility-Adjusted DMI
        
        Parameters:
        -----------
        window_sizes : list
            List of window sizes for feature calculation
        """
        print("Computing DMI features...")
        
        # Calculate high and low prices for each window
        for window in window_sizes:
            window_str = f'{window}'
            
            # Calculate True Range (TR)
            self.df[f'high_{window_str}'] = self.df[self.price_col].rolling(window=window, min_periods=1).max()
            self.df[f'low_{window_str}'] = self.df[self.price_col].rolling(window=window, min_periods=1).min()
            self.df[f'prev_close_{window_str}'] = self.df[self.price_col].shift(1)
            
            # True Range components
            self.df[f'tr1_{window_str}'] = self.df[f'high_{window_str}'] - self.df[f'low_{window_str}']
            self.df[f'tr2_{window_str}'] = abs(self.df[f'high_{window_str}'] - self.df[f'prev_close_{window_str}'])
            self.df[f'tr3_{window_str}'] = abs(self.df[f'low_{window_str}'] - self.df[f'prev_close_{window_str}'])
            
            # True Range is the maximum of the three components
            self.df[f'tr_{window_str}'] = self.df[[f'tr1_{window_str}', f'tr2_{window_str}', f'tr3_{window_str}']].max(axis=1)
            
            # Calculate Plus/Minus Directional Movement
            self.df[f'up_move_{window_str}'] = self.df[self.price_col] - self.df[self.price_col].shift(1)
            self.df[f'down_move_{window_str}'] = self.df[self.price_col].shift(1) - self.df[self.price_col]
            
            # Plus Directional Movement (+DM)
            self.df[f'plus_dm_{window_str}'] = np.where(
                (self.df[f'up_move_{window_str}'] > self.df[f'down_move_{window_str}']) & 
                (self.df[f'up_move_{window_str}'] > 0),
                self.df[f'up_move_{window_str}'],
                0
            )
            
            # Minus Directional Movement (-DM)
            self.df[f'minus_dm_{window_str}'] = np.where(
                (self.df[f'down_move_{window_str}'] > self.df[f'up_move_{window_str}']) & 
                (self.df[f'down_move_{window_str}'] > 0),
                self.df[f'down_move_{window_str}'],
                0
            )
            
            # Smoothed TR and DM
            self.df[f'smoothed_tr_{window_str}'] = self.df[f'tr_{window_str}'].rolling(window=window, min_periods=1).mean()
            self.df[f'smoothed_plus_dm_{window_str}'] = self.df[f'plus_dm_{window_str}'].rolling(window=window, min_periods=1).mean()
            self.df[f'smoothed_minus_dm_{window_str}'] = self.df[f'minus_dm_{window_str}'].rolling(window=window, min_periods=1).mean()
            
            # Plus/Minus Directional Indicators (+/-DI)
            self.df[f'plus_di_{window_str}'] = 100 * (self.df[f'smoothed_plus_dm_{window_str}'] / self.df[f'smoothed_tr_{window_str}'])
            self.df[f'minus_di_{window_str}'] = 100 * (self.df[f'smoothed_minus_dm_{window_str}'] / self.df[f'smoothed_tr_{window_str}'])
            
            # Average Directional Index (ADX)
            self.df[f'dx_{window_str}'] = 100 * abs(self.df[f'plus_di_{window_str}'] - self.df[f'minus_di_{window_str}']) / (self.df[f'plus_di_{window_str}'] + self.df[f'minus_di_{window_str}'])
            self.df[f'adx_{window_str}'] = self.df[f'dx_{window_str}'].rolling(window=window, min_periods=1).mean()
            
            # Directional Movement Ratio (DMR)
            self.df[f'dmr_{window_str}'] = self.df[f'plus_di_{window_str}'] / (self.df[f'minus_di_{window_str}'] + 1e-10)
            
            # Volatility-Adjusted DMI
            # Use volatility to normalize the directional movement
            vol = self.df[self.volatility_col].rolling(window=window, min_periods=1).mean()
            self.df[f'vol_adjusted_plus_di_{window_str}'] = self.df[f'plus_di_{window_str}'] / (vol + 1e-10)
            self.df[f'vol_adjusted_minus_di_{window_str}'] = self.df[f'minus_di_{window_str}'] / (vol + 1e-10)
            
            # Store computed features
            self.features.update({
                f'tr_{window_str}': self.df[f'tr_{window_str}'].values,
                f'plus_di_{window_str}': self.df[f'plus_di_{window_str}'].values,
                f'minus_di_{window_str}': self.df[f'minus_di_{window_str}'].values,
                f'adx_{window_str}': self.df[f'adx_{window_str}'].values,
                f'dmr_{window_str}': self.df[f'dmr_{window_str}'].values,
                f'vol_adjusted_plus_di_{window_str}': self.df[f'vol_adjusted_plus_di_{window_str}'].values,
                f'vol_adjusted_minus_di_{window_str}': self.df[f'vol_adjusted_minus_di_{window_str}'].values
            })
            
        # Clean up temporary columns
        cols_to_drop = [col for col in self.df.columns if any(x in col for x in 
                       ['high_', 'low_', 'prev_close_', 'tr1_', 'tr2_', 'tr3_', 
                        'up_move_', 'down_move_', 'plus_dm_', 'minus_dm_', 
                        'smoothed_', 'dx_'])]
        self.df = self.df.drop(columns=cols_to_drop)
            
        return self
        
    def extract_all_features(self, window_sizes=[10, 50, 100]):
        """
        Extract all features and return as numpy array.
        
        Parameters:
        -----------
        window_sizes : list
            List of window sizes for feature calculation
            
        Returns:
        --------
        tuple
            (feature_array, feature_names, dataframe)
        """
        # Store window sizes
        self.window_sizes = window_sizes
        
        # Compute all features
        self._compute_time_features()
        self.compute_microstructure_features(window_sizes)
        self.compute_order_flow_metrics(window_sizes)
        self.compute_dmi_features(window_sizes)
        
        # Add basic features
        self.features.update({
            'price': self.df[self.price_col].values,
            'volume': self.df[self.volume_col].values,
            'volatility': self.df[self.volatility_col].values,
            'return': self.df['return'].values
        })
        
        # Create feature array and names
        feature_names = list(self.features.keys())
        feature_array = np.column_stack([self.features[name] for name in feature_names])
        
        print(f"Extracted {len(feature_names)} features")
        
        return feature_array, feature_names, self.df 