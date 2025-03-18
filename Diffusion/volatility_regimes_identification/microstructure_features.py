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
        
    def compute_momentum_features(self, window_sizes=[10, 50, 100]):
        """
        Compute momentum features including:
        - Short-term momentum
        - Momentum acceleration
        - RSI (Relative Strength Index)
        
        Parameters:
        -----------
        window_sizes : list
            List of window sizes for feature calculation
        """
        print("Computing momentum features...")
        
        # Calculate returns for different lookback periods
        for window in window_sizes:
            window_str = f'{window}'
            
            # 1. Short-term momentum (price change over window)
            self.df[f'momentum_{window_str}'] = (
                (self.df[self.price_col] - self.df[self.price_col].shift(window)) / 
                self.df[self.price_col].shift(window)
            )
            
            # 2. Momentum acceleration (change in momentum)
            if window > 1:
                self.df[f'momentum_prev_{window_str}'] = (
                    (self.df[self.price_col].shift(1) - self.df[self.price_col].shift(window+1)) / 
                    self.df[self.price_col].shift(window+1)
                )
                
                self.df[f'momentum_acceleration_{window_str}'] = (
                    self.df[f'momentum_{window_str}'] - self.df[f'momentum_prev_{window_str}']
                )
                
                # Clean up temporary column
                self.df = self.df.drop(columns=[f'momentum_prev_{window_str}'])
            
            # 3. RSI (Relative Strength Index)
            # Calculate gains and losses
            self.df[f'price_change_{window_str}'] = self.df[self.price_col].diff(1)
            self.df[f'gain_{window_str}'] = np.where(
                self.df[f'price_change_{window_str}'] > 0, 
                self.df[f'price_change_{window_str}'], 
                0
            )
            self.df[f'loss_{window_str}'] = np.where(
                self.df[f'price_change_{window_str}'] < 0, 
                abs(self.df[f'price_change_{window_str}']), 
                0
            )
            
            # Calculate average gain and loss
            self.df[f'avg_gain_{window_str}'] = self.df[f'gain_{window_str}'].rolling(window=window, min_periods=1).mean()
            self.df[f'avg_loss_{window_str}'] = self.df[f'loss_{window_str}'].rolling(window=window, min_periods=1).mean()
            
            # Calculate RS and RSI
            self.df[f'rs_{window_str}'] = (
                self.df[f'avg_gain_{window_str}'] / self.df[f'avg_loss_{window_str}'].replace(0, 1e-10)
            )
            self.df[f'rsi_{window_str}'] = 100 - (100 / (1 + self.df[f'rs_{window_str}']))
            
            # Store computed features
            self.features.update({
                f'momentum_{window_str}': self.df[f'momentum_{window_str}'].values,
                f'rsi_{window_str}': self.df[f'rsi_{window_str}'].values
            })
            
            if window > 1:
                self.features[f'momentum_acceleration_{window_str}'] = self.df[f'momentum_acceleration_{window_str}'].values
            
        # Clean up temporary columns
        cols_to_drop = [col for col in self.df.columns if any(x in col for x in 
                       ['price_change_', 'gain_', 'loss_', 'avg_gain_', 'avg_loss_', 'rs_'])]
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
        # Compute all features
        self._compute_time_features()
        self.compute_microstructure_features(window_sizes)
        self.compute_order_flow_metrics(window_sizes)
        self.compute_momentum_features(window_sizes)
        
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
        
        # Return features, names, and enhanced dataframe
        return feature_array, feature_names, self.df 