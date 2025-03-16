import numpy as np
import pandas as pd

class MarketMicrostructureSimulator:
    def __init__(self, 
                 n_samples=23401,
                 initial_price=100.0,
                 initial_volatility=0.2,
                 seed=None):
       
        self.n_steps = n_samples - 1  # Number of steps 
        self.initial_price = initial_price
        self.initial_volatility = initial_volatility
        self.rng = np.random.RandomState(seed)
        
        # Parameters for semi-martingale model
        self.drift_param = 0.05  
        self.vol_of_vol = 0.3    
        self.mean_reversion = 5.0  
        self.vol_long_run_mean = 0.2  
        self.vol_drift = 0.0     
        self.jump_intensity = 10.0   
        self.jump_mean = 0.0     
        self.jump_std = 0.01 
        self.vol_of_vol_2 = 0.2      
        self.vol_jump_intensity = 5.0 
        self.vol_jump_mean = 0.0     
        self.vol_jump_std = 0.005    
        
        # Parameters for quote spread noise
        self.base_spread = 0.01  
        self.spread_vol_coef = 0.5  
        self.spread_noise_std = 0.2  
        
        # Parameters for order flow noise (Hawkes process)
        self.hawkes_mu = 0.1     
        self.hawkes_alpha = 0.8  
        self.hawkes_beta = 1.0   
        self.order_size_shape = 1.5  
        self.momentum_impact = 0.3  
        
        # Parameters for strategic order splitting
        self.parent_order_rate = 0.01 
        self.parent_size_mean = 100    
        self.parent_size_std = 50     
        self.twap_ratio = 0.4          
        self.vwap_ratio = 0.4          
        self.iceberg_ratio = 0.2       
        
        # Parameters for strategic quote positioning
        self.quote_revision_threshold = 0.0001  
        self.quote_imbalance_impact = 0.5       
        self.spoof_probability = 0.05           
        self.market_maker_count = 5             
        
        # Index array instead of time grid
        self.index_array = np.arange(n_samples)
        
    def simulate_efficient_price(self):
        """
        Simulate the latent efficient price process using the semi-martingale model.
        dX_t = b_t dt + σ_t dW_t + dJ_t
        dσ_t = κ(θ - σ_t)dt + ξσ_t dW_t^(1) + ξ_2σ_t dW_t^(2) + dJ_t^σ
        """
        X = np.zeros(self.n_steps + 1)
        sigma = np.zeros(self.n_steps + 1)
        
        X[0] = np.log(self.initial_price)
        sigma[0] = self.initial_volatility
        
        # Using small constant scaling factor instead of dt
        scaling_factor = 0.0001
        
        # Simulate Brownian motion increments for price and volatility
        dW1 = self.rng.normal(0, np.sqrt(scaling_factor), self.n_steps)  # Price Brownian
        dW2 = self.rng.normal(0, np.sqrt(scaling_factor), self.n_steps)  # Independent vol Brownian
        
        # Generate price jumps
        price_jump_prob = self.jump_intensity * scaling_factor
        price_jump_times = self.rng.binomial(1, price_jump_prob, self.n_steps)
        price_jumps = price_jump_times * self.rng.normal(self.jump_mean, self.jump_std, self.n_steps)
        
        # Generate volatility jumps
        vol_jump_prob = self.vol_jump_intensity * scaling_factor  # New parameter needed
        vol_jump_times = self.rng.binomial(1, vol_jump_prob, self.n_steps)
        vol_jumps = vol_jump_times * self.rng.normal(self.vol_jump_mean, self.vol_jump_std, self.n_steps)  # New parameters needed
        
        # Simulate path
        for i in range(self.n_steps):
            # Volatility process with two Brownian motions
            dsigma = (self.mean_reversion * (self.vol_long_run_mean - sigma[i]) * scaling_factor + 
                    self.vol_of_vol * sigma[i] * dW1[i] +  
                    self.vol_of_vol_2 * sigma[i] * dW2[i] +  
                    vol_jumps[i])  
            sigma[i+1] = max(1e-6, sigma[i] + dsigma)
            
            # Price process
            dX = (self.drift_param * scaling_factor + 
                sigma[i] * dW1[i] + 
                price_jumps[i])
            X[i+1] = X[i] + dX
        
        # Convert log-price to price
        price = np.exp(X)
        
        return {
            'index': self.index_array,
            'log_price': X,
            'price': price,
            'volatility': sigma,
            'price_jumps': price_jumps,
            'vol_jumps': vol_jumps
        }
    
    def add_quote_spread_noise(self, efficient_price_data):
        """
        Adjust bid-ask prices by adding spread sensitivity to volatility and random noise.
        """
        price = efficient_price_data['price']
        volatility = efficient_price_data['volatility']
        
        spread_pct = self.base_spread * (1 + self.spread_vol_coef * volatility)
        spread = spread_pct * price
        
        bid_noise = self.rng.normal(0, self.spread_noise_std, len(price))
        ask_noise = self.rng.normal(0, self.spread_noise_std, len(price))
        
        bid = price - spread / 2 * (1 + bid_noise)
        ask = price + spread / 2 * (1 + ask_noise)
        
        # Identify cases where bid >= ask (crossed market scenario, which is invalid)
        cross_mask = bid >= ask
        mid = (bid + ask) / 2  
        new_half_spread = spread / 2  
        
        # Adjust bid and ask prices to ensure bid < ask 
        bid = np.where(cross_mask, mid - new_half_spread, bid)
        ask = np.where(cross_mask, mid + new_half_spread, ask)
        
        # Instead of intraday seasonality use a random seasonality factor based on index position
        # Using a combination of sine waves of different frequencies
        n = len(price)
        x = np.linspace(0, 10, n)  
        seasonality = (1 + 0.2 * np.sin(x) + 
                        0.1 * np.sin(2 * x) + 
                        0.05 * np.sin(5 * x))
        
        bid = price - (price - bid) * seasonality
        ask = price + (ask - price) * seasonality
        
        return {
            'bid': bid,
            'ask': ask,
            'mid': (bid + ask) / 2,
            'spread': ask - bid,
            'spread_pct': spread_pct
        }
    
    def simulate_hawkes_process(self):
        """
        Hawkes >> Poisson as the market has self-exciting behaviour
        Simulate order arrival using a Hawkes process.
        """
        intensity = np.zeros(self.n_steps + 1)
        events = np.zeros(self.n_steps + 1, dtype=bool)
        
        intensity[0] = self.hawkes_mu
        
        decay_factor = 0.95  # Constant decay per tick
        
        for i in range(self.n_steps):
            # Generate event with current intensity
            if self.rng.rand() < intensity[i]:
                events[i] = True
            
            # Update intensity
            intensity[i+1] = (self.hawkes_mu + 
                             intensity[i] * decay_factor + 
                             (events[i] * self.hawkes_alpha * decay_factor))
        
        event_indices = self.index_array[events]
        return event_indices, events
    
    def add_order_flow_noise(self, efficient_price_data):
        """
        order flow noise simulation for tick data
        """
        price = efficient_price_data['price']
        log_price = efficient_price_data['log_price']
        volatility = efficient_price_data['volatility']
    
        _, buy_events = self.simulate_hawkes_process()
        _, sell_events = self.simulate_hawkes_process()

        # Momentum(strength of trend) calculation using rolling window
        lookback = 5
        with np.errstate(divide='ignore', invalid='ignore'):
            price_momentum = np.zeros_like(log_price)
            price_momentum[lookback:] = (log_price[lookback:] - log_price[:-lookback]) / \
                                    np.maximum(volatility[lookback:] * np.sqrt(lookback * 0.0001), 1e-8)
        price_momentum = np.nan_to_num(price_momentum, nan=0.0, posinf=0.0, neginf=0.0)
        momentum_factor = 0.5 + self.momentum_impact * np.tanh(price_momentum)

        rand_vals = self.rng.random(self.n_steps + 1)
        
        both_mask = buy_events & sell_events
        only_buy_mask = buy_events & ~sell_events
        only_sell_mask = sell_events & ~buy_events

        # Resolve conflicts where both buy and sell events occur
        keep_buy = rand_vals[both_mask] < momentum_factor[both_mask]
        sell_events[both_mask] = ~keep_buy
        buy_events[both_mask] = keep_buy

        # Determening whether a buy or sell event occurs based on the momentum factor 
        # If the momentum is not strong, introducing randomness by flipping buy orders to sell orders and vice versa. 
        flip_to_sell = rand_vals[only_buy_mask] > momentum_factor[only_buy_mask]
        buy_events[only_buy_mask] = ~flip_to_sell
        sell_events[only_buy_mask] = flip_to_sell

        flip_to_buy = rand_vals[only_sell_mask] < momentum_factor[only_sell_mask]
        sell_events[only_sell_mask] = ~flip_to_buy
        buy_events[only_sell_mask] = flip_to_buy

        # Generate order sizes using vectorized power law distribution
        all_events = buy_events | sell_events
        event_indices = np.where(all_events)[0]
        
        order_sizes = np.zeros(self.n_steps + 1)
        if event_indices.size > 0:
            u = self.rng.uniform(size=event_indices.size)
            size_mins = np.ones_like(event_indices)
            size_scales = np.full_like(event_indices, 10.0)
            
            order_sizes[event_indices] = size_mins * (1 - u) ** (-1/self.order_size_shape)
            order_sizes[event_indices] = np.minimum(order_sizes[event_indices], size_scales * 100)

        # Create direction array using vector operations
        order_directions = np.zeros(self.n_steps + 1, dtype='U4')
        order_directions[buy_events] = 'buy'
        order_directions[sell_events] = 'sell'

        valid_events = all_events.nonzero()[0]
        if valid_events.size == 0:
            return pd.DataFrame(columns=['index', 'direction', 'size', 'price'])
        
        orders_df = pd.DataFrame({
            'index': self.index_array[valid_events],
            'direction': order_directions[valid_events],
            'size': order_sizes[valid_events],
            'price': price[valid_events]
        })
        
        return orders_df
    
    def add_strategic_order_splitting(self, efficient_price_data, quote_data):
        """
        Simulate strategic order splitting by generating parent orders and breaking them into child orders.
        """
        
        parent_order_events = self.rng.binomial(1, self.parent_order_rate, self.n_steps + 1)
        parent_order_indices = np.where(parent_order_events)[0]
        
        all_child_orders = []
        
        for idx in parent_order_indices:
            # Parent order characteristics
            parent_size = max(10, self.rng.normal(self.parent_size_mean, self.parent_size_std))
            parent_direction = 'buy' if self.rng.rand() < 0.5 else 'sell'
            
            # Determine execution strategy
            strategy_rand = self.rng.rand()
            if strategy_rand < self.twap_ratio:
                strategy = 'TWAP'
            elif strategy_rand < self.twap_ratio + self.vwap_ratio:
                strategy = 'VWAP'
            else:
                strategy = 'Iceberg'
            
            # Generate child orders based on strategy
            child_orders = []
            
            if strategy == 'TWAP':
                # Time-weighted average price strategy
                execution_range = min(self.rng.randint(10, 100), self.n_steps - idx)
                n_child_orders = min(self.rng.randint(5, 20), execution_range)
                child_indices = sorted(self.rng.choice(range(idx, idx + execution_range), n_child_orders, replace=False))
                
                remaining_size = parent_size
                for i, child_idx in enumerate(child_indices):
                    if i == len(child_indices) - 1:
                        child_size = remaining_size
                    else:
                        slice_pct = self.rng.uniform(0.1, 0.3)
                        child_size = max(1, round(slice_pct * remaining_size))
                        remaining_size -= child_size
                    
                    child_price = quote_data['bid'][child_idx] if parent_direction == 'sell' else quote_data['ask'][child_idx]
                    
                    child_orders.append({
                        'parent_id': idx,
                        'index': self.index_array[child_idx],
                        'direction': parent_direction,
                        'size': child_size,
                        'price': child_price,
                        'strategy': strategy
                    })
            
            elif strategy == 'VWAP':
                # Volume-weighted average price strategy
                execution_range = min(self.rng.randint(10, 100), self.n_steps - idx)
                n_child_orders = min(self.rng.randint(5, 20), execution_range)
                
                positions = np.linspace(0, 1, execution_range)
                weight = 1 + 0.5 * np.exp(-((positions - 0.2) ** 2) / 0.05) + 0.5 * np.exp(-((positions - 0.8) ** 2) / 0.05)
                weight /= weight.sum()
                
                child_indices = sorted(self.rng.choice(
                    range(idx, idx + execution_range), 
                    size=n_child_orders, 
                    p=weight, 
                    replace=False
                ))
                
                remaining_size = parent_size
                for i, child_idx in enumerate(child_indices):
                    if i == len(child_indices) - 1:
                        child_size = remaining_size
                    else:
                        position_weight = weight[child_idx - idx] / weight[child_indices - idx].sum()
                        child_size = max(1, round(position_weight * parent_size))
                        child_size = min(child_size, remaining_size)
                        remaining_size -= child_size
                    
                    child_price = quote_data['bid'][child_idx] if parent_direction == 'sell' else quote_data['ask'][child_idx]
                    
                    child_orders.append({
                        'parent_id': idx,
                        'index': self.index_array[child_idx],
                        'direction': parent_direction,
                        'size': child_size,
                        'price': child_price,
                        'strategy': strategy
                    })
            
            else: 
                # Iceberg order strategy - show only a small portion of order at a time
                visible_size = max(1, round(parent_size * self.rng.uniform(0.05, 0.2)))
                remaining_size = parent_size
                
                child_idx = idx
                while remaining_size > 0 and child_idx < self.n_steps:
                    child_size = min(visible_size, remaining_size)
                    remaining_size -= child_size
                    
                    child_price = quote_data['bid'][child_idx] if parent_direction == 'sell' else quote_data['ask'][child_idx]
                    
                    child_orders.append({
                        'parent_id': idx,
                        'index': self.index_array[child_idx],
                        'direction': parent_direction,
                        'size': child_size,
                        'price': child_price,
                        'strategy': strategy
                    })
                    
                    # Random wait ticks until next slice
                    wait_ticks = self.rng.randint(1, 10)
                    child_idx += wait_ticks
            
            all_child_orders.extend(child_orders)
        
        return pd.DataFrame(all_child_orders)
    
    def add_strategic_quote_positioning(self, efficient_price_data, quote_data):
        """
        Simulate strategic quote positioning by market makers.
        """
        price = efficient_price_data['price']
        base_bid = quote_data['bid']
        base_ask = quote_data['ask']
        
        # Increase threshold sensitivity and ensure more revision points
        price_prev = np.roll(price, 1)
        price_prev[0] = price[0]
        price_change = np.abs(price - price_prev) / np.maximum(price_prev, 1e-8)
        
        # More frequent revisions by lowering threshold and adding periodic revisions
        periodic_revision = np.zeros_like(price, dtype=bool)
        periodic_revision[::20] = True  # Revise every 20 ticks
        need_revision = (price_change >= self.quote_revision_threshold * 0.1) | periodic_revision | (np.arange(len(price)) == 0)
        revision_indices = np.where(need_revision)[0]

        self.market_maker_count = max(10, self.market_maker_count)  
        mm_params = [(
            self.rng.uniform(1.0, 2.0),  
            self.rng.uniform(0.8, 1.2),
            self.rng.uniform(0.1, self.spoof_probability * 3)  
        ) for _ in range(self.market_maker_count)]

        all_quotes = []
        for mm_id, (mm_aggression, mm_risk_aversion, mm_spoofing_tendency) in enumerate(mm_params):
            rev_indices = self.index_array[revision_indices]
            current_prices = price[revision_indices]
            
            bid_offsets = self.rng.normal(0, 0.001*current_prices)  
            ask_offsets = self.rng.normal(0, 0.001*current_prices)  
            mm_bids = base_bid[revision_indices] + bid_offsets
            mm_asks = base_ask[revision_indices] + ask_offsets


            # Spoofing decisions
            spoof_decisions = self.rng.rand(len(revision_indices)) < mm_spoofing_tendency
            
            # Track state between revisions
            prev_bid = prev_ask = None
            current_quotes = []
            
            for i, idx in enumerate(revision_indices):
                current_index = self.index_array[idx]
                
                # Cancel previous quotes
                if prev_bid is not None:
                    current_quotes.extend([
                        {'index': current_index, 'action': 'cancel', 'side': 'bid',
                         'price': prev_bid, 'mm_id': mm_id, 'is_spoof': False},
                        {'index': current_index, 'action': 'cancel', 'side': 'ask',
                         'price': prev_ask, 'mm_id': mm_id, 'is_spoof': False}
                    ])

                # Place new quotes
                new_bid, new_ask = mm_bids[i], mm_asks[i]
                is_spoof = spoof_decisions[i]
                
                current_quotes.extend([
                    {'index': current_index, 'action': 'place', 'side': 'bid',
                     'price': new_bid, 'mm_id': mm_id, 'is_spoof': is_spoof},
                    {'index': current_index, 'action': 'place', 'side': 'ask',
                     'price': new_ask, 'mm_id': mm_id, 'is_spoof': is_spoof}
                ])

                # Handle spoof cancellations
                if is_spoof and idx < self.n_steps:
                    cancel_delay = self.rng.randint(1, 5)
                    cancel_idx = min(idx + cancel_delay, self.n_steps)
                    cancel_index = self.index_array[cancel_idx]
                    
                    current_quotes.extend([
                        {'index': cancel_index, 'action': 'cancel', 'side': 'bid',
                         'price': new_bid, 'mm_id': mm_id, 'is_spoof': True},
                        {'index': cancel_index, 'action': 'cancel', 'side': 'ask',
                         'price': new_ask, 'mm_id': mm_id, 'is_spoof': True}
                    ])
                    prev_bid = prev_ask = None
                else:
                    prev_bid, prev_ask = new_bid, new_ask

            all_quotes.extend(current_quotes)

        return pd.DataFrame(all_quotes)
    
    def simulate_full_microstructure(self):
        """
        Run the full simulation pipeline for tick data.
        """
        print("Simulating efficient price process...")
        efficient_price_data = self.simulate_efficient_price()
        
        print("Adding quote spread noise...")
        quote_data = self.add_quote_spread_noise(efficient_price_data)
        
        print("Simulating order flow noise...")
        order_flow_data = self.add_order_flow_noise(efficient_price_data)
        
        print("Simulating strategic order splitting...")
        strategic_orders = self.add_strategic_order_splitting(efficient_price_data, quote_data)
        
        print("Simulating strategic quote positioning...")
        strategic_quotes = self.add_strategic_quote_positioning(efficient_price_data, quote_data)
        
        result = {
            'efficient_price': efficient_price_data,
            'quotes': quote_data,
            'order_flow': order_flow_data,
            'strategic_orders': strategic_orders,
            'strategic_quotes': strategic_quotes
        }
        
        return result
    
    def extract_noise_components(self, simulation_result):
        efficient_price = simulation_result['efficient_price']['price']
        observed_mid = simulation_result['quotes']['mid']
        
        total_noise = observed_mid - efficient_price
        total_std = np.std(total_noise)
        
        raw_spread = simulation_result['quotes']['spread'] / 2
        quote_spread_noise = (raw_spread - np.mean(raw_spread))
        quote_spread_noise = quote_spread_noise * (0.4 * total_std / np.std(quote_spread_noise))
        
        order_flow_noise = np.zeros_like(efficient_price)
        if len(simulation_result['order_flow']) > 0:
            for _, order in simulation_result['order_flow'].iterrows():
                idx = np.searchsorted(self.index_array, order['index'])
                idx = min(idx, len(order_flow_noise) - 1) 
                direction_mult = 1 if order['direction'] == 'buy' else -1
                impact = direction_mult * order['size'] * 0.00002 * efficient_price[idx]
                order_flow_noise[idx:] += impact
        
        strategic_order_noise = np.zeros_like(efficient_price)
        if len(simulation_result['strategic_orders']) > 0:
            for _, order in simulation_result['strategic_orders'].iterrows():
                idx = np.searchsorted(self.index_array, order['index'])
                idx = min(idx, len(strategic_order_noise) - 1)  
                direction_mult = 1 if order['direction'] == 'buy' else -1
                impact = direction_mult * order['size'] * 0.00002 * efficient_price[idx]
                strategic_order_noise[idx:] += impact
        
        quote_positioning_noise = np.zeros_like(efficient_price)
        if len(simulation_result['strategic_quotes']) > 0:
            spoof_quotes = simulation_result['strategic_quotes'][simulation_result['strategic_quotes']['is_spoof'] == True]
            active_impacts = {}
            
            for _, quote in spoof_quotes.iterrows():
                idx = np.searchsorted(self.index_array, quote['index'])
                idx = min(idx, len(quote_positioning_noise) - 1)
                
                quote_id = f"{quote['mm_id']}_{quote['side']}_{quote['price']}"
                
                if quote['action'] == 'place':
                    direction_mult = 1 if quote['side'] == 'ask' else -1
                    price_deviation = abs(quote['price'] - efficient_price[idx]) / efficient_price[idx]
                    impact = direction_mult * 0.1 * efficient_price[idx] * (1 + price_deviation)  # Significantly increased
                    active_impacts[quote_id] = (idx, impact)
                    quote_positioning_noise[idx:] += impact
                    
                elif quote['action'] == 'cancel' and quote_id in active_impacts:
                    start_idx, impact = active_impacts[quote_id]
                    quote_positioning_noise[idx:] -= impact
                    del active_impacts[quote_id]
        
        def apply_decay_and_scale(noise, decay_rate, target_std_ratio):
            decayed = np.zeros_like(noise)
            for i in range(len(noise)):
                if noise[i] != 0:
                    decay = np.exp(-decay_rate * np.arange(len(noise) - i))
                    decayed[i:] += noise[i] * decay[:len(noise)-i]
            
            if np.std(decayed) > 0:
                return (decayed - np.mean(decayed)) * (target_std_ratio * total_std / np.std(decayed))
            return decayed
        
        order_flow_noise = apply_decay_and_scale(order_flow_noise, decay_rate=0.1, target_std_ratio=0.3)
        strategic_order_noise = apply_decay_and_scale(strategic_order_noise, decay_rate=0.1, target_std_ratio=0.2)
        quote_positioning_noise = apply_decay_and_scale(quote_positioning_noise, decay_rate=0.2, target_std_ratio=0.1)
        
        explained_noise = (quote_spread_noise + 
                        order_flow_noise + 
                        strategic_order_noise + 
                        quote_positioning_noise)
        
        explained_noise = (explained_noise - np.mean(explained_noise)) * (total_std / np.std(explained_noise))
        explained_noise += np.mean(total_noise)
        
        residual_noise = total_noise - explained_noise

        return {
            'total_noise': total_noise,
            'quote_spread_noise': quote_spread_noise,
            'order_flow_noise': order_flow_noise,
            'strategic_order_noise': strategic_order_noise,
            'quote_positioning_noise': quote_positioning_noise,
            'residual_noise': residual_noise
        }