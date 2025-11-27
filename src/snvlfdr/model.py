import numpy as np
import jax.numpy as jnp
import time
from .core import calculate_f0, calculate_f1_batch, calculate_lfdr

class SNVLFDR:
    def __init__(self, pi0_initial=0.95, epsilon=0.01, lfdr_threshold=0.01,
                 bq_threshold=20, mq_threshold=20, af_threshold=0.01, dp_threshold=10,
                 method='empirical', fixed_error=None, batch_size=10000):
        self.pi0 = pi0_initial
        self.epsilon = epsilon
        self.lfdr_threshold = lfdr_threshold
        self.bq_t = bq_threshold
        self.mq_t = mq_threshold
        self.af_t = af_threshold
        self.dp_t = dp_threshold
        self.method = method
        self.fixed_error = fixed_error
        self.batch_size = batch_size
        self.results_ = None
        
    def _prepare_data(self, df):
        counts_jax = jnp.array(np.vstack(df['Counts'].values))
        af_jax = jnp.array(df['AF'].values)
        if self.fixed_error is not None:
            if np.isscalar(self.fixed_error): error_rates = jnp.full(len(df), self.fixed_error)
            else: error_rates = jnp.array(self.fixed_error)
        else:
            error_rates = jnp.array(df['Error'].values)
        return counts_jax, af_jax, error_rates

    def _batched_f1(self, counts, error_rates, thetas):
        n_samples = counts.shape[0]
        results = []
        for i in range(0, n_samples, self.batch_size):
            end = min(i + self.batch_size, n_samples)
            c_batch = counts[i:end]
            e_batch = error_rates[i:end]
            res = calculate_f1_batch(c_batch, e_batch, thetas)
            results.append(np.array(res))
        return jnp.concatenate(results)

    def fit(self, df):
        # 1. Filters (Standard)
        mask_hq = (df['Alt_BQ'] >= self.bq_t) & (df['Ref_BQ'] >= self.bq_t) & (df['Alt_MQ'] >= self.mq_t) & (df['Ref_MQ'] >= self.mq_t)
        mask_edge = (df['Ref_MQ'] == 0) & (df['Ref_BQ'] > 0) & (df['Alt_BQ'] >= self.bq_t) & (df['Alt_MQ'] >= self.mq_t)
        df_clean = df[mask_hq | mask_edge].copy()
        
        # Note: We apply AF filter here for the *training candidate set* if we wanted to match R exactly,
        # but since you pass AF_T=0, we keep everything.
        mask_filters = (df_clean['AF'] >= self.af_t) & (df_clean['Depth'] >= self.dp_t)
        df_final = df_clean[mask_filters].copy()
        
        if len(df_final) == 0: return None
        print(f"\nTraining on {len(df_final)} sites...")

        counts_jax, af_jax, error_rates = self._prepare_data(df_final)
        
        # 2. Initialize Thetas (Robust Initialization)
        np.random.seed(42)
        
        # STRICT CONSTRAINT: Never allow the model to think a mutation can have AF < 0.01.
        # This matches the implicit logic of the R package (which filters inputs).
        MIN_THETA = 0.01
        
        if self.method == 'uniform':
             thetas = jnp.array(np.random.uniform(MIN_THETA, 1, 1000))
        elif self.method == 'uniform_empirical':
             thetas = jnp.array(np.random.uniform(MIN_THETA, 1, 1000))
        else: 
             # Initialize from high-confidence sites only
             # We use 0.05 (5%) to start safe.
             init_candidates = df_final['AF'].values[df_final['AF'].values > 0.05]
             
             if len(init_candidates) > 50:
                 if len(init_candidates) > 1000:
                     thetas = jnp.array(np.random.choice(init_candidates, 1000, replace=False))
                 else:
                     thetas = jnp.array(init_candidates)
             else:
                 thetas = jnp.array(np.random.uniform(0.05, 1.0, 1000))

        # 3. EM Loop
        f0 = calculate_f0(counts_jax, error_rates)
        curr_pi0 = self.pi0
        iteration = 0
        
        while iteration < 100:
            iteration += 1
            
            # Batched E-Step
            f1 = self._batched_f1(counts_jax, error_rates, thetas)
            lfdr = calculate_lfdr(curr_pi0, f0, f1)
            
            # M-Step
            new_pi0 = jnp.sum(lfdr > self.lfdr_threshold) / len(df_final)
            
            if abs(new_pi0 - curr_pi0) < self.epsilon: break
            curr_pi0 = new_pi0
            
            # Update Thetas (Empirical Step) with STRICT FLOOR
            if self.method != 'uniform':
                mutant_mask = lfdr <= self.lfdr_threshold
                possible_thetas = af_jax[mutant_mask]
                
                # --- THE FIX: PREVENT DRIFT ---
                # We strictly ignore any "mutant" candidates that are actually just noise.
                # This ensures the 'g(theta)' distribution never creeps below 1%.
                possible_thetas = possible_thetas[possible_thetas >= MIN_THETA]
                
                if len(possible_thetas) > 0:
                    if len(possible_thetas) > 1000:
                        thetas = possible_thetas[np.random.choice(len(possible_thetas), 1000, replace=False)]
                    else:
                        thetas = possible_thetas
                # If no valid high-AF mutants found, keep previous thetas (don't collapse to 0)

        self.results_ = df_final.copy()
        self.results_['LFDR'] = np.array(lfdr)
        self.results_['Mutant'] = (self.results_['LFDR'] <= self.lfdr_threshold).astype(int)
        self.results_['estimated_pi0'] = float(curr_pi0)
        
        return self.results_
